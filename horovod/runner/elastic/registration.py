# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import threading

from collections import defaultdict

from horovod.runner.elastic import constants

READY = 'READY'
SUCCESS = 'SUCCESS'
FAILURE = 'FAILURE'


class WorkerStateRegistry(object):
    def __init__(self, driver, host_manager, reset_limit=None, verbose=False):
        self._driver = driver
        self._host_manager = host_manager
        self._reset_limit = reset_limit
        self._reset_count = 0
        self._lock = threading.Lock()
        self._states = {}
        self._workers = defaultdict(set)
        self._barrier = None
        self._rendezvous_id = 0
        self._verbose = verbose
        self._size = 0
        self._action_event = threading.Event()
        self._action_event.set()

    def get_recorded_slots(self):
        return self._states.keys()

    def get(self, state):
        return self._workers[state]

    def count(self, state):
        return len(self._workers[state])

    def reset(self, size):
        with self._lock:
            logging.info('reset workers: {}'.format(size))
            self._states.clear()
            self._workers.clear()
            self._barrier = threading.Barrier(parties=size, action=self._action)
            self._rendezvous_id += 1
            self._size = size

    def size(self):
        return self._size

    def last_rendezvous(self):
        return self._rendezvous_id

    def record_ready(self, host, slot):
        return self._record_state(host, slot, READY)

    def record_success(self, host, slot):
        return self._record_state(host, slot, SUCCESS)

    def record_failure(self, host, slot):
        return self._record_state(host, slot, FAILURE)

    def _record_state(self, host, slot, state):
        if self._driver.finished():
            logging.info('driver finished, ignoring registration: {}[{}] = {}'.format(host, slot, state))
            return self._rendezvous_id

        if self._host_manager.is_blacklisted(host):
            logging.warning('host registers state %s but is already blacklisted, ignoring: %s', state, host)
            return self._rendezvous_id

        # we should wait for _action finished if a _record_state called when _action is running
        self._action_event.wait()

        key = (host, slot)
        with self._lock:
            if key in self._states:
                if state == FAILURE:
                    # Worker originally recorded itself as READY, but the worker failed while waiting at the barrier. As
                    # such, we need to update the state to FAILURE, and we don't want two threads coming from the same
                    # worker at the barrier.
                    #
                    # In order to ensure that the new failing thread can record results in cases of total job failure,
                    # we also need to block this thread by waiting on the barrier. This requires us to reset the barrier,
                    # as otherwise this worker will be double-counted (once for the READY thread and once for FAILURE),
                    # which would cause the barrier to complete too early.
                    logging.info('key exists, reset barrier: {}[{}] = {} -> {}'
                                 .format(host, slot, self._states[key], state))
                    self._barrier.reset()
                else:
                    logging.error('key exists and new state %s not FAILURE, '
                                  'ignoring (current state is %s)', state, self._states[key])

            if key not in self._states or state == FAILURE:
                logging.info('record state: {}[{}] = {}'.format(host, slot, state))
                self._states[key] = state
                self._workers[state].add(key)

            rendezvous_id = self._rendezvous_id

        rendezvous_id = self._wait(key, state, rendezvous_id)
        return rendezvous_id

    def _wait(self, key, state, rendezvous_id):
        while True:
            try:
                self._barrier.wait()
                return rendezvous_id
            except threading.BrokenBarrierError:
                if self._barrier.broken:
                    # Timeout or other non-recoverable error, so exit
                    raise

                # Barrier has been reset
                with self._lock:
                    # Check to make sure the reset was not caused by a change of state for this key
                    rendezvous_id = self._rendezvous_id
                    saved_state = self._states.get(key, state)
                    if saved_state != state:
                        # This worker changed its state, so do not attempt to wait again to avoid double-counting
                        raise RuntimeError('State {} overridden by {}'.format(state, saved_state))

    def _action(self):
        self._action_event.clear()
        self._on_workers_recorded()
        self._action_event.set()

    def _on_workers_recorded(self):
        logging.info('all {} workers recorded'.format(self.size()))

        # Check for success state, if any process succeeded, shutdown all other processes
        if self.count(SUCCESS) > 0:
            logging.info('success count == {} -> stop running'.format(self.count(SUCCESS)))
            self._driver.stop()
            return

        # Check that all processes failed, indicating that processing should stop
        if self.count(FAILURE) == self._size:
            logging.error('failure count == {} -> stop running'.format(self._size))
            self._driver.stop()
            return

        # Check for failures, and add them to the blacklisted hosts list
        failures = self.get(FAILURE)
        for host, slot in failures:
            self._host_manager.blacklist(host)

        # If every active host is blacklisted, then treat this as job failure
        if all([self._host_manager.is_blacklisted(host) for host, slot in self.get_recorded_slots()]):
            logging.error('blacklisted slots count == {} -> stop running'.format(self._size))
            self._driver.stop()
            return

        # Check that we have already reset the maximum number of allowed times
        if self._reset_limit is not None and self._reset_count >= self._reset_limit:
            logging.error('reset count {} has exceeded limit {} -> stop running'
                          .format(self._reset_count, self._reset_limit))
            self._driver.stop(error_message=constants.RESET_LIMIT_EXCEEDED_MESSAGE.format(self._reset_limit))
            return

        try:
            self._reset_count += 1
            self._driver.resume()
        except Exception:
            logging.exception('failed to activate new hosts -> stop running')
            self._driver.stop()
