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

import collections
import io

from collections.abc import Iterable

import cloudpickle
import torch

from horovod.torch.mpi_ops import allgather, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import rank, size
from horovod.torch.mpi_ops import ProcessSet, global_process_set


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the ``model.state_dict()``,
    ``model.named_parameters()``, or ``model.parameters()``.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    from horovod.torch.optimizer import DistributedOptimizer
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        #  the entire state_dict, as its structure is deeply nested and contains
        #  None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad and id(p) not in state_dict['state']:
                    p.grad = p.data.new(p.size()).zero_()
                    if isinstance(optimizer, torch.optim.SparseAdam):
                        p.grad = p.grad.to_sparse()

        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    scalars = {}
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we place the scalars into a single dict,
    # then pickle and broadcast with broadcast_object (under the assumption
    # that there are not many scalars, and so the overhead of pickling will
    # be relatively low). Because broadcast_obect is performed out-of-place,
    # we then use a callback to assign the new value to the correct element
    # of the optimizer state.
    def _create_state_callback(pid, name):
        def _assign_state(v):
            state_dict['state'][pid][name] = v
        return _assign_state

    def _create_option_callback(index, option_key):
        def _assign_option(v):
            optimizer.param_groups[index][option_key] = v
        return _assign_option

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be broadcast separately
            key = '%s.%d' % (option_key, index)
            scalars[key] = option_value
            callbacks[key] = _create_option_callback(index, option_key)

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            if pid not in state_dict['state']:
                # The param has not set requires_grad, so skip broadcast
                continue

            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if torch.is_tensor(p):
                    # Tensor -> use broadcast_parameters
                    params.append((key, p))
                else:
                    # Scalar -> use broadcast_object
                    scalars[key] = p
                    callbacks[key] = _create_state_callback(pid, name)

    # Synchronized broadcast of all tensor parameters
    broadcast_parameters(params, root_rank)

    # Broadcast and cleanup for non-tensor parameters
    scalars = broadcast_object(scalars, root_rank)
    for key, p in scalars.items():
        callbacks[key](p)


def broadcast_object(obj, root_rank=0, name=None, process_set=global_process_set):
    """
    Serializes and broadcasts an object from root rank to all other processes.
    Typical usage is to broadcast the `optimizer.state_dict()`, for example:

    .. code-block:: python

        state_dict = broadcast_object(optimizer.state_dict(), 0)
        if hvd.rank() > 0:
            optimizer.load_state_dict(state_dict)

    Arguments:
        obj: An object capable of being serialized without losing any context.
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
        name: Optional name to use during broadcast, will default to the class
              type.
        process_set: Process set object to limit this operation to a subset of
                     Horovod processes. Default is the global process set.
    Returns:
        The object that was broadcast from the `root_rank`.
    """
    if name is None:
        name = type(obj).__name__

    if rank() == root_rank:
        b = io.BytesIO()
        cloudpickle.dump(obj, b)
        t = torch.ByteTensor(bytearray(b.getvalue()))
        sz = torch.IntTensor([t.shape[0]])
        broadcast_(sz, root_rank, name + '.sz', process_set)
    else:
        sz = torch.IntTensor([0])
        broadcast_(sz, root_rank, name + '.sz', process_set)
        t = torch.ByteTensor(sz.tolist()[0])

    broadcast_(t, root_rank, name + '.t', process_set)

    if rank() != root_rank:
        buf = io.BytesIO(t.numpy().tobytes())
        obj = cloudpickle.load(buf)

    return obj


def allgather_object(obj, name=None):
    """
    Serializes and allgathers an object from all other processes.

    Arguments:
        obj: An object capable of being serialized without losing any context.
        name: Optional name to use during allgather, will default to the class
              type.

    Returns:
        The list of objects that were allgathered across all ranks.
    """
    if name is None:
        name = type(obj).__name__

    def load(byte_array):
        buf = io.BytesIO(byte_array.tobytes())
        return cloudpickle.load(buf)

    b = io.BytesIO()
    cloudpickle.dump(obj, b)

    t = torch.ByteTensor(bytearray(b.getvalue()))
    sz = torch.IntTensor([t.shape[0]])

    sizes = allgather(sz, name=name + '.sz').numpy()
    gathered = allgather(t, name=name + '.t').numpy()

    def select(i):
        start = sum(sizes[:i])
        end = start + sizes[i]
        return gathered[start:end]

    return [load(select(i)) for i in range(size())]
