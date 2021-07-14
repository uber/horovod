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

import os
import unittest
import warnings

import mock

from elastic_common import BaseElasticTests


class ElasticTorchTests(BaseElasticTests, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        training_script = os.path.join(os.path.dirname(__file__), 'data/elastic_torch_main.py')
        super(ElasticTorchTests, self).__init__(training_script, *args, **kwargs)
        warnings.simplefilter('module')

    @mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.runner.gloo_run._get_min_start_hosts', return_value=1)
    def test_all_hosts_blacklisted(self, mock_get_min_start_hosts):
        self.skipTest('This test fails due to https://github.com/horovod/horovod/issues/2030')

    @mock.patch('horovod.runner.elastic.driver.ELASTIC_TIMEOUT_SECS', 1)
    @mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.runner.gloo_run._get_min_start_hosts', return_value=1)
    def test_min_hosts_timeout(self, mock_get_min_start_hosts):
        self.skipTest('This test fails due to https://github.com/horovod/horovod/issues/2030')

