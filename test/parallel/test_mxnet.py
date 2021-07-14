# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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
import sys
import itertools
import unittest
from distutils.version import LooseVersion

import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import skip_or_fail_gpu_test

try:
    import mxnet as mx
    from mxnet.base import MXNetError
    from mxnet.test_utils import almost_equal, same
    import horovod.mxnet as hvd

    has_gpu = mx.context.num_gpus() > 0

    ccl_supported_types = set(['int32', 'int64', 'float32', 'float64'])

    # MXNet 1.4.x will kill test MPI process if error occurs during operation enqueue. Skip
    # those tests for versions earlier than 1.5.0.
    _skip_enqueue_errors = LooseVersion(mx.__version__) < LooseVersion('1.5.0')

    HAS_MXNET = True
except ImportError:
    has_gpu = False
    _skip_enqueue_errors = False
    HAS_MXNET = False


@pytest.mark.skipif(not HAS_MXNET, reason='MXNet unavailable')
class MXTests(unittest.TestCase):
    """
    Tests for ops in horovod.mxnet.
    """

    def _current_context(self):
        if has_gpu:
            return mx.gpu(hvd.local_rank())
        else:
            return mx.current_context()

    def filter_supported_types(self, types):
        if 'CCL_ROOT' in os.environ:
           types = [t for t in types if t in ccl_supported_types]
        return types

    def test_gpu_required(self):
        if not has_gpu:
            skip_or_fail_gpu_test(self, "No GPUs available")

    def test_horovod_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            # MXNet uses gpu_id as part of the seed, so to get identical seeds
            # we must set a context.
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            summed = hvd.allreduce(tensor, average=False, name=str(count))
            multiplied = tensor * size
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert almost_equal(summed.asnumpy(), multiplied.asnumpy(), atol=threshold), \
                f'hvd.allreduce produces incorrect results: {hvd.rank()} {count} {dtype} {dim}'

    def test_horovod_allreduce_average(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            averaged = hvd.allreduce(tensor, average=True, name=str(count))
            tensor *= size
            tensor /= size
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 1
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert almost_equal(averaged.asnumpy(), tensor.asnumpy(), atol=threshold), \
                f'hvd.allreduce produces incorrect results for average: {hvd.rank()} {count} {dtype} {dim}'

    def test_horovod_allreduce_inplace(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            multiplied = tensor * size
            hvd.allreduce_(tensor, average=False, name=str(count))
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert almost_equal(tensor.asnumpy(), multiplied.asnumpy(), atol=threshold), \
                f'hvd.allreduce produces incorrect results for self: {hvd.rank()} {count} {dtype} {dim}'

    def test_horovod_allreduce_prescale(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors with prescaling."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float16', 'float32', 'float64'])
        int_types = ['int32', 'int64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 1
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            np.random.seed(1234)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            factor = np.random.uniform()
            scaled = hvd.allreduce(tensor, average=False, name=str(count),
                                   prescale_factor=factor)

            factor = mx.nd.array([factor], dtype='float64', ctx=ctx)
            if ctx != mx.cpu() and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
                # For integer types, scaling done in FP64
                factor = factor.astype('float64' if dtype in int_types else dtype)
                tensor = tensor.astype('float64' if dtype in int_types else dtype)
            else:
                # For integer types, scaling done in FP64, FP32 math for FP16 on CPU
                factor = factor.astype('float32' if dtype == 'float16' else
                                       'float64' if dtype in int_types else dtype)
                tensor = tensor.astype('float32' if dtype == 'float16' else
                                       'float64' if dtype in int_types else dtype)

            expected = factor * tensor
            expected = expected.astype(dtype)
            expected *= size
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert almost_equal(expected.asnumpy(), scaled.asnumpy(), atol=threshold), \
                f'hvd.allreduce produces incorrect results for prescaling: {hvd.rank()} {count} {dtype} {dim}'

    def test_horovod_allreduce_postscale(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors with postscaling."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float16', 'float32', 'float64'])
        int_types = ['int32', 'int64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 1
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            np.random.seed(1234)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            factor = np.random.uniform()
            scaled = hvd.allreduce(tensor, average=False, name=str(count),
                                   postscale_factor=factor)

            factor = mx.nd.array([factor], dtype='float64', ctx=ctx)
            if ctx != mx.cpu() and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
                # For integer types, scaling done in FP64
                factor = factor.astype('float64' if dtype in int_types else dtype)
                tensor = tensor.astype('float64' if dtype in int_types else dtype)
            else:
                # For integer types, scaling done in FP64, FP32 math for FP16 on CPU
                factor = factor.astype('float32' if dtype == 'float16' else
                                       'float64' if dtype in int_types else dtype)
                tensor = tensor.astype('float32' if dtype == 'float16' else
                                       'float64' if dtype in int_types else dtype)

            expected = tensor * size
            expected *= factor
            expected = expected.astype(dtype)
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert almost_equal(expected.asnumpy(), scaled.asnumpy(), atol=threshold), \
                f'hvd.allreduce produces incorrect results for pre/post scaling: {hvd.rank()} {count} {dtype} {dim}'

    def test_horovod_allreduce_error(self):
        """Test that the allreduce raises an error if different ranks try to
           send tensors of different rank or dimension."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Same rank, different dimension
        ctx = self._current_context()

        shape = (17 + rank, 3)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw error'
        except (MXNetError, RuntimeError):
            pass

        # Same number of elements, different rank
        if rank == 0:
            shape = (17, 23 * 57)
        else:
            shape = (17, 23, 57)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_allreduce_type_error(self):
        """Test that the allreduce raises an error if different ranks try to
           send tensors of different type."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        ctx = self._current_context()
        shape = (17, 3)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        if rank % 2 == 0:
            tensor = tensor.astype('int32')
        else:
            tensor = tensor.astype('float32')

        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw error'
        except (MXNetError, RuntimeError):
            pass

    @unittest.skipUnless(has_gpu, "no gpu detected")
    def test_horovod_allreduce_cpu_gpu_error(self):
        """Test that the allreduce raises an error if different ranks try to
           perform reduction on CPU and GPU."""
        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        shape = (17, 17, 17)
        if rank % 2 == 0:
            ctx = mx.gpu(hvd.rank())
        else:
            ctx = mx.cpu(hvd.rank())
        tensor = mx.nd.ones(shape=shape, ctx=ctx)

        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw cpu-gpu error'
        except (MXNetError, RuntimeError):
            pass


    def test_horovod_allreduce_ndarray_lifetime(self):
        """Test that the input NDArray remains valid during async allreduce"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for i, dim in enumerate(dims):
            tensor = mx.nd.ones(shape=shapes[dim], ctx=ctx)
            # tensor*(i+1) result will be destroyed immediately after this call
            # See https://github.com/horovod/horovod/issues/1533
            sum = hvd.allreduce(tensor * (i + 1), average=False)
            expected = tensor * (i + 1) * size
            assert same(sum.asnumpy(), expected.asnumpy())

    def test_horovod_grouped_allreduce(self):
        """Test that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 1
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)

            tensors = [mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx) for _ in range(5)]

            tensors = [tensor.astype(dtype) for tensor in tensors]

            multiplied = [tensor * size for tensor in tensors]

            summed = hvd.grouped_allreduce(tensors, average=False, name=str(count))

            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([almost_equal(t1.asnumpy(), t2.asnumpy(), atol=threshold)
                for t1, t2 in zip(summed, multiplied)]), \
                f'hvd.grouped_allreduce produces incorrect results: {hvd.rank()} {count} {dtype} {dim}'

    def test_horovod_grouped_allreduce_average(self):
        """Test that the grouped allreduce correctly averages 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 1
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)

            tensors = [mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx) for _ in range(5)]

            tensors = [tensor.astype(dtype) for tensor in tensors]
            tensors = [tensor * size for tensor in tensors]
            tensors = [tensor / size for tensor in tensors]

            averaged = hvd.grouped_allreduce(tensors, average=True, name=str(count))

            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([almost_equal(t1.asnumpy(), t2.asnumpy(), atol=threshold)
                for t1, t2 in zip(averaged, tensors)]), \
                f'hvd.grouped_allreduce produces incorrect results for average: {hvd.rank()} {count} {dtype} {dim}'

    def test_horovod_grouped_allreduce_inplace(self):
        """Test that the in-place grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 1
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)

            tensors = [mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx) for _ in range(5)]

            tensors = [tensor.astype(dtype) for tensor in tensors]

            multiplied = [tensor * size for tensor in tensors]

            hvd.grouped_allreduce_(tensors, average=False, name=str(count))

            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            assert all([almost_equal(t1.asnumpy(), t2.asnumpy(), atol=threshold)
                for t1, t2 in zip(tensors, multiplied)]), \
                f'hvd.grouped_allreduce_ produces incorrect results: {hvd.rank()} {count} {dtype} {dim}'

    @unittest.skipUnless(has_gpu, "no gpu detected")
    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_grouped_allreduce_cpu_gpu_error(self):
        """Test that the grouped allreduce raises an error if the input tensor
           list contains a mix of tensors on CPU and GPU."""
        hvd.init()
        local_rank = hvd.local_rank()
        tensors = [mx.nd.ones(shape=[10], ctx=mx.gpu(local_rank) if i % 2
                   else mx.cpu(local_rank)) for i in range(5)]

        try:
            outputs = hvd.grouped_allreduce(tensors)
            mx.nd.waitall()
            assert False, 'hvd.grouped_allreduce did not throw cpu-gpu error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims,
                                                       root_ranks):
            tensor = mx.nd.ones(shapes[dim], ctx=ctx) * rank
            root_tensor = mx.nd.ones(shapes[dim], ctx=ctx) * root_rank
            tensor = tensor.astype(dtype)
            root_tensor = root_tensor.astype(dtype)

            broadcast_tensor = hvd.broadcast(tensor, root_rank=root_rank,
                                             name=str(count))
            if rank != root_rank:
                if same(tensor.asnumpy(), root_tensor.asnumpy()):
                    print("broadcast", count, dtype, dim,
                          mx.nd.max(tensor == root_tensor))
                    print("tensor", hvd.rank(), tensor)
                    print("root_tensor", hvd.rank(), root_tensor)
                    print("comparison", hvd.rank(), tensor == root_tensor)
                assert not same(tensor.asnumpy(), root_tensor.asnumpy()), \
                    'hvd.broadcast modifies source tensor'
            if not same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()):
                print("broadcast", count, dtype, dim)
                print("broadcast_tensor", hvd.rank(), broadcast_tensor)
                print("root_tensor", hvd.rank(), root_tensor)
                print("comparison", hvd.rank(),
                      broadcast_tensor == root_tensor)
            assert same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()), \
                'hvd.broadcast produces incorrect broadcasted tensor'
            count += 1

    def test_horovod_broadcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims,
                                                       root_ranks):
            tensor = mx.nd.ones(shapes[dim], ctx=ctx) * rank
            root_tensor = mx.nd.ones(shapes[dim], ctx=ctx) * root_rank
            tensor = tensor.astype(dtype)
            root_tensor = root_tensor.astype(dtype)

            # Only do broadcasting using broadcast_tensor
            broadcast_tensor = tensor.copy()
            hvd.broadcast_(broadcast_tensor, root_rank=root_rank,
                           name=str(count))
            if rank != root_rank:
                if same(tensor.asnumpy(), root_tensor.asnumpy()):
                    print("broadcast", count, dtype, dim,
                          mx.nd.max(tensor == root_tensor))
                    print("tensor", hvd.rank(), tensor)
                    print("root_tensor", hvd.rank(), root_tensor)
                    print("comparison", hvd.rank(), tensor == root_tensor)
                assert not same(tensor.asnumpy(), root_tensor.asnumpy()), \
                    'hvd.broadcast modifies source tensor'
            if not same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()):
                print("broadcast", count, dtype, dim)
                print("broadcast_tensor", hvd.rank(), broadcast_tensor)
                print("root_tensor", hvd.rank(), root_tensor)
                print("comparison", hvd.rank(),
                      broadcast_tensor == root_tensor)
            assert same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()), \
                'hvd.broadcast produces incorrect broadcasted tensor'
            count += 1

    def test_horovod_broadcast_parameters(self):
        """Test the correctness of broadcast_parameters."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        root_rank = 1
        tensor_dict = {}
        root_dict = {}
        for dtype, dim, in itertools.product(dtypes, dims):
            tensor_dict[count] = mx.nd.ones(shapes[dim], ctx=ctx) * rank
            root_dict[count] = mx.nd.ones(shapes[dim], ctx=ctx) * root_rank
            tensor_dict[count] = tensor_dict[count].astype(dtype)
            root_dict[count] = root_dict[count].astype(dtype)
            count += 1

        hvd.broadcast_parameters(tensor_dict, root_rank=root_rank)
        for i in range(count):
            if not same(tensor_dict[i].asnumpy(), root_dict[i].asnumpy()):
                print("broadcast", i, dtypes[i], dims[i])
                print("broadcast_tensor", hvd.rank(), tensor_dict[i])
                print("root_tensor", hvd.rank(), root_dict[i])
                print("comparison", hvd.rank(), tensor_dict[i] == root_dict[i])
            assert same(tensor_dict[i].asnumpy(), root_dict[i].asnumpy()), \
                'hvd.broadcast_parameters produces incorrect broadcasted tensor'

    def test_horovod_broadcast_error(self):
        """Test that the broadcast returns an error if any dimension besides
           the first is different among the tensors being broadcasted."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        ctx = self._current_context()
        shape = (17, rank+1)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)

        try:
            output = hvd.broadcast(tensor, 0)
            output.wait_to_read()
            assert False, 'hvd.broadcast did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast_type_error(self):
        """Test that the broadcast returns an error if the types being broadcasted
           differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        ctx = self._current_context()
        shape = (17, 3)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        if rank % 2 == 0:
            tensor = tensor.astype('int32')
        else:
            tensor = tensor.astype('float32')

        try:
            output = hvd.broadcast(tensor, 0)
            output.wait_to_read()
            assert False, 'hvd.broadcast did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast_rank_error(self):
        """Test that the broadcast returns an error if different ranks
           specify different root rank."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        ctx = self._current_context()
        shape = (17, 17, 17)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        try:
            output = hvd.broadcast(tensor, root_rank=rank)
            output.wait_to_read()
            assert False, 'hvd.broadcast did not throw rank error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast_deferred_init_parameters(self):
        """Test that the deferred initialized parameters are broadcasted."""
        hvd.init()
        root_rank = 0
        rank = hvd.rank()

        # This test does not apply if there is only one worker.
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        mx.random.seed(rank)
        layer = mx.gluon.nn.Conv2D(10, 2)
        layer.initialize()
        hvd.broadcast_parameters(layer.collect_params(), root_rank=root_rank)

        x = mx.nd.ones((5, 4, 10, 10))
        layer(x)
        tensors = [p.data() for _, p in sorted(layer.collect_params().items())]
        root_tensors = []
        for tensor in tensors:
            root_tensors.append(hvd.broadcast(tensor, root_rank=root_rank))

        for tensor, root_tensor in zip(tensors, root_tensors):
            assert same(tensor.asnumpy(), root_tensor.asnumpy()), \
                'horovod did not broadcast deferred initialized parameter correctly'

    def test_horovod_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = mx.ndarray.ones(shape=[17] * dim, dtype=dtype, ctx=ctx) * rank
            gathered = hvd.allgather(tensor)

            assert list(gathered.shape) == [17 * size] + [17] * (dim - 1)

            for i in range(size):
                rank_tensor = gathered[i * 17:(i + 1) * 17]
                assert list(rank_tensor.shape) == [17] * dim, \
                    'hvd.allgather produces incorrect gathered shape'
                assert rank_tensor.min() == i, 'hvd.allgather produces incorrect gathered tensor'
                assert rank_tensor.max() == i, 'hvd.allgather produces incorrect gathered tensor'

    def test_horovod_allgather_variable_size(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = mx.ndarray.ones(
                shape=[tensor_sizes[rank]] + [17] * (dim - 1), dtype=dtype, ctx=ctx) * rank

            gathered = hvd.allgather(tensor)

            expected_size = sum(tensor_sizes)
            assert list(gathered.shape) == [expected_size] + [17] * (dim - 1)

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = gathered[sum(
                    tensor_sizes[:i]):sum(tensor_sizes[:i + 1])]
                assert list(rank_tensor.shape) == rank_size
                assert rank_tensor.min() == i
                assert rank_tensor.max() == i

    def test_horovod_allgather_error(self):
        """Test that the allgather returns an error if any dimension besides
        the first is different among the tensors being gathered."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        ctx = self._current_context()

        tensor_size = [17] * 3
        tensor_size[1] = 10 * (rank + 1)
        tensor = mx.ndarray.ones(shape=tensor_size, ctx=ctx)

        try:
            hvd.allgather(tensor)
            assert False, 'hvd.allgather did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_allgather_type_error(self):
        """Test that the allgather returns an error if the types being gathered
        differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        ctx = self._current_context()

        tensor_size = [17] * 3
        if rank % 2 == 0:
            tensor = mx.ndarray.ones(shape=tensor_size, dtype="int32", ctx=ctx)
        else:
            tensor = mx.ndarray.ones(shape=tensor_size, dtype="float32", ctx=ctx)

        try:
            hvd.allgather(tensor)
            assert False, 'hvd.allgather did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_broadcast_object(self):
        hvd.init()

        expected_obj = {
            'hello': 123,
            0: [1, 2]
        }
        obj = expected_obj if hvd.rank() == 0 else {}

        obj = hvd.broadcast_object(obj, root_rank=0)
        self.assertDictEqual(obj, expected_obj)

        # To prevent premature shutdown from rank 0 for this test
        mx.nd.waitall()

    def test_allgather_object(self):
        hvd.init()

        d = {'metric_val_1': hvd.rank()}
        if hvd.rank() == 1:
            d['metric_val_2'] = 42

        results = hvd.allgather_object(d)

        expected = [{'metric_val_1': i} for i in range(hvd.size())]
        if hvd.size() > 1:
            expected[1] = {'metric_val_1': 1, 'metric_val_2': 42}

        self.assertEqual(len(results), hvd.size())
        self.assertListEqual(results, expected)

        # To prevent premature shutdown from rank 0 for this test
        mx.nd.waitall()

    def test_horovod_alltoall(self):
        """Test that the alltoall correctly distributes 1D, 2D, and 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1,2,3]
        ctx = self._current_context()
        for dtype, dim in itertools.product(dtypes, dims):
            vals = []
            for i in range(size):
              vals += [i] * (rank + 1)

            tensor = mx.ndarray.array(vals, dtype=dtype, ctx=ctx)
            for _ in range(dim - 1):
              tensor = mx.ndarray.expand_dims(tensor, axis=1)
              tensor = mx.ndarray.concat(tensor, tensor, dim=1)

            splits = mx.ndarray.array([rank + 1] * size, dtype='int32', ctx=ctx)
            collected, received_splits = hvd.alltoall(tensor, splits)

            assert collected.min() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.max() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.size == size * (size + 1) // 2 * 2**(dim - 1), 'hvd.alltoall collected wrong number of values'
            self.assertSequenceEqual(received_splits.asnumpy().tolist(), [rk + 1 for rk in range(size)],
                                     "hvd.alltoall returned incorrect received_splits")


    def test_horovod_alltoall_equal_split(self):
        """Test that the alltoall correctly distributes 1D tensors with default splitting."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1,2,3]
        ctx = self._current_context()
        for dtype, dim in itertools.product(dtypes, dims):
            vals = []
            for i in range(size):
              vals += [i] * (rank + 1)

            tensor = mx.ndarray.array(vals, dtype=dtype, ctx=ctx)
            for _ in range(dim - 1):
              tensor = mx.ndarray.expand_dims(tensor, axis=1)
              tensor = mx.ndarray.concat(tensor, tensor, dim=1)
            collected = hvd.alltoall(tensor)

            assert collected.min() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.max() == rank, 'hvd.alltoall produces incorrect collected tensor'
            assert collected.size == size * (size + 1) // 2 * 2**(dim - 1), 'hvd.alltoall collected wrong number of values'

    def test_horovod_alltoall_type_error(self):
        """Test that the alltoall returns an error if the tensor types differ
           across the processes."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        ctx = self._current_context()
        if rank % 2:
          tensor = mx.ndarray.empty([size], dtype='int32', ctx=ctx)
        else:
          tensor = mx.ndarray.empty([size], dtype='float32', ctx=ctx)

        try:
            output = hvd.alltoall(tensor)
            output.wait_to_read()
            assert False, 'hvd.alltoall did not throw error'
        except (MXNetError, RuntimeError):
            pass

    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_alltoall_equal_split_length_error(self):
        """Test that the alltoall with default splitting returns an error if the first dimension
        of tensor is not a multiple of the number of workers."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        ctx = self._current_context()
        tensor = mx.ndarray.empty([size + 1], ctx=ctx)
        try:
            hvd.alltoall(tensor)
            assert False, 'hvd.alltoall did not throw error'
        except (MXNetError, RuntimeError):
            pass

    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_alltoall_splits_error(self):
        """Test that the alltoall returns an error if the sum of the splits entries exceeds
        the first dimension of the input tensor."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        ctx = self._current_context()
        tensor = mx.ndarray.empty([size-1], ctx=ctx)
        splits = mx.ndarray.ones([size], dtype='int32', ctx=ctx)
        try:
            hvd.alltoall(tensor, splits)
            assert False, 'hvd.alltoall did not throw error'
        except (MXNetError, RuntimeError):
            pass

    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_alltoall_splits_type_error(self):
        """Test that the alltoall returns an error if the splits tensor does not
           contain 32-bit integers."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        ctx = self._current_context()
        tensor = mx.ndarray.empty([size], ctx=ctx)
        splits = mx.ndarray.ones([size], dtype='float32', ctx=ctx)
        try:
            hvd.alltoall(tensor, splits)
            assert False, 'hvd.alltoall did not throw error'
        except (MXNetError, ValueError):
            pass

    def test_two_trainer(self):
        """Test using horovod allreduce in MXNet Gluon trainer."""
        from mxnet import gluon
        from mxnet.gluon import Block, nn, HybridBlock

        hvd.init()
        rank = hvd.rank()
        ctx = mx.cpu(rank)

        net1 = nn.Dense(20, in_units=10)
        net2 = nn.Dense(30, in_units=10)
        net1.initialize(ctx=ctx)
        net2.initialize(ctx=ctx)

        params1 = net1.collect_params()
        params2 = net2.collect_params()
        hvd.broadcast_parameters(params1, prefix="net1")
        hvd.broadcast_parameters(params2, prefix="net2")
        trainer1 = hvd.DistributedTrainer(params1, 'sgd', {'learning_rate': 0.1}, prefix="net1")
        trainer2 = hvd.DistributedTrainer(params2, 'sgd', {'learning_rate': 0.1}, prefix="net2")

        for i in range(10):
            data = mx.nd.ones((5, 10), ctx=ctx)
            with mx.autograd.record():
                pred1 = net1(data).sum()
                pred2 = net2(data).sum()
            mx.autograd.backward([pred1, pred2])
            trainer1.step(1.0)
            trainer2.step(1.0)
            l = pred1.asscalar() + pred2.asscalar()

    def test_horovod_alltoall_rank_error(self):
        """Test that the alltoall returns an error if any dimension besides
        the first is different among the tensors being processed."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        ctx = self._current_context()

        tensor_size = [2 * size] * 3
        tensor_size[1] = 10 * (rank + 1)
        tensor = mx.ndarray.ones(shape=tensor_size, ctx=ctx)

        try:
            output = hvd.alltoall(tensor)
            output.wait_to_read()
            assert False, 'hvd.alltoall did not throw error'
        except (MXNetError, RuntimeError):
            pass


    @unittest.skipUnless(has_gpu, "no gpu detected")
    def test_gluon_trainer(self):
        """Test using horovod allreduce in MXNet Gluon trainer."""
        from mxnet import gluon
        from mxnet.gluon import Block, nn, HybridBlock

        hvd.init()
        rank = hvd.rank()
        np.random.seed(1000 + 10 * rank)
        mx.random.seed(1000 + 10 * rank)
        ctx = mx.gpu(rank)

        def gen_random_dataset(batch_size=64, dim=32, min_len=20, max_len=100,
                               size=1000):
            for _ in range(size):
                length = np.random.randint(min_len, max_len + 1)
                rand_src = mx.nd.random.normal(0, 1, (length, dim))
                rand_dst = mx.nd.random.normal(0, 1, (length, dim))
                yield rand_src, rand_dst

        class SimpleNet(HybridBlock):
            def __init__(self, layer_num=6, **kwargs):
                super(SimpleNet, self).__init__(**kwargs)
                self._layer_num = layer_num
                self.ln_l = nn.HybridSequential()
                self.dense_l = nn.HybridSequential()
                for i in range(layer_num):
                    self.dense_l.add(nn.Dense(units=32 + layer_num - 1 - i,
                        flatten=False))
                    self.ln_l.add(nn.LayerNorm())

            def hybrid_forward(self, F, data):
                """
                Parameters
                ----------
                data :
                    Shape (batch_size, seq_len, fea_dim)
                Returns
                -------
                out :
                    Shape (batch_size, seq_len, fea_dim)
                """
                for i in range(self._layer_num):
                   data = self.ln_l[i](data)
                   data = self.dense_l[i](data)
                return data

        net = SimpleNet()
        net.initialize(ctx=ctx)
        net.hybridize(static_alloc=True)

        params = net.collect_params()
        cnt = 0
        lr = 1E-4
        trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr},
            update_on_kvstore=False)

        data_gen = gen_random_dataset()
        for (src_data, dst_data) in data_gen:
            src_data = src_data.as_in_context(ctx).astype(np.float32)
            dst_data = dst_data.as_in_context(ctx).astype(np.float32)
            with mx.autograd.record():
                pred = net(src_data)
                loss = mx.nd.abs(pred - dst_data).mean()
                loss.backward()
            # Begin to update the parameter
            trainer.step(1.0)
            cnt += 1
            l = loss.asscalar()
            if cnt >= 10:
                for key, param in params.items():
                    hvd.allreduce_(param.list_data()[0])
                cnt = 0

    def test_compression_fp16(self):
        valid_dtypes = ['float16', 'float32', 'float64']
        invalid_dtypes = ['uint8', 'int8', 'int32', 'int64']

        tensor_size = (17, 3)
        compression = hvd.Compression.fp16

        for dtype in valid_dtypes:
            tensor = mx.nd.ones(shape=tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, np.float16)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, tensor.dtype)

            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - tensor_decompressed.asnumpy())
            self.assertLess(err, 0.00000001)

        for dtype in invalid_dtypes:
            tensor = mx.nd.ones(shape=tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, tensor.dtype)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, tensor.dtype)

            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - tensor_decompressed.asnumpy())
            self.assertLess(err, 0.00000001)


if __name__ == '__main__':
    unittest.main()