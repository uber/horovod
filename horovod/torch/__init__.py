# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright Microsoft
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

from horovod.common.util import check_extension

_MPI_LIB_AVAILABLE = True
try:
    check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib_v2')
except:
    try:
        check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib', '_mpi_lib')
    except Exception as e:
        # MPI libs are missing, but python applications are still avaiable.
        print(e)
        print("Warning! MPI libs are missing, but python applications are still avaiable.")
        _MPI_LIB_AVAILABLE = False

# only import following function when mpi is available.
if _MPI_LIB_AVAILABLE:
    from horovod.torch import elastic
    from horovod.torch.compression import Compression
    from horovod.torch.functions import allgather_object, broadcast_object, broadcast_optimizer_state, broadcast_parameters
    from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
    from horovod.torch.mpi_ops import grouped_allreduce, grouped_allreduce_async, grouped_allreduce_, grouped_allreduce_async_
    from horovod.torch.mpi_ops import sparse_allreduce_async
    from horovod.torch.mpi_ops import allgather, allgather_async
    from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
    from horovod.torch.mpi_ops import alltoall, alltoall_async
    from horovod.torch.mpi_ops import join
    from horovod.torch.mpi_ops import poll, synchronize
    from horovod.torch.mpi_ops import init, shutdown
    from horovod.torch.mpi_ops import is_initialized, start_timeline, stop_timeline
    from horovod.torch.mpi_ops import size, local_size, cross_size, rank, local_rank, cross_rank
    from horovod.torch.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
    from horovod.torch.mpi_ops import gloo_enabled, gloo_built
    from horovod.torch.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
    from horovod.torch.mpi_ops import ProcessSet, global_process_set, add_process_set, remove_process_set
    from horovod.torch.mpi_ops import Average, Sum, Adasum
    from horovod.torch.optimizer import DistributedOptimizer
    from horovod.torch.sync_batch_norm import SyncBatchNorm

# Please run this function in a subprocess
def _check_has_gpu():
    import torch
    return torch.cuda.is_available()
