// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef HOROVOD_NCCL_OPERATIONS_H
#define HOROVOD_NCCL_OPERATIONS_H

#include <nccl.h>

#include "../mpi_context.h"
#include "cuda_operations.h"

namespace horovod {
namespace common {

struct NCCLContext {
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;

  void ErrorCheck(std::string op_name, ncclResult_t nccl_result);

  virtual void ShutDown();
};

// subclass of NCCLContext, but will add another nccl comm for end thread
struct ParallelNCCLContext : public NCCLContext {
  // the nccl comm used for end thread
  std::unordered_map<std::vector<int32_t>, ncclComm_t> end_nccl_comms;

  void ShutDown() override;
}

// a multi-thread cuda context, will add a new stream for end-thread
struct ParallelCUDAContext : public CUDAContext {
  // cuda sream used for end thread
  std::unordered_map<int, cudaStream_t> end_streams;
}

class NCCLAllreduce : public CUDAAllreduce {
public:
  NCCLAllreduce(NCCLContext* nccl_context, MPIContext* mpi_context,
                CUDAContext* cuda_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

protected:
  void InitNCCLComm(const std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& nccl_device_map);

  virtual void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                        Communicator& nccl_id_bcast_comm);

  NCCLContext* nccl_context_;
  ncclComm_t* nccl_comm_;

  MPIContext* mpi_context_;
};

class NCCLHierarchicalAllreduce : public NCCLAllreduce {
public:
  NCCLHierarchicalAllreduce(NCCLContext* nccl_context, MPIContext* mpi_context,
                            CUDAContext* cuda_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                Communicator& nccl_id_bcast_comm) override;
};

// like NCCLHierarchicalAllreduce, but use 3 thread to do allreduce
class ParallelNCCLHierarchicalAllreduce : public NCCLHierarchicalAllreduce {
public:
  ParallelNCCLHierarchicalAllreduce(ParallelNCCLContext *parallel_nccl_context, 
                                    MPIContext *parallel_mpi_context,
                                    ParallelCUDAContext *parallel_cuda_context, 
                                    HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  // init the parallelnccl_commm
  void InitParallelNCCLComm(const std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& nccl_device_map);

  // init parallel cuda context
  void InitParallelCUDA(const std::vector<TensorTableEntry>& entries);

private:
  // in parallel alleduce it will use 3 thread, the main thread (horovod background thread), mpi thread and end thread
  // mpi_queue used for do MPI all reduce
  // end_queue used for do "end task" (include, copy data back to GPU, ncclAllGather, ncclBcast and copy data back to tensor)
  HorovodSingleQueue mpi_queue_;
  HorovodSingleQueue end_queue_;

  // special context used for parallel allreduce
  // for seperatd from NCCLHierarchicalAllreduce, use prefix "parallel"
  ParallelNCCLContext parallel_nccl_context_;  
  ParallelCUDAContext parallel_cuda_context_;

  ncclComm_t* end_nccl_comm_;
  cudaStream_t* end_stream_;
}

} // namespace common
} // namespace horovod

#endif //HOROVOD_NCCL_OPERATIONS_H
