// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#if HAVE_CUDA
#include <nccl.h>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 7, 0)
#define NCCL_P2P_SUPPORTED
#endif
#elif HAVE_ROCM
#include <rccl.h>
#endif

#if HAVE_MPI
#include "../mpi/mpi_context.h"
#endif

#include "gpu_operations.h"

#include <functional>

namespace horovod {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor);

struct NCCLContext {
  std::vector<std::unordered_map<std::vector<int32_t>, ncclComm_t>> nccl_comms;

  void ErrorCheck(std::string op_name, ncclResult_t nccl_result, ncclComm_t& nccl_comm);

  void ShutDown();
};

class NCCLOpContext {
public:
  NCCLOpContext(NCCLContext* nccl_context, HorovodGlobalState* global_state,
                horovod::common::Communicator communicator_type)
      : nccl_comm_(nullptr),
        error_check_callback_(std::bind(&NCCLOpContext::AsyncErrorCheck, this)),
        nccl_context_(nccl_context),
        global_state_(global_state),
        communicator_type_(communicator_type){};

  void InitNCCLComm(const std::vector<TensorTableEntry>& entries,
                    const std::vector<int32_t>& nccl_device_map);

  void AsyncErrorCheck();

  ncclComm_t* nccl_comm_;
  std::function<void()> error_check_callback_;

private:
  void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                Communicator& nccl_id_bcast_comm);

  NCCLContext* nccl_context_;
  HorovodGlobalState* global_state_;
  horovod::common::Communicator communicator_type_;
};

class NCCLAllreduce : public GPUAllreduce {
public:
  NCCLAllreduce(NCCLContext* nccl_context, GPUContext* gpu_context,
                HorovodGlobalState* global_state,
                horovod::common::Communicator communicator_type = Communicator::GLOBAL)
      : GPUAllreduce(gpu_context, global_state),
        nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, communicator_type),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

class NCCLBroadcast : public GPUBroadcast {
public:
  NCCLBroadcast(NCCLContext* nccl_context, GPUContext* gpu_context,
                HorovodGlobalState* global_state)
      : GPUBroadcast(gpu_context, global_state),
        nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

class NCCLAlltoall : public GPUAlltoall {
public:
  NCCLAlltoall(NCCLContext* nccl_context, GPUContext* gpu_context,
               HorovodGlobalState* global_state)
      : GPUAlltoall(gpu_context, global_state),
        nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

#if HAVE_MPI
class NCCLHierarchicalAllreduce : public NCCLAllreduce {
public:
  NCCLHierarchicalAllreduce(NCCLContext* nccl_context, MPIContext* mpi_context,
                            GPUContext* gpu_context,
                            HorovodGlobalState* global_state)
      : NCCLAllreduce(nccl_context, gpu_context, global_state, Communicator::LOCAL),
        mpi_context_(mpi_context){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

private:
  MPIContext* mpi_context_;
};
#endif

class NCCLAllgather : public GPUAllgather {
public:
  NCCLAllgather(NCCLContext* nccl_context, GPUContext* gpu_context,
                  HorovodGlobalState* global_state)
      : GPUAllgather(gpu_context, global_state),
        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};


} // namespace common
} // namespace horovod

#endif // HOROVOD_NCCL_OPERATIONS_H
