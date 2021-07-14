// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef HOROVOD_MXNET_MPI_OPS_H
#define HOROVOD_MXNET_MPI_OPS_H

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/c_api_error.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>

#include "adapter.h"
#include "tensor_util.h"

namespace horovod {
namespace mxnet {

using namespace horovod::common;

typedef ::mxnet::NDArray NDArray;
typedef ::mxnet::Engine::CallbackOnComplete CallbackOnComplete;
typedef Request::RequestType OperationType;
typedef std::shared_ptr<MXTensor> MXTensorSharedPtr;
typedef std::shared_ptr<NDArray> NDArraySharedPtr;

struct MpiOpsParam {
  std::vector<NDArraySharedPtr> input_tensors;
  std::vector<NDArraySharedPtr> output_tensors;
  std::vector<NDArray*> outputs;
  std::vector<NDArraySharedPtr> cpu_input_tensors;
  std::vector<NDArraySharedPtr> cpu_output_tensors;
  OperationType op_type;
  std::vector<std::string> op_names;
  int root_rank;
  NDArraySharedPtr splits_tensor;
  NDArraySharedPtr received_splits_tensor;
  bool average;
  double prescale_factor;
  double postscale_factor;
  int process_set_id;
  int del_count = 0;

  MpiOpsParam(std::vector<NDArraySharedPtr>&& input_tensors,
              std::vector<NDArraySharedPtr>&& output_tensors,
              std::vector<NDArray*>&& outputs,
              const std::vector<NDArraySharedPtr>& cpu_input_tensors,
              const std::vector<NDArraySharedPtr>& cpu_output_tensors,
              const OperationType& op_type,
              std::vector<std::string>&& op_names,
              int root_rank, bool average,
              NDArraySharedPtr splits_tensor,
              NDArraySharedPtr received_splits_tensor,
              double prescale_factor,
              double postscale_factor,
              int process_set_id)
      : input_tensors(std::move(input_tensors)),
        output_tensors(std::move(output_tensors)),
        outputs(std::move(outputs)),
        cpu_input_tensors(cpu_input_tensors),
        cpu_output_tensors(cpu_output_tensors),
        op_type(op_type),
        op_names(std::move(op_names)),
        root_rank(root_rank),
        splits_tensor(splits_tensor),
        received_splits_tensor(received_splits_tensor),
        average(average),
        prescale_factor(prescale_factor),
        postscale_factor(postscale_factor),
        process_set_id(process_set_id) {
  }
};

inline MpiOpsParam* CreateMpiOpsParam(std::vector<NDArraySharedPtr>&& input_tensors,
                                      std::vector<NDArraySharedPtr>&& output_tensors,
                                      std::vector<NDArray*>&& outputs,
                                      const std::vector<NDArraySharedPtr>& cpu_input_tensors,
                                      const std::vector<NDArraySharedPtr>& cpu_output_tensors,
                                      const OperationType& op_type,
                                      std::vector<std::string>&& op_names,
                                      int root_rank, bool average,
                                      NDArraySharedPtr splits_tensor,
                                      NDArraySharedPtr received_splits_tensor,
                                      double prescale_factor,
                                      double postscale_factor,
                                      int process_set_id) {
  return new MpiOpsParam(
      std::move(input_tensors), std::move(output_tensors), std::move(outputs),
      cpu_input_tensors, cpu_output_tensors, op_type, std::move(op_names),
      root_rank, average, splits_tensor, received_splits_tensor,
      prescale_factor, postscale_factor, process_set_id);
}

void DeleteMpiOpsParam(void* param) {
  auto ops_param = static_cast<MpiOpsParam*>(param);
  delete ops_param;
}

extern "C" int horovod_mxnet_allreduce_async(NDArray* const * inputs,
                                             NDArray* const * outputs,
                                             const char* name, bool average,
                                             int priority,
                                             double prescale_factor,
                                             double postscale_factor,
                                             int num_tensors,
                                             int process_set_id);
extern "C" int horovod_mxnet_allgather_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int priority,
                                             int process_set_id);
extern "C" int horovod_mxnet_broadcast_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int root_rank,
                                             int priority,
                                             int process_set_id);
extern "C" int horovod_mxnet_alltoall_async(NDArray* input,
                                            NDArray* output,
                                            const char* name,
                                            NDArray* splits,
                                            NDArray* output_received_splits,
                                            int priority,
                                            int process_set_id);

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_MPI_OPS_H
