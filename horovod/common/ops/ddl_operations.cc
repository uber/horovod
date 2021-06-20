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

#include "ddl_operations.h"
#include "../logging.h"

namespace horovod {
namespace common {

DDL_Type GetDDLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
    case HOROVOD_FLOAT32:
      return DDL_TYPE_FLOAT;
    case HOROVOD_FLOAT16:
      return DDL_TYPE_HALF;
    default:
      throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                             " is not supported in DDL mode.");
  }
}

DDLAllreduce::DDLAllreduce(DDLContext* ddl_context,
                           GPUContext* gpu_context,
                           HorovodGlobalState* global_state)
    : GPUAllreduce(gpu_context, global_state),
      ddl_context_(ddl_context) {}

Status DDLAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto& timeline = global_state_->timeline;
  if (ddl_context_->ddl_local_device_id != first_entry.device) {
    throw std::logic_error("DDL does not support more than one GPU device per process.");
  }

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    if (timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_IN_FUSION_BUFFER, *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  int64_t num_elements = buffer_len / DataType_Size(first_entry.tensor->dtype());

  if (response.prescale_factor() != 1.0) {
    // Execute prescaling op
    ScaleBuffer(response.prescale_factor(), entries, fused_input_data, buffer_data, num_elements);
    fused_input_data = buffer_data; // for unfused, scale is done out of place
  }

  // Do allreduce.
  if (entries.size() == 1) {
    // Copy input buffer content to output buffer
    // because DDL only supports in-place allreduce
    gpu_context_->MemcpyAsyncD2D(buffer_data, fused_input_data, buffer_len, *gpu_op_context_.stream);
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_IN_FUSION_BUFFER, *gpu_op_context_.stream);
  }

  // Synchronize.
  gpu_context_->WaitForEvents(gpu_op_context_.event_queue, entries, timeline, nullptr, global_state_->elastic_enabled);

  DDL_Type ddl_data_type = GetDDLDataType(first_entry.tensor);
  auto ddl_result = ddl_allreduce(buffer_data, (size_t) num_elements, ddl_data_type,
                                  DDL_OP_SUM);
  if (ddl_result != DDL_SUCCESS) {
    throw std::logic_error("ddl_allreduce failed.");
  }

  if (response.postscale_factor() != 1.0) {
    // Execute postscaling op
    ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data, num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);

    if (timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_OUT_FUSION_BUFFER, *gpu_op_context_.stream);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(entries);
}

void DDLAllreduce::DDLInit(DDLContext* ddl_context, GPUContext* gpu_context) {
  LOG(WARNING) << "DDL backend has been deprecated. Please, start using the NCCL backend by "
                  "building Horovod with 'HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL'.";
  auto ddl_options = std::getenv("DDL_OPTIONS");
  if (ddl_options == nullptr) {
    throw std::logic_error("DDL_OPTIONS env variable needs to be set to use DDL.");
  }
  auto ddl_result = ddl_init(ddl_options);
  if (ddl_result != DDL_SUCCESS) {
    throw std::logic_error("ddl_init failed.");
  }
  ddl_context->ddl_local_device_id = gpu_context->GetDevice();
}

} // namespace common
} // namespace horovod
