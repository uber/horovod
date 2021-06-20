// Copyright 2019 Microsoft. All Rights Reserved.
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

#include "adasum_mpi_operations.h"

namespace horovod {
namespace common {
AdasumMPIAllreduceOp::AdasumMPIAllreduceOp(MPIContext* mpi_context,
                                           HorovodGlobalState* global_state)
    : AdasumMPI(mpi_context, global_state), AllreduceOp(global_state) {}

bool AdasumMPIAllreduceOp::Enabled(const ParameterManager& param_manager,
                                   const std::vector<TensorTableEntry>& entries,
                                   const Response& response) const {
  return true;
}

Status AdasumMPIAllreduceOp::Execute(std::vector<TensorTableEntry>& entries,
                                     const Response& response) {
  if (entries.empty()) {
    return Status::OK();
  }

  WaitForData(entries);

  // Lazily initialize reduction communicators for VHDD algorithm when Adasum reduction is actually called.
  if (!reduction_comms_initialized) {
    InitializeVHDDReductionComms();
  }

  auto& first_entry = entries[0];

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*)first_entry.output->data();
    buffer_len = (size_t)first_entry.output->size();
    if (first_entry.tensor->data() != first_entry.output->data()) {
      std::memcpy(buffer_data, (void*)first_entry.tensor->data(), buffer_len);
    }
    fused_input_data = buffer_data;
  }

  std::vector<int> tensor_counts;
  int64_t num_elements = 0;
  for (auto& e : entries) {
    tensor_counts.push_back(e.tensor->shape().num_elements());
    num_elements += e.tensor->shape().num_elements();
  }

  if (response.prescale_factor() != 1.0) {
    // Execute prescaling op
    ScaleBuffer(response.prescale_factor(), entries, fused_input_data, buffer_data, num_elements);
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ADASUM_ALLREDUCE);
  auto recv_buffer = GetRecvBuffer(buffer_len);
  DispatchFusedAllreduce(entries, buffer_data, recv_buffer, tensor_counts,
                         1, // start_level
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL),
                         0, // tag
                         reduction_comms_, first_entry.tensor->dtype(),
                         global_state_);
  timeline.ActivityEndAll(entries);

  if (response.postscale_factor() != 1.0) {
    // Execute postscaling op
    ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data, num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}
} // namespace common
} // namespace horovod
