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

#include "adasum_gpu_operations.h"

namespace horovod {
namespace common {

AdasumGpuAllreduceOp::AdasumGpuAllreduceOp(MPIContext* mpi_context,
                                           NCCLContext* nccl_context,
                                           GPUContext* gpu_context,
                                           HorovodGlobalState* global_state)
    : AdasumMPI(mpi_context, global_state),
      NCCLAllreduce(nccl_context, gpu_context, global_state, Communicator::LOCAL) {
  // Pre-allocate host buffer size equal to the fusion buffer length
  current_host_buffer_length =
      global_state->parameter_manager.TensorFusionThresholdBytes();
  gpu_op_context_.host_buffer = (uint8_t*)malloc(current_host_buffer_length);
}

AdasumGpuAllreduceOp::~AdasumGpuAllreduceOp() {
  if (gpu_op_context_.host_buffer != nullptr) {
    free(gpu_op_context_.host_buffer);
  }
}

void AdasumGpuAllreduceOp::WaitForData(std::vector<TensorTableEntry>& entries) {
  HorovodOp::WaitForData(entries);
}

Status AdasumGpuAllreduceOp::Execute(std::vector<TensorTableEntry>& entries,
                                     const Response& response) {
  if (entries.empty()) {
    return Status::OK();
  }

  WaitForData(entries);

  // Lazily initialize reduction communicators for VHDD algorithm when Adasum reduction is actually called.
  if (!reduction_comms_initialized) {
    InitializeVHDDReductionComms();
  }
  return NcclHierarchical(entries, response);
}

uint8_t* AdasumGpuAllreduceOp::GetHostBuffer(uint64_t buffer_length) {
  return CheckBufferAndReallocate((uint8_t**)&gpu_op_context_.host_buffer,
                                  buffer_length, current_host_buffer_length);
}

Status
AdasumGpuAllreduceOp::NcclHierarchical(std::vector<TensorTableEntry>& entries,
                                       const Response& response) {
  auto& first_entry = entries[0];

  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> nccl_device_map;
  nccl_device_map.reserve(
      global_state_->controller->GetLocalCommRanks().size());
  for (size_t rank : global_state_->controller->GetLocalCommRanks()) {
    nccl_device_map.push_back(response.devices()[rank]);
  }
  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, nccl_device_map);
  gpu_op_context_.InitGPUQueue(entries, response);
  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;
  uint8_t* host_buffer;
  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                 MEMCPY_IN_FUSION_BUFFER,
                                 *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    buffer_len = (size_t)first_entry.output->size();
  }

  int64_t num_elements = buffer_len / DataType_Size(first_entry.tensor->dtype());

  if (response.prescale_factor() != 1.0) {
    // Execute prescaling op
    ScaleBuffer(response.prescale_factor(), entries, fused_input_data, buffer_data, num_elements);
    fused_input_data = buffer_data; // for unfused, scale is done out of place
  }

  // Do allreduce.
  int element_size = mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());
  int local_size = global_state_->controller->GetLocalSize();
  int local_rank = global_state_->controller->GetLocalRank();

  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (global_state_->controller->IsHomogeneous() && entries.size() > 1) {
    // Making sure the number of elements is divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for improved performance
    int div = local_size * FUSION_BUFFER_ATOMIC_UNIT;
    num_elements = ((num_elements + div - 1) / div) * div;
    buffer_len = num_elements * element_size;
  }

  // Split the elements into two groups: num_elements_per_rank*local_size,
  // and num_elements_remaining. Cross-node reduction for the first group
  // is done by all local_rank's in parallel, while for the second group
  // it it is only done by the root_rank. If the cluster is not
  // homogeneous first group is zero, and root_rank is 0.

  // Homogeneous case:
  // For the part of data divisible by local_size, perform NCCL
  // ReduceScatter - Parallelized MPI Allreduce - NCCL Allgather. For the
  // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
  // MPI Allreduce (across rank (local_size-1)'s), and NCCL Bcast

  int64_t num_elements_per_rank = global_state_->controller->IsHomogeneous()
                                      ? num_elements / local_size
                                      : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void* buffer_data_at_rank_offset =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_rank;

  int64_t num_elements_remaining = global_state_->controller->IsHomogeneous()
                                       ? num_elements % local_size
                                       : num_elements;

  size_t buffer_len_remaining = element_size * num_elements_remaining;

  void* buffer_data_remainder =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_size;

  void* fused_input_data_remainder =
      (uint8_t*)fused_input_data + buffer_len_per_rank * local_size;

  int root_rank =
      global_state_->controller->IsHomogeneous() ? local_size - 1 : 0;
  bool is_root_rank = local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;
  int64_t total_buffer_len = is_root_rank
                                 ? buffer_len_per_rank + buffer_len_remaining
                                 : buffer_len_per_rank;

  auto& timeline = global_state_->timeline;
  if (num_elements_per_rank > 0) {
    auto nccl_result = ncclReduceScatter(
        fused_input_data, buffer_data_at_rank_offset,
        (size_t)num_elements_per_rank, GetNCCLDataType(first_entry.tensor),
        ncclSum, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);

    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result, *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                 NCCL_REDUCESCATTER, *gpu_op_context_.stream);
    }
  }

  if (num_elements_remaining > 0) {
    // Reduce the remaining data at local_size-1 to append to
    // existing buffer
    auto nccl_result = ncclReduce(
        fused_input_data_remainder, buffer_data_remainder,
        (size_t)num_elements_remaining, GetNCCLDataType(first_entry.tensor),
        ncclSum, root_rank, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);

    nccl_context_->ErrorCheck("ncclReduce", nccl_result, *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
                                 *gpu_op_context_.stream);
    }
  }

  if (global_state_->controller->IsHomogeneous() || is_root_rank) {
    // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
    // a buffer is not safe since the tensor can be arbitrarily large.
    host_buffer = GetHostBuffer((uint64_t)total_buffer_len);
    // Synchronize.
    gpu_context_->WaitForEvents(gpu_op_context_.event_queue, entries,
                                 timeline, nullptr, global_state_->elastic_enabled);

    // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
    // api-sync-behavior.html#api-sync-behavior__memcpy-async,
    // cudaMemcpyAsync is synchronous with respect to the host, so we
    // memcpy (effectively) synchronously to generate an accurate timeline
    timeline.ActivityStartAll(entries, MEMCPY_IN_HOST_BUFFER);
    gpu_context_->MemcpyAsyncD2H(host_buffer, buffer_data_at_rank_offset,
                                 total_buffer_len, *gpu_op_context_.stream);

    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MPI_ADASUM_ALLREDUCE);

    // Since Adasum is not a per-element operation, an allreduce for fused
    // tensors needs to know boundaries of tensors. Calculate here the count
    // of elements for each tensor owned by this rank.
    std::vector<int> tensor_counts(entries.size());
    if (global_state_->controller->IsHomogeneous()) {
      // For homogeneous clusters each rank owns a slice of the fused tensor.

      int64_t num_elements_sofar = 0;
      size_t i = 0;
      for (auto& e : entries) {
        int64_t e_num_elements = e.tensor->shape().num_elements();
        int64_t left_boundary =
            std::max(num_elements_sofar, local_rank * num_elements_per_rank);
        int64_t right_boundary =
            std::min(num_elements_sofar + e_num_elements,
                     (local_rank + 1) * num_elements_per_rank);
        tensor_counts[i] = std::max(right_boundary - left_boundary, (int64_t)0);
        if (is_root_rank) {
          if (num_elements_sofar + e_num_elements >=
              local_size * num_elements_per_rank) {
            left_boundary = std::max(num_elements_sofar,
                                     local_size * num_elements_per_rank);
            right_boundary = num_elements_sofar + e_num_elements;
            tensor_counts[i] +=
                std::max(right_boundary - left_boundary, (int64_t)0);
          }
        }

        num_elements_sofar += e_num_elements;
        i++;
      }
    } else {
      // For non-homogeneous clusters the root rank owns everything.

      if (is_root_rank) {
        size_t i = 0;
        for (auto& e : entries) {
          int e_num_elements = e.tensor->shape().num_elements();
          tensor_counts[i] = e_num_elements;
          i++;
        }
      }
    }

    auto recv_buffer = GetRecvBuffer(total_buffer_len);
    DispatchFusedAllreduce(
        entries, (void*)host_buffer, (void*)recv_buffer, tensor_counts,
        local_size, // start_level
        global_state_->controller->IsHomogeneous()
            ? MPI_COMM_WORLD
            : mpi_context_->GetMPICommunicator(Communicator::CROSS),
        0, reduction_comms_, first_entry.tensor->dtype(), global_state_);
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MEMCPY_OUT_HOST_BUFFER);
    gpu_context_->MemcpyAsyncH2D(buffer_data_at_rank_offset,
                                 host_buffer, total_buffer_len,
                                 *gpu_op_context_.stream);
    timeline.ActivityEndAll(entries);
  }

  if (num_elements_per_rank > 0) {
    nccl_context_->ErrorCheck(
        "ncclAllGather", ncclAllGather(buffer_data_at_rank_offset, buffer_data,
                                       (size_t)num_elements_per_rank,
                                       GetNCCLDataType(first_entry.tensor),
                                       *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
        *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
                                *gpu_op_context_.stream);
    }
  }
  if (num_elements_remaining > 0) {
    nccl_context_->ErrorCheck(
        "ncclBcast",
        ncclBcast(buffer_data_remainder, (size_t)num_elements_remaining,
                  GetNCCLDataType(first_entry.tensor), root_rank, *nccl_op_context_.nccl_comm_,
                  *gpu_op_context_.stream),
        *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
                                *gpu_op_context_.stream);
    }
  }

  if (response.postscale_factor() != 1.0) {
    // Execute postscaling op
    ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data, num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(entries, false);
}

bool AdasumGpuAllreduceOp::Enabled(
    const ParameterManager& param_manager,
    const std::vector<TensorTableEntry>& entries,
    const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}
} // namespace common
} // namespace horovod
