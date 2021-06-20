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

#include "gpu_operations.h"
#if HAVE_CUDA
#include "cuda/cuda_kernels.h"
#endif

#include <thread>

namespace horovod {
namespace common {

GPUOpContext::GPUOpContext(GPUContext* context, HorovodGlobalState* global_state)
    : gpu_context_(context), global_state_(global_state) {}

void GPUOpContext::InitGPU(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  gpu_context_->SetDevice(first_entry.device);

  // Ensure stream is in the map before executing reduction.
  gpuStream_t& stream = gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device];
  if (stream == nullptr) {
    gpu_context_->StreamCreate(&stream);
  }
}

void GPUOpContext::InitGPUQueue(const std::vector<TensorTableEntry>& entries, const Response& response) {
  event_queue = std::queue<std::pair<std::string, Event>>();
  stream = &gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device];

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(event_queue, QUEUE, *stream);
  }
}

Status GPUOpContext::FinalizeGPUQueue(std::vector<TensorTableEntry>& entries, bool free_host_buffer /*= true*/,
                                      const std::function<void()>& error_check_callback) {
  // Use completion marker via event because it's faster than
  // blocking gpuStreamSynchronize() in this thread.
  if (!global_state_->enable_async_completion) {
    gpu_context_->RecordEvent(event_queue, "", *stream);
  }

  auto& first_entry = entries[0];
  void* cpu_buffer = host_buffer;
  auto& evt_queue = event_queue;
  auto& timeline = global_state_->timeline;
  auto& gpu_context = gpu_context_;

  // Claim a std::shared_ptr to the fusion buffer to prevent its memory from being reclaimed
  // during finalization.
  auto fusion_buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);

  bool elastic = global_state_->elastic_enabled;
  bool enable_async_completion = global_state_->enable_async_completion;
  auto current_stream = *stream;
  gpu_context_->finalizer_thread_pool.execute([entries, first_entry, cpu_buffer, fusion_buffer, free_host_buffer,
                                               evt_queue, &timeline, &gpu_context, error_check_callback,
                                               elastic, enable_async_completion, current_stream]() mutable {
    gpu_context->SetDevice(first_entry.device);

    Event event;
    if (!enable_async_completion || timeline.Initialized()) {
      // If timeline is enabled, wait for events on CPU for accurate timings.
      gpu_context->WaitForEvents(evt_queue, entries, timeline, error_check_callback, elastic);
    } else {
      gpu_context->ClearEvents(evt_queue, entries, timeline, error_check_callback, elastic);
      event = gpu_context->RecordEvent(current_stream);
    }

    if (free_host_buffer && cpu_buffer != nullptr) {
      free(cpu_buffer);
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      auto status = Status::OK();
      status.event = event;
      e.FinishWithCallback(status);
    }
    if (enable_async_completion) {
      gpu_context->ReleaseEvent(event);
    }
  });

  // Update current stream
  global_state_->current_nccl_stream = (global_state_->current_nccl_stream + 1) %
                                  global_state_->num_nccl_streams;

  return Status::InProgress();
}

GPUAllreduce::GPUAllreduce(GPUContext* context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), gpu_context_(context), gpu_op_context_(context, global_state) {}

bool GPUAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

#if HAVE_CUDA
void GPUAllreduce::MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
                                        void*& buffer_data, size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    auto& first_entry = entries[0];
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = (void*) e.tensor->data();
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset += BATCHED_D2D_PADDING * ((e.tensor->size() + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int) entries.size()) {
        // Perform batched d2d memcpy
        BatchedD2DMemcpyCudaImpl(d2d_params, count, gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
        // TODO: https://github.com/horovod/horovod/issues/2230
        //gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl", cudaGetLastError());
        count = 0;
      }
    }
    buffer_len = (size_t)offset;

  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
      offset += e.tensor->size();
    }

    buffer_len = (size_t) offset;
  }

  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}
#endif


void GPUAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(buffer_data_at_offset, e.tensor->data(), (size_t) e.tensor->size(),
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

#if HAVE_CUDA
void GPUAllreduce::MemcpyOutFusionBuffer(const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    auto& first_entry = entries[0];
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = (void*)(e.output->data());
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset += BATCHED_D2D_PADDING * ((e.tensor->size() + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int) entries.size()) {
        // Perform batched d2d memcpy
        BatchedD2DMemcpyCudaImpl(d2d_params, count, gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
        // TODO: https://github.com/horovod/horovod/issues/2230
        //gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl", cudaGetLastError());
        count = 0;
      }
    }

  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e);
      offset += e.tensor->size();
    }
  }
}
#endif

void GPUAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                               const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D((void*) e.output->data(), buffer_data_at_offset, (size_t) e.tensor->size(),
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void GPUAllreduce::ScaleBuffer(double scale_factor, const std::vector<TensorTableEntry>& entries,
                               const void* fused_input_data, void* buffer_data, int64_t num_elements) {
  gpu_context_->ScaleBufferImpl(fused_input_data, buffer_data, num_elements, scale_factor, entries[0].tensor->dtype(),
                                gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);

}

GPUAllgather::GPUAllgather(GPUContext* context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), gpu_context_(context), gpu_op_context_(context, global_state) {}

bool GPUAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void GPUAllgather::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(buffer_data_at_offset, e.tensor->data(), (size_t) e.tensor->size(),
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void GPUAllgather::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const void* buffer_data_at_offset, TensorTableEntry& e,
                                              int64_t entry_offset, size_t entry_size) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D((int8_t*)e.output->data() + entry_offset, buffer_data_at_offset, entry_size,
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

GPUBroadcast::GPUBroadcast(GPUContext* context,
                           HorovodGlobalState* global_state)
    : BroadcastOp(global_state), gpu_context_(context), gpu_op_context_(context, global_state) {}

bool GPUBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

GPUAlltoall::GPUAlltoall(GPUContext* context,
		         HorovodGlobalState* global_state)
    : AlltoallOp(global_state), gpu_context_(context), gpu_op_context_(context, global_state) {}

bool GPUAlltoall::Enabled(const ParameterManager& param_manager,
                          const std::vector<TensorTableEntry>& entries,
                          const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

} // namespace common
} // namespace horovod
