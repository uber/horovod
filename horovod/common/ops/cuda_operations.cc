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
#include "cuda/cuda_kernels.h"
#include "../message.h"
#include "../hashes.h"

#include <thread>

namespace horovod {
namespace common {
class GPUContext::impl {
public:
  cudaError_t GetGpuEvent(Event* event, cudaStream_t stream) {
    int device;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    auto& mutex = cuda_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto key = std::make_pair(device, stream);
      auto& queue = cuda_events[key];
      if (!prepopulated[key]) {
        // On first call for device and stream pair, prepopulate event queue.
        // This is to minimize event reuse of callback events passed to framework.
        for (int i = 0; i < N_CUDA_EVENTS_PREPOPULATE; ++i) {
          cudaEvent_t ev;
          status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
          queue.emplace(std::make_shared<cudaEvent_t>(ev), stream);
        }
        prepopulated[key] = true;
      }
      if (!queue.empty()) {
        *event = queue.front();
        queue.pop();
        return cudaSuccess;
      }
    }

    cudaEvent_t ev;
    status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    event->event = std::make_shared<cudaEvent_t>(ev);
    event->stream = stream;

    return status;
  }

  cudaError_t ReleaseGpuEvent(Event event) {
    int device;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    auto& mutex = cuda_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = cuda_events[std::make_pair(device, event.stream)];
      queue.push(event);
    }

    return cudaSuccess;
  }

  void ErrorCheck(std::string op_name, cudaError_t cuda_result) {
    if (cuda_result != cudaSuccess) {
      throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
    }
  }

  void RecordEvent(std::queue<std::pair<std::string, Event>>& event_queue, std::string name, cudaStream_t& stream) {
    Event event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
    ErrorCheck("cudaEventRecord", cudaEventRecord(*(event.event), event.stream));
    event_queue.emplace(name, event);
  }

  Event RecordEvent(cudaStream_t& stream) {
    Event event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
    ErrorCheck("cudaEventRecord", cudaEventRecord(*(event.event), event.stream));
    return event;
  }

  void WaitForEvents(std::queue<std::pair<std::string, Event>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline& timeline,
      const std::function<void()>& error_check_callback,
      bool elastic) {
    while (!event_queue.empty()) {
      std::string name;
      Event event;
      std::tie(name, event) = event_queue.front();
      event_queue.pop();
      if (name != "") {
        timeline.ActivityStartAll(entries, name);
      }

      // Check for async (networking) errors while waiting for the event to complete
      if (elastic) {
        cudaError_t cuda_result;
        while (true) {
          cuda_result = cudaEventQuery(*(event.event));
          if (cuda_result == cudaSuccess) {
            break;
          }

          if (cuda_result != cudaErrorNotReady) {
            throw std::logic_error(std::string("cudaEventQuery failed: ") + cudaGetErrorString(cuda_result));
          }

          if (error_check_callback) {
            error_check_callback();
          }
          std::this_thread::yield();
        }
      } else {
        cudaError_t cuda_result = cudaEventSynchronize(*(event.event));
        if (cuda_result != cudaSuccess) {
          throw std::logic_error(std::string("cudaEventSynchronize failed: ") + cudaGetErrorString(cuda_result));
        }
        if (error_check_callback) {
          error_check_callback();
        }
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void ClearEvents(std::queue<std::pair<std::string, Event>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline& timeline,
      const std::function<void()>& error_check_callback,
      bool elastic) {
    while (!event_queue.empty()) {
      std::string name;
      Event event;
      std::tie(name, event) = event_queue.front();
      event_queue.pop();
      if (name != "") {
        timeline.ActivityStartAll(entries, name);
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void StreamCreate(cudaStream_t *stream) {
    int greatest_priority;
    ErrorCheck("cudaDeviceGetStreamPriorityRange",
        cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    ErrorCheck("cudaStreamCreateWithPriority",
        cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority));
  }

  void StreamSynchronize(cudaStream_t stream) {
    ErrorCheck("cudaStreamSynchronize", cudaStreamSynchronize(stream));
  }

  int GetDevice() {
    int device;
    ErrorCheck("cudaGetDevice", cudaGetDevice(&device));
    return device;
  }

  void SetDevice(int device) {
    ErrorCheck("cudaSetDevice", cudaSetDevice(device));
  }

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
  }

  void MemcpyAsyncH2D(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
  }

  void MemcpyAsyncD2H(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
  }

  void ScaleBufferImpl(const void* fused_input_data, void* buffer_data, int64_t num_elements,
                       double scale_factor, DataType dtype, cudaStream_t stream) {
    ScaleBufferCudaImpl(fused_input_data, buffer_data, num_elements, scale_factor, dtype, stream);

    // TODO: https://github.com/horovod/horovod/issues/2230
    //ErrorCheck("ScaleBufferCudaImpl", cudaGetLastError());
  }

private:
  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::unordered_map<std::pair<int, cudaStream_t>, std::queue<Event>> cuda_events;
  std::unordered_map<std::pair<int, cudaStream_t>, bool> prepopulated;
  std::mutex cuda_events_mutex;

  static constexpr int N_CUDA_EVENTS_PREPOPULATE = 128;
};

#include "gpu_context_impl.cc"

} // namespace common
} // namespace horovod
