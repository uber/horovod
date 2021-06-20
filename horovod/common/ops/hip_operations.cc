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
#include "../message.h"

#include <thread>

namespace horovod {
namespace common {

class GPUContext::impl {
public:
  hipError_t GetGpuEvent(hipEvent_t* event) {
    int device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) {
      return status;
    }

    auto& mutex = hip_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = hip_events[device];
      if (!queue.empty()) {
        *event = queue.front();
        queue.pop();
        return hipSuccess;
      }
    }

    return hipEventCreateWithFlags(event, hipEventDisableTiming);
  }

  hipError_t ReleaseGpuEvent(hipEvent_t event) {
    int device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) {
      return status;
    }

    auto& mutex = hip_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = hip_events[device];
      queue.push(event);
    }

    return hipSuccess;
  }

  void ErrorCheck(std::string op_name, hipError_t hip_result) {
    if (hip_result != hipSuccess) {
      throw std::logic_error(std::string(op_name) + " failed: " + hipGetErrorString(hip_result));
    }
  }

  void RecordEvent(std::queue<std::pair<std::string, hipEvent_t>>& event_queue, std::string name, hipStream_t& stream) {
    hipEvent_t event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event));
    ErrorCheck("hipEventRecord", hipEventRecord(event, stream));
    event_queue.emplace(name, event);
  }

  void WaitForEvents(std::queue<std::pair<std::string, hipEvent_t>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline& timeline,
      const std::function<void()>& error_check_callback,
      bool elastic) {
    while (!event_queue.empty()) {
      std::string name;
      hipEvent_t event;
      std::tie(name, event) = event_queue.front();
      event_queue.pop();
      if (name != "") {
        timeline.ActivityStartAll(entries, name);
      }

      // Check for async (networking) errors while waiting for the event to complete
      hipError_t hip_result;
      while (true) {
        hip_result = hipEventQuery(event);
        if (hip_result == hipSuccess) {
          break;
        }

        if (hip_result != hipErrorNotReady) {
          throw std::logic_error(std::string("hipEventQuery failed: ") + hipGetErrorString(hip_result));
        }

        if (error_check_callback) {
          error_check_callback();
        }
        std::this_thread::yield();
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void StreamCreate(hipStream_t *stream) {
    int greatest_priority;
    ErrorCheck("hipDeviceGetStreamPriorityRange",
        hipDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    ErrorCheck("hipStreamCreateWithPriority",
        hipStreamCreateWithPriority(stream, hipStreamNonBlocking, greatest_priority));
  }

  void StreamSynchronize(hipStream_t stream) {
    ErrorCheck("hipStreamSynchronize", hipStreamSynchronize(stream));
  }

  int GetDevice() {
    int device;
    ErrorCheck("hipGetDevice", hipGetDevice(&device));
    return device;
  }

  void SetDevice(int device) {
    ErrorCheck("hipSetDevice", hipSetDevice(device));
  }

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count, hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync", hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream));
  }

  void MemcpyAsyncH2D(void* dst, const void* src, size_t count, hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync", hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
  }

  void MemcpyAsyncD2H(void* dst, const void* src, size_t count, hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync", hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
  }

  void ScaleBufferImpl(const void* fused_input_data, void* buffer_data, int64_t num_elements,
                   double scale_factor, DataType dtype, hipStream_t stream) {
    throw std::logic_error("ScaleBuffer not implemented for AMD GPUs.");
  }

private:
  // We reuse HIP events as it appears that their creation carries non-zero cost.
  std::unordered_map<int, std::queue<hipEvent_t>> hip_events;
  std::mutex hip_events_mutex;
};

#include "gpu_context_impl.cc"

} // namespace common
} // namespace horovod
