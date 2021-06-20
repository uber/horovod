// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#if HAVE_GPU
#if TORCH_VERSION >= 1005000000
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#else
#include <THC/THC.h>
#endif
#include <cassert>
#include <mutex>
#include <queue>
#include <unordered_map>
#else
#include <stdexcept>
#endif

#include "ready_event.h"
#include "cuda_util.h"

#if TORCH_VERSION < 1005000000
#if HAVE_GPU
extern THCState* state;
#endif
#endif

namespace horovod {
namespace torch {

#if HAVE_GPU
struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

TorchReadyEvent::TorchReadyEvent(int device) : device_(device) {
  assert(device_ != CPU_DEVICE_ID);

  with_device device_context(device_);
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    if (!queue.empty()) {
      cuda_event_ = queue.front();
      queue.pop();
    } else {
      #if TORCH_VERSION >= 1005000000
      C10_CUDA_CHECK(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
      #else
      THCudaCheck(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
      #endif
    }
  }
  #if TORCH_VERSION >= 1005000000
  auto stream = c10::cuda::getCurrentCUDAStream(device_);
  C10_CUDA_CHECK(cudaEventRecord(cuda_event_, stream));
  #else
  auto stream = THCState_getCurrentStreamOnDevice(state, device_);
  THCudaCheck(cudaEventRecord(cuda_event_, stream));
  #endif
}

TorchReadyEvent::~TorchReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(cuda_event_);
  }
}

bool TorchReadyEvent::Ready() const {
  #if TORCH_VERSION >= 1005000000
  C10_CUDA_CHECK(cudaEventSynchronize(cuda_event_));
  #else
  THCudaCheck(cudaEventSynchronize(cuda_event_));
  #endif
  return true;
}

gpuEvent_t TorchReadyEvent::event() const {
  return cuda_event_;
}
#endif

// On GPU this event will signal that GPU computations are done and data is
// ready.
std::shared_ptr<ReadyEvent> RecordReadyEvent(int device) {
  if (device == CPU_DEVICE_ID) {
    return std::shared_ptr<ReadyEvent>();
  } else {
#if HAVE_GPU
    return std::make_shared<TorchReadyEvent>(device);
#else
    throw std::logic_error("Internal error. Requested ReadyEvent "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

} // namespace torch
} // namespace horovod
