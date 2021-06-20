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

#ifndef HOROVOD_TORCH_READY_EVENT_H
#define HOROVOD_TORCH_READY_EVENT_H

#if HAVE_GPU
#include "cuda_runtime.h"
#endif

#include <memory>

#include "../common/common.h"

namespace horovod {
namespace torch {

using namespace horovod::common;

#if HAVE_GPU
class TorchReadyEvent : public ReadyEvent {
public:
  TorchReadyEvent(int device);
  ~TorchReadyEvent();
  virtual bool Ready() const override;
  gpuEvent_t event() const override;

private:
  int device_ = CPU_DEVICE_ID;
  gpuEvent_t cuda_event_ = nullptr;
};
#endif

std::shared_ptr<ReadyEvent> RecordReadyEvent(int device);

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_READY_EVENT_H
