// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
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

#if HAVE_GPU
#if TORCH_VERSION >= 1005000000
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#else
#include <THC/THC.h>
#endif
#endif

#include <chrono>
#include <memory>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/operations.h"
#include "adapter_v2.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "ready_event.h"

#if TORCH_VERSION < 1005000000
#if HAVE_GPU
extern THCState* state;
#endif
#endif

namespace horovod {
namespace torch {

static HandleManager handle_manager;

namespace {

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

int GetDeviceID(const ::torch::Tensor& tensor) {
  if (tensor.device().is_cuda()) {
    return tensor.device().index();
  }
  return CPU_DEVICE_ID;
}

} // namespace

void DivideInPlace(::torch::Tensor& tensor, int divisor) {
#if TORCH_VERSION >= 1005000000
  if (isIntegralType(tensor.scalar_type())) {
    tensor.floor_divide_(divisor);
    return;
  }
#endif
  tensor.div_(divisor);
}

#if HAVE_GPU
gpuStream_t GetGPUStream(int device) {
  #if TORCH_VERSION >= 1005000000
  return c10::cuda::getCurrentCUDAStream(device);
  #else
  return THCState_getCurrentStreamOnDevice(state, device);
  #endif
}
#endif

int DoAllreduce(::torch::Tensor tensor, ::torch::Tensor output, int divisor,
                const std::string& name, int reduce_op_int,
                double prescale_factor, double postscale_factor,
                int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  auto hvd_output = std::make_shared<TorchTensor>(output);

  ReduceOp reduce_op = static_cast<ReduceOp>(reduce_op_int);
  
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, ready_event_list,
      GetOpName("allreduce", name, handle), device,
      [handle, divisor, output, device](const Status& status) mutable {
#if HAVE_GPU
        auto hvd_event = status.event;
        if (hvd_event.event) {
          auto stream = GetGPUStream(device);
          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
        }
#endif
        // Will execute in the `device` context.
        if (divisor > 1) {
          DivideInPlace(output, divisor);
        }
        handle_manager.MarkDone(handle, status);
      }, reduce_op, prescale_factor, postscale_factor, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllreduceCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output, int divisor,
                         const std::string& name, int reduce_op_int,
                         double prescale_factor, double postscale_factor,
                         int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_buffer =
      tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
  auto hvd_cpu_buffer = std::make_shared<TorchTensor>(cpu_buffer);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif

  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_buffer);

  ReduceOp reduce_op = static_cast<ReduceOp>(reduce_op_int);
  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event_list,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, divisor, cpu_buffer, output,
       device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        output.copy_(cpu_buffer);
        if (divisor > 1) {
          DivideInPlace(output, divisor);
        }
        handle_manager.MarkDone(handle, status);
      }, reduce_op, prescale_factor, postscale_factor, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoGroupedAllreduce(const std::vector<::torch::Tensor>& tensors,
                       const std::vector<::torch::Tensor>& outputs, int divisor,
                       const std::string& name, int reduce_op_int,
                       double prescale_factor, double postscale_factor,
                       int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensors[0]);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif

  std::vector<std::shared_ptr<Tensor>> hvd_tensors;
  std::vector<std::shared_ptr<OpContext>> hvd_contexts;
  std::vector<std::shared_ptr<Tensor>> hvd_outputs;
  std::vector<common::ReadyEventList> ready_event_lists;
  std::vector<StatusCallback> callbacks;
  std::vector<std::string> names;

  auto num_tensors = tensors.size();
  hvd_tensors.reserve(num_tensors);
  hvd_contexts.reserve(num_tensors);
  hvd_outputs.reserve(num_tensors);
  ready_event_lists.reserve(num_tensors);
  names.reserve(num_tensors);
  callbacks.reserve(num_tensors);

  auto base_name = GetOpName("grouped_allreduce", name, handle);

  auto callback_mutex = std::make_shared<std::mutex>();
  auto callback_count = std::make_shared<int>(0);
  for (int i = 0; i < num_tensors; ++i) {
    if (GetDeviceID(tensors[i]) != device) {
      throw std::logic_error("Tensors in list must be on same device.");
    }
    hvd_tensors.emplace_back(std::make_shared<TorchTensor>(tensors[i]));
    hvd_contexts.emplace_back(std::make_shared<TorchOpContext>(device, outputs[i]));
    hvd_outputs.emplace_back(std::make_shared<TorchTensor>(outputs[i]));
    ready_event_lists.emplace_back(ready_event_list); // Same for all tensors in group
    names.emplace_back(base_name + "_" + std::to_string(i+1) + "of" + std::to_string(num_tensors));
    auto output = outputs[i];
    callbacks.emplace_back(
      [handle, divisor, output, callback_mutex, callback_count, num_tensors,
       device](const Status& status) mutable {
#if HAVE_GPU
        auto hvd_event = status.event;
        if (hvd_event.event) {
          auto stream = GetGPUStream(device);
          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
        }
#endif
        // Will execute in the `device` context.
        if (divisor > 1) {
          DivideInPlace(output, divisor);
        }
        // Must only call MarkDone on last tensor.
        std::lock_guard<std::mutex> guard(*callback_mutex);
        (*callback_count)++;
        if (*callback_count == num_tensors) {
          handle_manager.MarkDone(handle, status);
        }
      }
    );
  }

  ReduceOp reduce_op = static_cast<ReduceOp>(reduce_op_int);

  auto enqueue_result = EnqueueTensorAllreduces(
      hvd_contexts, hvd_tensors, hvd_outputs, ready_event_lists,
      names, device, callbacks, reduce_op, prescale_factor, postscale_factor,
      process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoGroupedAllreduceCudaOnCPU(const std::vector<::torch::Tensor>& tensors,
                                const std::vector<::torch::Tensor>& outputs,
                                int divisor, const std::string& name,
                                int reduce_op_int, double prescale_factor,
                                double postscale_factor, int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensors[0]);

  std::vector<std::shared_ptr<Tensor>> cpu_buffers;
  std::vector<std::shared_ptr<OpContext>> hvd_contexts;
  std::vector<common::ReadyEventList> ready_event_lists;
  std::vector<StatusCallback> callbacks;
  std::vector<std::string> names;

  auto num_tensors = tensors.size();
  cpu_buffers.reserve(num_tensors);
  hvd_contexts.reserve(num_tensors);
  ready_event_lists.reserve(num_tensors);
  names.reserve(num_tensors);
  callbacks.reserve(num_tensors);

  auto base_name = GetOpName("grouped_allreduce", name, handle);

  auto callback_mutex = std::make_shared<std::mutex>();
  auto callback_count = std::make_shared<int>(0);
  for (int i = 0; i < num_tensors; ++i) {
    if (GetDeviceID(tensors[i]) != device) {
      throw std::logic_error("Tensors in list must be on same device.");
    }
    auto cpu_buffer =
        tensors[i].to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
    cpu_buffers.emplace_back(std::make_shared<TorchTensor>(cpu_buffer));
    hvd_contexts.emplace_back(std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_buffer));
    common::ReadyEventList ready_event_list;
#if HAVE_GPU
    ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif
    ready_event_lists.emplace_back(ready_event_list);
    names.emplace_back(base_name + "_" + std::to_string(i+1) + "of" + std::to_string(num_tensors));
    auto output = outputs[i];
    callbacks.emplace_back(
      [handle, divisor, cpu_buffer, output, device, callback_mutex, callback_count,
       num_tensors](const Status& status) mutable {
        // Will execute in the `device` context.
        if (divisor > 1) {
          DivideInPlace(output, divisor);
        }
        with_device device_guard(device);
        output.copy_(cpu_buffer);

        // Must only call MarkDone on last tensor.
        std::lock_guard<std::mutex> guard(*callback_mutex);
        (*callback_count)++;
        if (*callback_count == num_tensors) {
          handle_manager.MarkDone(handle, status);
        }
      }
    );
  }

  ReduceOp reduce_op = static_cast<ReduceOp>(reduce_op_int);

  auto enqueue_result = EnqueueTensorAllreduces(
      hvd_contexts, cpu_buffers, cpu_buffers, ready_event_lists, names, device,
      callbacks, reduce_op, prescale_factor, postscale_factor, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllgather(::torch::Tensor tensor, ::torch::Tensor output,
                const std::string& name, int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result =
      EnqueueTensorAllgather(hvd_context, hvd_tensor, ready_event_list,
                             GetOpName("allgather", name, handle), device,
                             [handle, device](const Status& status) {
#if HAVE_GPU
                               auto hvd_event = status.event;
                               if (hvd_event.event) {
                                 auto stream = GetGPUStream(device);
                                 HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
                               }
#endif
                               handle_manager.MarkDone(handle, status);
                             }, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllgatherCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output,
                         const std::string& name, int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_tensor =
      tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
  auto hvd_cpu_tensor = std::make_shared<TorchTensor>(cpu_tensor);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif

  auto cpu_output = ::torch::empty_like(cpu_tensor);
  auto hvd_cpu_output = std::make_shared<TorchTensor>(cpu_output);
  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_output);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event_list,
      GetOpName("allgather", name, handle), CPU_DEVICE_ID,
      [handle, cpu_output, output, device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        // output needs to be resized before copying in the CPU tensor.
        output.resize_(cpu_output.sizes());
        output.copy_(cpu_output);
        handle_manager.MarkDone(handle, status);
      }, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoBroadcast(::torch::Tensor tensor, ::torch::Tensor output, int root_rank,
                const std::string& name, int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor.data_ptr() != output.data_ptr()) {
      with_device device_guard(device);
      output.copy_(tensor);
    }
  } else {
    hvd_output = std::make_shared<TorchTensor>(output);
  }

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result =
      EnqueueTensorBroadcast(hvd_context, hvd_tensor, hvd_output, root_rank,
                             ready_event_list, GetOpName("broadcast", name, handle),
                             device, [handle, device](const Status& status) {
#if HAVE_GPU
                               auto hvd_event = status.event;
                               if (hvd_event.event) {
                                 auto stream = GetGPUStream(device);
                                 HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
                               }
#endif
                               handle_manager.MarkDone(handle, status);
                             }, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoBroadcastCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output, int root_rank,
                         const std::string& name, int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_buffer =
      tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
  auto hvd_cpu_buffer = std::make_shared<TorchTensor>(cpu_buffer);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif

  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_buffer);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event_list,
      GetOpName("broadcast", name, handle), CPU_DEVICE_ID,
      [handle, cpu_buffer, output, device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        output.copy_(cpu_buffer);
        handle_manager.MarkDone(handle, status);
      }, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAlltoall(::torch::Tensor tensor, ::torch::Tensor splits,
               ::torch::Tensor output, ::torch::Tensor output_received_splits,
               const std::string& name, int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);

  // Make sync copy of splits tensor to CPU if needed
  auto cpu_splits = (GetDeviceID(splits) != CPU_DEVICE_ID) ?
      splits.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false) :
      splits;
  auto hvd_cpu_splits = std::make_shared<TorchTensor>(cpu_splits);

  // Deal with possibility of output_received_splits being on GPU
  auto received_splits_device = GetDeviceID(output_received_splits);
  auto cpu_received_splits = (received_splits_device != CPU_DEVICE_ID)
                                 ? ::torch::empty_like(cpu_splits)
                                 : output_received_splits;
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  hvd_context->AddOutput(CPU_DEVICE_ID, cpu_received_splits);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAlltoall(
      hvd_context, hvd_tensor, hvd_cpu_splits, ready_event_list,
      GetOpName("alltoall", name, handle), device,
      [handle, cpu_received_splits, output_received_splits,
       received_splits_device, device](const Status& status) mutable {
#if HAVE_GPU
        auto hvd_event = status.event;
        if (hvd_event.event) {
          auto stream = GetGPUStream(device);
          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
        }
#endif
        if (received_splits_device != CPU_DEVICE_ID) {
          with_device device_guard(received_splits_device);
          output_received_splits.resize_(cpu_received_splits.sizes());
          output_received_splits.copy_(cpu_received_splits);
        }
        handle_manager.MarkDone(handle, status); 
      }, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAlltoallCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor splits,
                        ::torch::Tensor output,
                        ::torch::Tensor output_received_splits,
                        const std::string& name, int process_set_id) {
  ThrowIfError(common::CheckInitialized());

  // Make sync copy of splits tensor to CPU if needed
  auto cpu_splits =
      (GetDeviceID(splits) != CPU_DEVICE_ID)
          ? splits.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false)
          : splits;
  auto hvd_cpu_splits = std::make_shared<TorchTensor>(cpu_splits);

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_tensor =
      tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
  auto hvd_cpu_tensor = std::make_shared<TorchTensor>(cpu_tensor);
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif

  auto cpu_output = ::torch::empty_like(cpu_tensor);
  auto hvd_cpu_output = std::make_shared<TorchTensor>(cpu_output);

  // Deal with possibility of output_received_splits being on GPU
  auto received_splits_device = GetDeviceID(output_received_splits);
  auto cpu_received_splits = (received_splits_device != CPU_DEVICE_ID)
                                 ? ::torch::empty_like(cpu_splits)
                                 : output_received_splits;
  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_output);
  hvd_context->AddOutput(CPU_DEVICE_ID, cpu_received_splits);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAlltoall(
      hvd_context, hvd_cpu_tensor, hvd_cpu_splits, ready_event_list,
      GetOpName("alltoall", name, handle), CPU_DEVICE_ID,
      [handle, cpu_output, output, device, cpu_received_splits,
       output_received_splits,
       received_splits_device](const Status& status) mutable {
        { // Since the operation was on CPU, need to perform copy with the GPU
          // device guard.
          with_device device_guard(device);
          // output needs to be resized before copying in the CPU tensor.
          output.resize_(cpu_output.sizes());
          output.copy_(cpu_output);
        }
        if (received_splits_device != CPU_DEVICE_ID) {
          with_device device_guard(received_splits_device);
          output_received_splits.resize_(cpu_received_splits.sizes());
          output_received_splits.copy_(cpu_received_splits);
        }
        handle_manager.MarkDone(handle, status);
      }, process_set_id);
  ThrowIfError(enqueue_result);

  return handle;
}

int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void WaitAndClear(int handle) {
  while (true) {
    if (handle_manager.PollHandle(handle)) break;
    std::this_thread::yield();
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

int DoJoin(int device) {
  ThrowIfError(common::CheckInitialized());

#if !HOROVOD_GPU_ALLREDUCE
  device = CPU_DEVICE_ID;
#endif

  auto handle = handle_manager.AllocateHandle();
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(RecordReadyEvent(device));
#endif
  auto output = ::torch::empty(1);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);

  auto enqueue_result = EnqueueJoin(
      hvd_context, ready_event_list,
      JOIN_TENSOR_NAME, device,
      [handle, device](const Status& status) mutable {
#if HAVE_GPU
        auto hvd_event = status.event;
        if (hvd_event.event) {
          auto stream = GetGPUStream(device);
          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
        }
#endif
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  WaitAndClear(handle);
  return handle;
}

void Reset() {
  handle_manager.Reset();
}


PYBIND11_MODULE(mpi_lib_v2, m) {
  // allreduce
  m.def("horovod_torch_allreduce_async_torch_IntTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_LongTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_HalfTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_FloatTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_DoubleTensor", &DoAllreduce);
#if HOROVOD_GPU_ALLREDUCE
  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_HalfTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor", &DoAllreduce);
#else
  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_HalfTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor",
        &DoAllreduceCudaOnCPU);
#endif

  // grouped allreduce
  m.def("horovod_torch_grouped_allreduce_async_torch_IntTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_LongTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_HalfTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_FloatTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_DoubleTensor", &DoGroupedAllreduce);
#if HOROVOD_GPU_ALLREDUCE
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_IntTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_LongTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_HalfTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_FloatTensor", &DoGroupedAllreduce);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_DoubleTensor", &DoGroupedAllreduce);
#else
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_IntTensor",
        &DoGroupedAllreduceCudaOnCPU);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_LongTensor",
        &DoGroupedAllreduceCudaOnCPU);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_HalfTensor",
        &DoGroupedAllreduceCudaOnCPU);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_FloatTensor",
        &DoGroupedAllreduceCudaOnCPU);
  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_DoubleTensor",
        &DoGroupedAllreduceCudaOnCPU);
#endif

  // allgather
  m.def("horovod_torch_allgather_async_torch_ByteTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_CharTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_ShortTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_IntTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_LongTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_HalfTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_FloatTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_DoubleTensor", &DoAllgather);
#if HOROVOD_GPU_ALLGATHER
  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_HalfTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor", &DoAllgather);
#else
  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_HalfTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor",
        &DoAllgatherCudaOnCPU);
#endif

  // broadcast
  m.def("horovod_torch_broadcast_async_torch_ByteTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_CharTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_ShortTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_IntTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_LongTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_HalfTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_FloatTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_DoubleTensor", &DoBroadcast);
#if HOROVOD_GPU_BROADCAST
  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_HalfTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor", &DoBroadcast);
#else
  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_HalfTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor",
        &DoBroadcastCudaOnCPU);
#endif

  // alltoall
  m.def("horovod_torch_alltoall_async_torch_ByteTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_CharTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_ShortTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_IntTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_LongTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_HalfTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_FloatTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_DoubleTensor", &DoAlltoall);
#if HOROVOD_GPU_ALLTOALL
  m.def("horovod_torch_alltoall_async_torch_cuda_ByteTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_cuda_CharTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_cuda_ShortTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_cuda_IntTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_cuda_LongTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_cuda_HalfTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_cuda_FloatTensor", &DoAlltoall);
  m.def("horovod_torch_alltoall_async_torch_cuda_DoubleTensor", &DoAlltoall);
#else
  m.def("horovod_torch_alltoall_async_torch_cuda_ByteTensor",
        &DoAlltoallCudaOnCPU);
  m.def("horovod_torch_alltoall_async_torch_cuda_CharTensor",
        &DoAlltoallCudaOnCPU);
  m.def("horovod_torch_alltoall_async_torch_cuda_ShortTensor",
        &DoAlltoallCudaOnCPU);
  m.def("horovod_torch_alltoall_async_torch_cuda_IntTensor",
        &DoAlltoallCudaOnCPU);
  m.def("horovod_torch_alltoall_async_torch_cuda_LongTensor",
        &DoAlltoallCudaOnCPU);
  m.def("horovod_torch_alltoall_async_torch_cuda_HalfTensor",
        &DoAlltoallCudaOnCPU);
  m.def("horovod_torch_alltoall_async_torch_cuda_FloatTensor",
        &DoAlltoallCudaOnCPU);
  m.def("horovod_torch_alltoall_async_torch_cuda_DoubleTensor",
        &DoAlltoallCudaOnCPU);
#endif

  // join
  m.def("horovod_torch_join", &DoJoin);

  // basics
  m.def("horovod_torch_poll", &PollHandle);
  m.def("horovod_torch_wait_and_clear", &WaitAndClear);
  m.def("horovod_torch_reset", &Reset);
}

} // namespace torch
} // namespace horovod
