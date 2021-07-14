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

#include <atomic>

#include "../common/operations.h"
#include "cuda_util.h"
#include "mpi_ops.h"

namespace horovod {
namespace mxnet {

namespace {

std::atomic_int op_count;

std::string GetOpName(const char* prefix, const char* name) {
  if (name != nullptr) {
    return std::string(prefix) + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return std::string(prefix) + ".noname." + std::to_string(op_count);
}
} // namespace

static const auto MX_EXEC_CTX = Context();
static const auto MX_FUNC_PROP = FnProperty::kCPUPrioritized;
static const char* ALLREDUCE_OP_TYPE_NAME = "horovod_allreduce";
static const char* ALLGATHER_OP_TYPE_NAME = "horovod_allgather";
static const char* BROADCAST_OP_TYPE_NAME = "horovod_broadcast";
static const char* ALLTOALL_OP_TYPE_NAME = "horovod_alltoall";

inline void InvokeCompleteCallback(CallbackOnComplete on_complete, const Status& status) {
  if (status.ok()) {
    on_complete();
  } else {
    auto error = dmlc::Error(status.reason());
    on_complete(&error);
  }
}

inline const char* GetOpTypeName(OperationType op_type) {
  switch (op_type) {
    case OperationType::ALLREDUCE:
      return ALLREDUCE_OP_TYPE_NAME;
    case OperationType::ALLGATHER:
      return ALLGATHER_OP_TYPE_NAME;
    case OperationType::BROADCAST:
      return BROADCAST_OP_TYPE_NAME;
    case OperationType::ALLTOALL:
      return ALLTOALL_OP_TYPE_NAME;
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }
}

bool IsTensorOnCPU(NDArray* tensor) {
  return tensor->ctx().dev_mask() == cpu::kDevMask;
}

void DoHorovodOperation(void*, void* on_complete_ptr, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto on_complete = *static_cast<CallbackOnComplete*>(on_complete_ptr);
  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto first_tensor = ops_param->input_tensors[0].get();
  auto device = TensorUtil::GetDevice(first_tensor);
  auto average = ops_param->average;
  auto prescale_factor = ops_param->prescale_factor;
  auto postscale_factor = ops_param->postscale_factor;
  auto num_tensors = ops_param->input_tensors.size();
  auto process_set_id = ops_param->process_set_id;

  std::vector<std::shared_ptr<Tensor>> hvd_tensors;
  std::vector<std::shared_ptr<OpContext>> hvd_contexts;
  std::vector<StatusCallback> callbacks;
  std::vector<ReadyEventList> ready_event_lists;
  hvd_tensors.reserve(num_tensors);
  hvd_contexts.reserve(num_tensors);
  ready_event_lists.resize(num_tensors);
  callbacks.reserve(num_tensors);

  auto callback_mutex = std::make_shared<std::mutex>();
  for (int i = 0; i < num_tensors; ++i) {
    auto input_tensor = ops_param->input_tensors[i].get();
    auto output = ops_param->outputs[i];

    hvd_tensors.emplace_back(std::make_shared<MXTensor>(input_tensor));
    if (TensorUtil::GetDevice(input_tensor) != device) {
      throw std::logic_error("Tensors in list must be on same device.");
    }
    auto ctx = std::make_shared<MXOpContext>(device, output);
    if (ops_param->received_splits_tensor) {
      ctx->AddOutput(ops_param->received_splits_tensor.get());
    }
    hvd_contexts.push_back(ctx);
    callbacks.emplace_back([on_complete, ops_param, callback_mutex, i](const Status& status) {
#if HAVE_CUDA
      auto hvd_event = status.event;
      if (hvd_event.event) {
        HVD_GPU_CHECK(gpuEventSynchronize(*(hvd_event.event)));
      }
#endif

      // Must only invoke MXNet callback on last tensor to prevent premature deletion of
      // shared ops_param structure. Guard logic is here instead of within DeleteMpiOpsParam
      // function as on_complete can only be invoked once due to MXNet internally
      // pairing up the engine op completion callback with DeleteMpiOpsParam.
      std::lock_guard<std::mutex> guard(*callback_mutex);
      ops_param->del_count++;
      if (ops_param->del_count == ops_param->input_tensors.size()) {
        InvokeCompleteCallback(on_complete, status);
      }
    });

  }

  Status enqueue_result;
  std::vector<std::shared_ptr<Tensor>> hvd_outputs;
  hvd_outputs.reserve(num_tensors);
  switch (ops_param->op_type) {
    case OperationType::ALLREDUCE:
      for (int i = 0; i < num_tensors; ++i) {
        hvd_outputs.emplace_back(std::make_shared<MXTensor>(ops_param->output_tensors[i].get()));
      }

      enqueue_result = EnqueueTensorAllreduces(
          hvd_contexts, hvd_tensors, hvd_outputs, ready_event_lists,
          ops_param->op_names, device, callbacks,
          (average) ? ReduceOp::AVERAGE : ReduceOp::SUM, prescale_factor,
          postscale_factor, process_set_id);
      break;
    case OperationType::ALLGATHER:
      enqueue_result = EnqueueTensorAllgather(
          hvd_contexts[0], hvd_tensors[0], ready_event_lists[0],
          ops_param->op_names[0], device, callbacks[0], process_set_id);
      break;
    case OperationType::BROADCAST:
      if (horovod_rank() != ops_param->root_rank) {
        hvd_outputs.emplace_back(std::make_shared<MXTensor>(ops_param->output_tensors[0].get()));
      } else {
        hvd_outputs.emplace_back(nullptr);
      }

      enqueue_result = EnqueueTensorBroadcast(
          hvd_contexts[0], hvd_tensors[0], hvd_outputs[0], ops_param->root_rank,
          ready_event_lists[0], ops_param->op_names[0], device, callbacks[0],
          process_set_id);
      break;
    case OperationType::ALLTOALL:
    {
      auto hvd_splits = std::make_shared<MXTensor>(ops_param->splits_tensor.get());
      enqueue_result = EnqueueTensorAlltoall(
          hvd_contexts[0], hvd_tensors[0], hvd_splits, ready_event_lists[0],
          ops_param->op_names[0], device, callbacks[0], process_set_id);
      break;
    }
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }

  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperation(OperationType op_type, NDArray* const * inputs,
                                 NDArray* const * outputs, const char* name,
                                 int priority, int num_tensors,
                                 int process_set_id = 0,
                                 int root_rank = -1,
                                 bool average = true,
                                 NDArray* splits = nullptr,
                                 NDArray* output_received_splits = nullptr,
                                 double prescale_factor = 1.0,
                                 double postscale_factor = 1.0) {
  auto op_type_name = GetOpTypeName(op_type);

  auto first_input_var = inputs[0]->var();
  auto first_output_var = outputs[0]->var();

  bool inplace = first_input_var == first_output_var;

  // We need to create a shared_ptr to NDArray object with
  // shallow copy to prevent from NDArray object being freed
  // before MXNet engine process it
  std::vector<std::shared_ptr<NDArray>> input_copies;
  std::vector<std::shared_ptr<NDArray>> output_copies;
  std::vector<NDArray*> outputs_vec;
  std::vector<std::shared_ptr<NDArray>> cpu_input_tensors; // empty
  std::vector<std::shared_ptr<NDArray>> cpu_output_tensors; // empty
  std::vector<std::string> op_names;
  std::vector<void*> input_vars;
  std::vector<void*> output_vars;

  input_copies.reserve(num_tensors);
  output_copies.reserve(num_tensors);
  outputs_vec.reserve(num_tensors);
  op_names.reserve(num_tensors);
  output_vars.reserve(num_tensors);
  if (!inplace) input_vars.reserve(num_tensors);

  auto base_name = GetOpName(op_type_name, name);
  for (int i = 0; i < num_tensors; ++i) {
    input_copies.emplace_back(std::make_shared<NDArray>(*inputs[i]));
    output_copies.emplace_back(std::make_shared<NDArray>(*outputs[i]));
    outputs_vec.emplace_back(outputs[i]);
    output_vars.emplace_back(outputs[i]->var());
    if (num_tensors > 1) {
      op_names.emplace_back(base_name + "_" + std::to_string(i+1) + "of" + std::to_string(num_tensors));
    } else {
      op_names.emplace_back(base_name);
    }

    if (!inplace) {
      input_vars.emplace_back(inputs[i]->var());
    }
  }

  std::shared_ptr<NDArray> splits_tensor;
  std::shared_ptr<NDArray> received_splits_tensor;
  if (splits) {
#if HAVE_CUDA
    // We expect splits to be a tensor on CPU. Create CPU copy if required.
    if (!IsTensorOnCPU(splits)) {
      splits_tensor = std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
      splits->dtype());
      TensorUtil::AsyncCopyCudaToCPU(splits, splits_tensor.get());
    } else {
      splits_tensor = std::make_shared<NDArray>(*splits);
    }
#else
    splits_tensor = std::make_shared<NDArray>(*splits);
#endif
    if (!output_received_splits) {
      throw std::logic_error("output_received_splits must be passed if splits are passed");
    }
    if (!IsTensorOnCPU(output_received_splits)) {
      throw std::logic_error("output_received_splits should be on CPU");
    }
    received_splits_tensor = std::make_shared<NDArray>(*output_received_splits);
  }

  auto ops_param = CreateMpiOpsParam(
      std::move(input_copies), std::move(output_copies), std::move(outputs_vec),
      cpu_input_tensors, cpu_output_tensors, op_type, std::move(op_names),
      root_rank, average, splits_tensor, received_splits_tensor,
      prescale_factor, postscale_factor, process_set_id);

  // Not in-place
  if (!inplace) {
    if (splits) {
      // Add splits tensor to input list to enforce dependency on possible async D2H copy
      input_vars.push_back(splits_tensor->var());
      output_vars.push_back(received_splits_tensor->var());
    }
    MXEnginePushAsync(DoHorovodOperation, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, input_vars.data(), input_vars.size(), output_vars.data(), output_vars.size(),
                      &MX_FUNC_PROP, priority, op_type_name);
  // In-place
  } else {
    if (splits) {
      input_vars.push_back(splits_tensor->var());
      output_vars.push_back(received_splits_tensor->var());
    }
    MXEnginePushAsync(DoHorovodOperation, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, input_vars.data(), input_vars.size(), output_vars.data(), output_vars.size(),
                      &MX_FUNC_PROP, priority, op_type_name);
  }
}
#if HAVE_CUDA
void DoHorovodOperationCudaOnCPU(void*, void* on_complete_ptr, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto on_complete = *static_cast<CallbackOnComplete*>(on_complete_ptr);
  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto device = CPU_DEVICE_ID;
  auto average = ops_param->average;
  auto prescale_factor = ops_param->prescale_factor;
  auto postscale_factor = ops_param->postscale_factor;
  auto num_tensors = ops_param->cpu_input_tensors.size();
  auto process_set_id = ops_param->process_set_id;

  std::vector<std::shared_ptr<Tensor>> hvd_cpu_buffers;
  std::vector<std::shared_ptr<OpContext>> hvd_contexts;
  std::vector<StatusCallback> callbacks;
  std::vector<ReadyEventList> ready_event_lists;
  hvd_cpu_buffers.reserve(num_tensors);
  hvd_contexts.reserve(num_tensors);
  ready_event_lists.resize(num_tensors);
  callbacks.reserve(num_tensors);

  auto callback_mutex = std::make_shared<std::mutex>();
  for (int i = 0; i < num_tensors; ++i) {
    auto input = ops_param->cpu_input_tensors[i].get();
    auto output = ops_param->cpu_output_tensors[i].get();

    hvd_cpu_buffers.emplace_back(std::make_shared<MXTensor>(input));
    if (TensorUtil::GetDevice(input) != device) {
      throw std::logic_error("Tensors in list must be on same device.");
    }
    auto ctx = std::make_shared<MXOpContext>(device, output);
    if (ops_param->received_splits_tensor) {
      ctx->AddOutput(ops_param->received_splits_tensor.get());
    }
    hvd_contexts.push_back(ctx);
    callbacks.emplace_back([on_complete, ops_param, callback_mutex](const Status& status) {
      // Must only invoke callback on last tensor to prevent premature deletion of
      // shared ops_param structure. Guard logic is here instead of within DeleteMpiOpsParam
      // function as on_complete can only be invoked once due to MXNet internally
      // pairing up the engine op completion callback with DeleteMpiOpsParam.
      std::lock_guard<std::mutex> guard(*callback_mutex);
      ops_param->del_count++;
      if (ops_param->del_count == ops_param->cpu_input_tensors.size()) {
        InvokeCompleteCallback(on_complete, status);
      }
    });

  }

  Status enqueue_result;
  switch (ops_param->op_type) {
  case OperationType::ALLREDUCE:
    enqueue_result = EnqueueTensorAllreduces(
        hvd_contexts, hvd_cpu_buffers, hvd_cpu_buffers, ready_event_lists,
        ops_param->op_names, device, callbacks,
        (average) ? ReduceOp::AVERAGE : ReduceOp::SUM, prescale_factor,
        postscale_factor, process_set_id);
    break;
  case OperationType::ALLGATHER:
    enqueue_result = EnqueueTensorAllgather(
        hvd_contexts[0], hvd_cpu_buffers[0], ready_event_lists[0],
        ops_param->op_names[0], device, callbacks[0], process_set_id);
    break;
  case OperationType::BROADCAST:
    enqueue_result = EnqueueTensorBroadcast(
        hvd_contexts[0], hvd_cpu_buffers[0], hvd_cpu_buffers[0],
        ops_param->root_rank, ready_event_lists[0], ops_param->op_names[0],
        device, callbacks[0], process_set_id);
    break;
  case OperationType::ALLTOALL: {
    auto hvd_splits =
        std::make_shared<MXTensor>(ops_param->splits_tensor.get());
    enqueue_result = EnqueueTensorAlltoall(
        hvd_contexts[0], hvd_cpu_buffers[0], hvd_splits, ready_event_lists[0],
        ops_param->op_names[0], device, callbacks[0], process_set_id);
    break;
  }
  default:
    throw std::logic_error("Unsupported Horovod operation type.");
  }

  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperationCudaOnCPU(OperationType op_type, NDArray* const * inputs,
                                          NDArray* const * outputs, const char* name,
                                          int priority, int num_tensors,
                                          int process_set_id = 0,
                                          int root_rank = -1,
                                          bool average = true,
                                          NDArray* splits = nullptr,
                                          NDArray* output_received_splits = nullptr,
                                          double prescale_factor = 1.0,
                                          double postscale_factor = 1.0) {
  auto op_type_name = GetOpTypeName(op_type);

  std::vector<std::shared_ptr<NDArray>> input_copies; //empty
  std::vector<std::shared_ptr<NDArray>> output_copies; //empty
  std::vector<NDArray*> outputs_vec; //empty
  std::vector<std::shared_ptr<NDArray>> cpu_input_tensors;
  std::vector<std::shared_ptr<NDArray>> cpu_output_tensors;
  std::vector<std::string> op_names;
  cpu_input_tensors.reserve(num_tensors);
  cpu_output_tensors.reserve(num_tensors);
  op_names.reserve(num_tensors);

  auto base_name = GetOpName(op_type_name, name);
  for (int i = 0; i < num_tensors; ++i) {
    cpu_input_tensors.emplace_back(std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
    inputs[i]->dtype()));
    cpu_output_tensors.emplace_back(std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
    inputs[i]->dtype()));

    // Make async copy of input tensor to CPU tensor.
    TensorUtil::AsyncCopyCudaToCPU(inputs[i], cpu_input_tensors[i].get());

    if (num_tensors > 1) {
      op_names.emplace_back(base_name + "_" + std::to_string(i+1) + "of" + std::to_string(num_tensors));
    } else {
      op_names.emplace_back(base_name);
    }
  }

  std::shared_ptr<NDArray> splits_tensor;
  std::shared_ptr<NDArray> received_splits_tensor;
  if (splits) {
    // We expect splits to be a tensor on CPU. Create CPU copy if required.
    if (!IsTensorOnCPU(splits)) {
      splits_tensor = std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
      splits->dtype());
      TensorUtil::AsyncCopyCudaToCPU(splits, splits_tensor.get());
    } else {
      splits_tensor = std::make_shared<NDArray>(*splits);
    }
    if (!output_received_splits) {
      throw std::logic_error("output_received_splits must be passed if splits are passed");
    }
    if (!IsTensorOnCPU(output_received_splits)) {
      throw std::logic_error("output_received_splits should be on CPU");
    }
    received_splits_tensor = std::make_shared<NDArray>(*output_received_splits);
  }

  auto ops_param = CreateMpiOpsParam(
      std::move(input_copies), std::move(output_copies), std::move(outputs_vec),
      cpu_input_tensors, cpu_output_tensors, op_type, std::move(op_names),
      root_rank, average, splits_tensor, received_splits_tensor,
      prescale_factor, postscale_factor, process_set_id);

  std::vector<void*> cpu_input_vars;
  std::vector<void*> cpu_output_vars;
  cpu_input_vars.reserve(num_tensors);
  cpu_output_vars.reserve(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    cpu_input_vars.emplace_back(cpu_input_tensors[i]->var());
    cpu_output_vars.emplace_back(cpu_output_tensors[i]->var());
  }

  if (op_type == OperationType::ALLGATHER ||
      op_type == OperationType::ALLTOALL) {
    if (splits) {
      // Add splits tensor to input list to enforce dependency on possible async D2H copy
      cpu_input_vars.push_back(splits_tensor->var());
      cpu_output_vars.push_back(received_splits_tensor->var());
    }
    // Use out-of-place path for operations that have unknown output size (allgather)
    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, cpu_input_vars.data(), cpu_input_vars.size(), cpu_output_vars.data(), cpu_output_vars.size(),
                      &MX_FUNC_PROP, priority, op_type_name);

    for (int i = 0; i < num_tensors; ++i) {
       // Since cpu_output_tensor is resized in out-of-place path, need
       // to wait for operation to complete before copying to GPU output.
       cpu_output_tensors[i]->WaitToRead();

       // Make async copy of CPU output tensor to output tensor.
       TensorUtil::AsyncCopyCPUToCuda(cpu_output_tensors[i].get(), outputs[i]);
    }
  } else {
    // Use in-place otherwise
    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, nullptr, 0, cpu_input_vars.data(), cpu_input_vars.size(),
                      &MX_FUNC_PROP, priority, op_type_name);

    for (int i = 0; i < num_tensors; ++i) {
      // Make async copy of CPU input tensor to output tensor.
      TensorUtil::AsyncCopyCPUToCuda(cpu_input_tensors[i].get(), outputs[i]);
    }
  }

}
#endif

extern "C" int horovod_mxnet_allreduce_async(NDArray* const * inputs,
                                             NDArray* const * outputs,
                                             const char* name, bool average,
                                             int priority,
                                             double prescale_factor,
                                             double postscale_factor,
                                             int num_tensors,
                                             int process_set_id) {
  MX_API_BEGIN();

#if HAVE_ROCM
  // Averaging left at framework level for ROCm until ScaleBuffer implementation
  // added.
  bool average_in_framework = average;
  average = false;
#endif

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  if (IsTensorOnCPU(inputs[0]) && IsTensorOnCPU(outputs[0])) {
    PushHorovodOperation(OperationType::ALLREDUCE, inputs, outputs, name,
                         priority, num_tensors, process_set_id, -1, average,
                         nullptr, nullptr, prescale_factor, postscale_factor);
  } else {
    PushHorovodOperationCudaOnCPU(OperationType::ALLREDUCE, inputs, outputs,
                                  name, priority, num_tensors, process_set_id,
                                  -1, average, nullptr, nullptr,
                                  prescale_factor, postscale_factor);
  }
#else
  PushHorovodOperation(OperationType::ALLREDUCE, inputs, outputs, name,
                       priority, num_tensors, process_set_id, -1, average,
                       nullptr, nullptr, prescale_factor, postscale_factor);
#endif

#if HAVE_ROCM
  if (average_in_framework) {
    for (int i = 0; i < num_tensors; ++i) {
      *outputs[i] /= horovod_size();
    }
  }
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_allgather_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int priority,
                                             int process_set_id) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_ALLGATHER
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(OperationType::ALLGATHER, &input, &output,
                         name, priority, 1, process_set_id);
  } else {
    PushHorovodOperationCudaOnCPU(OperationType::ALLGATHER, &input, &output,
                                  name, priority, 1, process_set_id);
  }
#else
  PushHorovodOperation(OperationType::ALLGATHER, &input, &output,
                       name, priority, 1, process_set_id);
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_broadcast_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int root_rank,
                                             int priority,
                                             int process_set_id) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_BROADCAST
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(OperationType::BROADCAST, &input, &output,
                         name, priority, 1, process_set_id, root_rank);

  } else {
    PushHorovodOperationCudaOnCPU(OperationType::BROADCAST, &input, &output,
                                  name, priority, 1, process_set_id, root_rank);
  }
#else
  PushHorovodOperation(OperationType::BROADCAST, &input, &output, name,
                       priority, 1, process_set_id, root_rank);
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_alltoall_async(NDArray* input,
                                            NDArray* output,
                                            const char* name,
                                            NDArray* splits,
                                            NDArray* output_received_splits,
                                            int priority,
                                            int process_set_id) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_ALLTOALL
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(OperationType::ALLTOALL, &input, &output, name,
                         priority, 1, process_set_id, -1, false, splits,
                         output_received_splits);

  } else {
    PushHorovodOperationCudaOnCPU(OperationType::ALLTOALL, &input, &output,
                                  name, priority, 1, process_set_id, -1, false,
                                  splits, output_received_splits);
  }
#else
  PushHorovodOperation(OperationType::ALLTOALL, &input, &output, name, priority,
                       1, process_set_id, -1, false, splits,
                       output_received_splits);
#endif

  MX_API_END();
}

} // namespace mxnet
} // namespace horovod
