// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
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

#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../common/common.h"

#define EIGEN_USE_THREADS

#if HAVE_GPU

#if HAVE_CUDA
#include <cuda_runtime.h>
using GpuStreamHandle = cudaStream_t;
#define gpuMemsetAsync cudaMemsetAsync
#elif HAVE_ROCM
#include <hip/hip_runtime.h>
using GpuStreamHandle = hipStream_t;
#define gpuMemsetAsync hipMemsetAsync
#endif // HAVE_CUDA, HAVE_ROCM

// Forward declaration of AsGpuStreamValue
namespace stream_executor {
namespace gpu {
GpuStreamHandle AsGpuStreamValue(Stream* stream);
} // namespace stream_executor
} // namespace gpu
#include "tensorflow/stream_executor/stream.h"
#endif // HAVE_GPU

#define OMPI_SKIP_MPICXX
#include "../common/operations.h"

using namespace tensorflow;
using namespace horovod;

namespace horovod {
namespace tensorflow {

namespace {

::tensorflow::DataType GetTFDataType(common::DataType dtype) {
  switch (dtype) {
  case common::HOROVOD_UINT8:
    return DT_UINT8;
  case common::HOROVOD_INT8:
    return DT_INT8;
  case common::HOROVOD_UINT16:
    return DT_UINT16;
  case common::HOROVOD_INT16:
    return DT_INT16;
  case common::HOROVOD_INT32:
    return DT_INT32;
  case common::HOROVOD_INT64:
    return DT_INT64;
  case common::HOROVOD_FLOAT16:
    return DT_HALF;
  case common::HOROVOD_FLOAT32:
    return DT_FLOAT;
  case common::HOROVOD_FLOAT64:
    return DT_DOUBLE;
  case common::HOROVOD_BOOL:
    return DT_BOOL;
  default:
    throw std::logic_error("Invalid data type.");
  }
}

Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
  case common::OK:
    return Status::OK();
  case common::UNKNOWN_ERROR:
    return errors::Unknown(status.reason());
  case common::PRECONDITION_ERROR:
    return errors::FailedPrecondition(status.reason());
  case common::ABORTED:
    return errors::Aborted(status.reason());
  case common::INVALID_ARGUMENT:
    return errors::InvalidArgument(status.reason());
  default:
    return errors::Unknown("Unknown error.");
  }
}

common::Status ConvertStatus(const Status& status) {
  switch (status.code()) {
  case error::Code::OK:
    return common::Status::OK();
  case error::Code::UNKNOWN:
    return common::Status::UnknownError(status.error_message());
  case error::Code::FAILED_PRECONDITION:
    return common::Status::PreconditionError(status.error_message());
  case error::Code::ABORTED:
    return common::Status::Aborted(status.error_message());
  case error::Code::INVALID_ARGUMENT:
    return common::Status::InvalidArgument(status.error_message());
  default:
    return common::Status::UnknownError("Unknown error.");
  }
}

int GetDeviceID(OpKernelContext* context);

#if HAVE_GPU
struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<gpuEvent_t>> gpu_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

class TFReadyEvent : public common::ReadyEvent {
public:
  TFReadyEvent(OpKernelContext* context);
  ~TFReadyEvent();
  bool Ready() const override;
  gpuEvent_t event() const override;

private:
  gpuEvent_t event_;
  int device_ = CPU_DEVICE_ID;
};
#endif

class TFPersistentBuffer : public common::PersistentBuffer {
public:
  TFPersistentBuffer(OpKernelContext* context, int64_t size);
  virtual const void*
  AccessData(std::shared_ptr<common::OpContext> context) const override;

private:
  std::shared_ptr<Tensor> tensor_;
};

class TFTensor : public common::Tensor {
public:
  TFTensor(::tensorflow::Tensor& tensor);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;
  const ::tensorflow::Tensor* tensor() const;

protected:
  ::tensorflow::Tensor tensor_;
};

class TFOpContext : public common::OpContext {
public:
  TFOpContext(OpKernelContext* context);
  virtual common::Status AllocatePersistent(
      int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) override;
  virtual common::Status
  AllocateOutput(common::TensorShape shape,
                 std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Status
  AllocateOutput(int output_index, common::TensorShape shape,
                 std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Status
  AllocateZeros(int64_t num_elements, common::DataType dtype,
                std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Framework framework() const override;
  OpKernelContext* GetKernelContext() const;

private:
  OpKernelContext* context_ = nullptr;
};

#if HAVE_GPU
TFReadyEvent::TFReadyEvent(OpKernelContext* context) {
  device_ = GetDeviceID(context);
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.gpu_events[device_];
    if (!queue.empty()) {
      event_ = queue.front();
      queue.pop();
    } else {
      HVD_GPU_CHECK(gpuEventCreateWithFlags(&event_, gpuEventDisableTiming));
    }
  }
  auto device_context = context->op_device_context();
  auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
  HVD_GPU_CHECK(gpuEventRecord(event_, stream));
}

bool TFReadyEvent::Ready() const {
  HVD_GPU_CHECK(gpuEventSynchronize(event_));
  return true;
}

gpuEvent_t TFReadyEvent::event() const {
  return event_;
}

TFReadyEvent::~TFReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.gpu_events[device_];
    queue.push(event_);
  }
}

#endif

TFPersistentBuffer::TFPersistentBuffer(OpKernelContext* context, int64_t size) {
  tensor_ = std::make_shared<Tensor>();
  TensorShape buffer_shape;
  buffer_shape.AddDim(size);
  Status status = context->allocate_temp(DT_INT8, buffer_shape, tensor_.get());
  if (!status.ok()) {
    throw status;
  }
#if HAVE_GPU
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
#endif
}

const void* TFPersistentBuffer::AccessData(
    std::shared_ptr<common::OpContext> context) const {
  return (const void *)tensor_->tensor_data().data();
}

TFTensor::TFTensor(::tensorflow::Tensor& tensor) : tensor_(tensor) {}

const common::DataType TFTensor::dtype() const {
  switch (tensor_.dtype()) {
  case DT_UINT8:
    return common::HOROVOD_UINT8;
  case DT_INT8:
    return common::HOROVOD_INT8;
  case DT_UINT16:
    return common::HOROVOD_UINT16;
  case DT_INT16:
    return common::HOROVOD_INT16;
  case DT_INT32:
    return common::HOROVOD_INT32;
  case DT_INT64:
    return common::HOROVOD_INT64;
  case DT_HALF:
    return common::HOROVOD_FLOAT16;
  case DT_FLOAT:
    return common::HOROVOD_FLOAT32;
  case DT_DOUBLE:
    return common::HOROVOD_FLOAT64;
  case DT_BOOL:
    return common::HOROVOD_BOOL;
  default:
    throw std::logic_error("Invalid tensor type.");
  }
}

const common::TensorShape TFTensor::shape() const {
  common::TensorShape shape;
  for (auto dim : tensor_.shape()) {
    shape.AddDim(dim.size);
  }
  return shape;
}

const void* TFTensor::data() const { return (const void*)tensor_.tensor_data().data(); }

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

const ::tensorflow::Tensor*  TFTensor::tensor() const { return &tensor_; }

TFOpContext::TFOpContext(OpKernelContext* context) : context_(context) {}

common::Status TFOpContext::AllocatePersistent(
    int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) {
  try {
    *tensor = std::make_shared<TFPersistentBuffer>(context_, size);
    return common::Status::OK();
  } catch (Status& status) {
    return ConvertStatus(status);
  }
}

common::Status
TFOpContext::AllocateOutput(common::TensorShape shape,
                            std::shared_ptr<common::Tensor>* tensor) {
  return TFOpContext::AllocateOutput(0, shape, tensor);
}

common::Status
TFOpContext::AllocateOutput(int output_index, common::TensorShape shape,
                            std::shared_ptr<common::Tensor>* tensor) {
  TensorShape tf_shape;
  for (int idx = 0; idx < shape.dims(); ++idx) {
    tf_shape.AddDim(shape.dim_size(idx));
  }
  Tensor* tf_tensor;
  Status status = context_->allocate_output(output_index, tf_shape, &tf_tensor);
  if (status.ok()) {
    *tensor = std::make_shared<TFTensor>(*tf_tensor);
  }
#if HAVE_GPU
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = context_->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
#endif
  return ConvertStatus(status);
}

int GetDeviceID(OpKernelContext* context);

common::Status
TFOpContext::AllocateZeros(int64_t num_elements, common::DataType dtype,
                           std::shared_ptr<common::Tensor>* tensor) {
  std::shared_ptr<Tensor> zero_tensor = std::make_shared<Tensor>();
  auto tf_data_type = GetTFDataType(dtype);
  ::tensorflow::AllocatorAttributes tf_attribute;
  int device_ = GetDeviceID(context_);
  auto hvd_context = std::make_shared<TFOpContext>(context_);
  if (device_ != CPU_DEVICE_ID) {
    tf_attribute.set_on_host(false);
  } else {
    tf_attribute.set_on_host(true);
  }

  Status status = context_->allocate_temp(tf_data_type, ::tensorflow::TensorShape({num_elements}), zero_tensor.get(), tf_attribute);

  if (device_ != CPU_DEVICE_ID) {
#if HAVE_GPU
    auto device_context = context_->op_device_context();
    auto stream = (device_context != nullptr) ? stream_executor::gpu::AsGpuStreamValue(device_context->stream()) : 0;
    void *ptr = (void*)zero_tensor->tensor_data().data();
    auto size = zero_tensor->tensor_data().size();
    gpuMemsetAsync(ptr, 0, size, stream);
#endif
  } else {
    memset((void*)zero_tensor->tensor_data().data(), 0, zero_tensor->tensor_data().size());
  }
  if (status.ok()) {
    *tensor = std::make_shared<TFTensor>(*zero_tensor);
  }

#if HAVE_GPU
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = context_->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
#endif
  return ConvertStatus(status);
}

common::Framework TFOpContext::framework() const {
  return common::Framework::TENSORFLOW;
}

OpKernelContext* TFOpContext::GetKernelContext() const { return context_; }

int GetDeviceID(OpKernelContext* context) {
  int device = CPU_DEVICE_ID;
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  return device;
}

// On GPU this event will signal that data is ready, and tensors are
// allocated.
#if HAVE_GPU
common::ReadyEvent* RecordReadyEvent(OpKernelContext* context) {
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    return new TFReadyEvent(context);
  }
  return nullptr;
}
#endif

} // namespace

class HorovodAllreduceOp : public AsyncOpKernel {
public:
  explicit HorovodAllreduceOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("reduce_op", &reduce_op_));
    OP_REQUIRES_OK(context, context->GetAttr("prescale_factor", &prescale_factor_));
    OP_REQUIRES_OK(context, context->GetAttr("postscale_factor", &postscale_factor_));
    OP_REQUIRES_OK(context, context->GetAttr("ignore_name_scope", &ignore_name_scope_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    if (ignore_name_scope_) {
      auto pos = node_name.find_last_of('/');
      if (pos != std::string::npos) {
        node_name = node_name.substr(pos + 1);
      }
    }
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    horovod::common::ReduceOp reduce_op = static_cast<horovod::common::ReduceOp>(reduce_op_);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    common::ReadyEventList ready_event_list;
#if HAVE_GPU
    ready_event_list.AddReadyEvent(std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context)));
#endif
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    auto hvd_output = std::make_shared<TFTensor>(*output);
    auto enqueue_result = EnqueueTensorAllreduce(
        hvd_context, hvd_tensor, hvd_output, ready_event_list, node_name, device,
        [context, done](const common::Status& status) {
#if HAVE_GPU
          auto hvd_event = status.event;
          if (hvd_event.event) {
            auto device_context = context->op_device_context();
            if (device_context != nullptr) {
                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
            }
          }
#endif
          context->SetStatus(ConvertStatus(status));
          done();
        }, reduce_op, (double) prescale_factor_, (double) postscale_factor_);
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }

private:
  int reduce_op_;
  // Using float since TF does not support double OP attributes
  float prescale_factor_;
  float postscale_factor_;
  bool ignore_name_scope_;
};

REGISTER_KERNEL_BUILDER(Name("HorovodAllreduce").Device(DEVICE_CPU),
                        HorovodAllreduceOp);
#if HOROVOD_GPU_ALLREDUCE
REGISTER_KERNEL_BUILDER(Name("HorovodAllreduce").Device(DEVICE_GPU),
                        HorovodAllreduceOp);
#endif

REGISTER_OP("HorovodAllreduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("reduce_op: int")
    .Attr("prescale_factor: float")
    .Attr("postscale_factor: float")
    .Attr("ignore_name_scope: bool = False")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensor:     A tensor to reduce.

Output
    sum:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");

class HorovodGroupedAllreduceOp : public AsyncOpKernel {
public:
  explicit HorovodGroupedAllreduceOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("reduce_op", &reduce_op_));
    OP_REQUIRES_OK(context, context->GetAttr("prescale_factor", &prescale_factor_));
    OP_REQUIRES_OK(context, context->GetAttr("postscale_factor", &postscale_factor_));
    OP_REQUIRES_OK(context, context->GetAttr("ignore_name_scope", &ignore_name_scope_));
    OP_REQUIRES_OK(context, context->GetAttr("num_tensors", &num_tensors_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    if (ignore_name_scope_) {
      auto pos = node_name.find_last_of('/');
      if (pos != std::string::npos) {
        node_name = node_name.substr(pos + 1);
      }
    }
    auto device = GetDeviceID(context);
    horovod::common::ReduceOp reduce_op = static_cast<horovod::common::ReduceOp>(reduce_op_);
    std::vector<Tensor*> outputs(num_tensors_);

    std::vector<common::ReadyEventList> ready_event_lists;
    std::vector<std::shared_ptr<common::OpContext>> hvd_contexts;
    std::vector<std::shared_ptr<common::Tensor>> hvd_tensors;
    std::vector<std::shared_ptr<common::Tensor>> hvd_outputs;
    std::vector<common::StatusCallback> callbacks;
    std::vector<std::string> names;
    ready_event_lists.reserve(num_tensors_);
    hvd_contexts.reserve(num_tensors_);
    hvd_tensors.reserve(num_tensors_);
    hvd_outputs.reserve(num_tensors_);
    callbacks.reserve(num_tensors_);
    names.reserve(num_tensors_);
    auto callback_mutex = std::make_shared<std::mutex>();
    auto callback_count = std::make_shared<int>(0);
    int num_tensors = num_tensors_;

    for (int i = 0; i < num_tensors_; ++i) {
      auto tensor = context->input(i);
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(i, tensor.shape(), &outputs[i]),
          done);
    }

    // ReadyEvent makes sure input tensors are ready, and outputs are allocated.
    common::ReadyEventList ready_event_list;
#if HAVE_GPU
    ready_event_list.AddReadyEvent(std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context)));
#endif

    for (int i = 0; i < num_tensors_; ++i) {
      auto tensor = context->input(i);
      ready_event_lists.emplace_back(ready_event_list); // Same for all tensors in group
      hvd_contexts.emplace_back(std::make_shared<TFOpContext>(context));
      hvd_tensors.emplace_back(std::make_shared<TFTensor>(tensor));
      names.emplace_back(node_name + "_" + std::to_string(i + 1) + "of" +
                         std::to_string(num_tensors));
      hvd_outputs.emplace_back(std::make_shared<TFTensor>(*outputs[i]));
      callbacks.emplace_back(
          [context, done, callback_mutex, callback_count, num_tensors]
          (const common::Status& status) {
            // Must only invoke callback on last tensor.
            std::lock_guard<std::mutex> guard(*callback_mutex);
            (*callback_count)++;
            if (*callback_count == num_tensors) {
#if HAVE_GPU
              auto hvd_event = status.event;
              if (hvd_event.event) {
                auto device_context = context->op_device_context();
                if (device_context != nullptr) {
                    auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
                    HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
                }
              }
#endif
              context->SetStatus(ConvertStatus(status));
              done();
            }
          });
    }

    auto enqueue_result = EnqueueTensorAllreduces(
        hvd_contexts, hvd_tensors, hvd_outputs, ready_event_lists, names, device,
        callbacks, reduce_op, (double) prescale_factor_, (double) postscale_factor_);
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }

private:
  int reduce_op_;
  // Using float since TF does not support double OP attributes
  float prescale_factor_;
  float postscale_factor_;
  bool ignore_name_scope_;
  int num_tensors_;
};

REGISTER_KERNEL_BUILDER(Name("HorovodGroupedAllreduce").Device(DEVICE_CPU),
                        HorovodGroupedAllreduceOp);
#if HOROVOD_GPU_ALLREDUCE
REGISTER_KERNEL_BUILDER(Name("HorovodGroupedAllreduce").Device(DEVICE_GPU),
                        HorovodGroupedAllreduceOp);
#endif

REGISTER_OP("HorovodGroupedAllreduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("reduce_op: int")
    .Attr("prescale_factor: float")
    .Attr("postscale_factor: float")
    .Attr("ignore_name_scope: bool = False")
    .Attr("num_tensors: int")
    .Input("tensors: num_tensors*T")
    .Output("sum: num_tensors*T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_inputs(); ++i) {
          c->set_output(i, c->input(i));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a list tensors. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensors:     A list of tensors to reduce.

Output
    sum:    A list of tensors with the same shape as corresponding tensors in `tensors`, summed across all MPI processes.
)doc");

class HorovodAllgatherOp : public AsyncOpKernel {
public:
  explicit HorovodAllgatherOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ignore_name_scope", &ignore_name_scope_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    if (ignore_name_scope_) {
      auto pos = node_name.find_last_of('/');
      if (pos != std::string::npos) {
        node_name = node_name.substr(pos + 1);
      }
    }
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    // ReadyEvent makes sure input tensor is ready.  We cannot pre-allocate
    // output for allgather, since shape of result is only known after all
    // ranks make a request.
    common::ReadyEventList ready_event_list;
#if HAVE_GPU
    ready_event_list.AddReadyEvent(std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context)));
#endif
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    auto enqueue_result = EnqueueTensorAllgather(
        hvd_context, hvd_tensor, ready_event_list, node_name, device,
        [context, done](const common::Status& status) {
#if HAVE_GPU
          auto hvd_event = status.event;
          if (hvd_event.event) {
            auto device_context = context->op_device_context();
            if (device_context != nullptr) {
                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
            }
          }
#endif
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }

private:
  bool ignore_name_scope_;
};

REGISTER_KERNEL_BUILDER(Name("HorovodAllgather").Device(DEVICE_CPU),
                        HorovodAllgatherOp);
#if HOROVOD_GPU_ALLGATHER
REGISTER_KERNEL_BUILDER(Name("HorovodAllgather").Device(DEVICE_GPU),
                        HorovodAllgatherOp);
#endif

REGISTER_OP("HorovodAllgather")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("ignore_name_scope: bool = False")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allgather on a tensor. All other processes that do a gather on a
tensor with the same name must have the same rank for that tensor, and have the
same dimension on all but the first dimension.

Arguments
    tensor:     A tensor to gather.

Output
    gathered:    A tensor with the same shape as `tensor` except for the first dimension.
)doc");

class HorovodBroadcastOp : public AsyncOpKernel {
public:
  explicit HorovodBroadcastOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank_));
    OP_REQUIRES_OK(context, context->GetAttr("ignore_name_scope", &ignore_name_scope_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    if (ignore_name_scope_) {
      auto pos = node_name.find_last_of('/');
      if (pos != std::string::npos) {
        node_name = node_name.substr(pos + 1);
      }
    }
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    Tensor* output = nullptr;
    if (common::horovod_rank() == root_rank_) {
      context->set_output(0, tensor);
    } else {
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, tensor.shape(), &output), done);
    }
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    common::ReadyEventList ready_event_list;
#if HAVE_GPU
    ready_event_list.AddReadyEvent(std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context)));
#endif
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    std::shared_ptr<TFTensor> hvd_output = nullptr;
    if (output != nullptr) {
      hvd_output = std::make_shared<TFTensor>(*output);
    }
    auto enqueue_result = EnqueueTensorBroadcast(
        hvd_context, hvd_tensor, hvd_output, root_rank_, ready_event_list, node_name,
        device, [context, done](const common::Status& status) {
#if HAVE_GPU
          auto hvd_event = status.event;
          if (hvd_event.event) {
            auto device_context = context->op_device_context();
            if (device_context != nullptr) {
                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
            }
          }
#endif
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }

private:
  int root_rank_;
  bool ignore_name_scope_;
};

REGISTER_KERNEL_BUILDER(Name("HorovodBroadcast").Device(DEVICE_CPU),
                        HorovodBroadcastOp);
#if HOROVOD_GPU_BROADCAST
REGISTER_KERNEL_BUILDER(Name("HorovodBroadcast").Device(DEVICE_GPU),
                        HorovodBroadcastOp);
#endif

REGISTER_OP("HorovodBroadcast")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("root_rank: int")
    .Attr("ignore_name_scope: bool = False")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Broadcast on a tensor. All other processes that do a broadcast
on a tensor with the same name must have the same dimension for that tensor.

Arguments
    tensor:     A tensor to broadcast.
    root_rank:  Rank that will send data, other ranks will receive data.

Output
    output:    A tensor with the same shape as `tensor` and same value as
               `tensor` on root rank.
)doc");

class HorovodJoinOp : public AsyncOpKernel {
public:
  explicit HorovodJoinOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);
    auto device = GetDeviceID(context);
    common::ReadyEventList ready_event_list;
#if HAVE_GPU
    ready_event_list.AddReadyEvent(std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context)));
#endif
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto enqueue_result = EnqueueJoin(
      hvd_context, ready_event_list,
      JOIN_TENSOR_NAME, device,
        [context, done](const common::Status& status) {
#if HAVE_GPU
          auto hvd_event = status.event;
          if (hvd_event.event) {
            auto device_context = context->op_device_context();
            if (device_context != nullptr) {
                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
            }
          }
#endif
          context->SetStatus(ConvertStatus(status));
          done();
        });

   OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
};

REGISTER_KERNEL_BUILDER(Name("HorovodJoin").Device(DEVICE_CPU),
                        HorovodJoinOp);
#if HOROVOD_GPU_ALLREDUCE
REGISTER_KERNEL_BUILDER(Name("HorovodJoin").Device(DEVICE_GPU),
                        HorovodJoinOp);
#endif

REGISTER_OP("HorovodJoin")
    .Doc(R"doc(
Perform an join on a tensor,
)doc");

template <typename T, T f()> class HorovodReturnScalarOp : public OpKernel {
public:
  explicit HorovodReturnScalarOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));

    // Write integer to output tensor
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    auto flat = output->flat<T>();
    flat(0) = f();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("HorovodSize").Device(DEVICE_CPU).HostMemory("size"),
    HorovodReturnScalarOp<int, common::horovod_size>);
#if HAVE_GPU
REGISTER_KERNEL_BUILDER(
    Name("HorovodSize").Device(DEVICE_GPU).HostMemory("size"),
    HorovodReturnScalarOp<int, common::horovod_size>);
#endif

REGISTER_OP("HorovodSize")
    .Output("size: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the number of Horovod processes.

Output
    size:    An integer scalar containing the number of Horovod processes.
)doc");

REGISTER_KERNEL_BUILDER(
    Name("HorovodLocalSize").Device(DEVICE_CPU).HostMemory("local_size"),
    HorovodReturnScalarOp<int, common::horovod_local_size>);
#if HAVE_GPU
REGISTER_KERNEL_BUILDER(
    Name("HorovodLocalSize").Device(DEVICE_GPU).HostMemory("local_size"),
    HorovodReturnScalarOp<int, common::horovod_local_size>);
#endif

REGISTER_OP("HorovodLocalSize")
    .Output("local_size: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the number of Horovod processes within the node the current process is
running on.

Output
    local_size:    An integer scalar containing the number of local Horovod
                   processes.
)doc");

REGISTER_KERNEL_BUILDER(
    Name("HorovodRank").Device(DEVICE_CPU).HostMemory("rank"),
    HorovodReturnScalarOp<int, common::horovod_rank>);
#if HAVE_GPU
REGISTER_KERNEL_BUILDER(
    Name("HorovodRank").Device(DEVICE_GPU).HostMemory("rank"),
    HorovodReturnScalarOp<int, common::horovod_rank>);
#endif

REGISTER_OP("HorovodRank")
    .Output("rank: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the Horovod rank of the calling process.

Output
    rank:    An integer scalar with the Horovod rank of the calling process.
)doc");

REGISTER_KERNEL_BUILDER(
    Name("HorovodLocalRank").Device(DEVICE_CPU).HostMemory("local_rank"),
    HorovodReturnScalarOp<int, common::horovod_local_rank>);
#if HAVE_GPU
REGISTER_KERNEL_BUILDER(
    Name("HorovodLocalRank").Device(DEVICE_GPU).HostMemory("local_rank"),
    HorovodReturnScalarOp<int, common::horovod_local_rank>);
#endif

REGISTER_OP("HorovodLocalRank")
    .Output("local_rank: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Returns the local Horovod rank of the calling process, within the node that it
is running on. For example, if there are seven processes running on a node,
their local ranks will be zero through six, inclusive.

Output
    local_rank:    An integer scalar with the local Horovod rank of the calling
                   process.
)doc");

class HorovodAlltoallOp : public AsyncOpKernel {
public:
  explicit HorovodAlltoallOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ignore_name_scope", &ignore_name_scope_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    if (ignore_name_scope_) {
      auto pos = node_name.find_last_of('/');
      if (pos != std::string::npos) {
        node_name = node_name.substr(pos + 1);
      }
    }
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    auto splits = context->input(1);
    common::ReadyEventList ready_event_list;
#if HAVE_GPU
    ready_event_list.AddReadyEvent(std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context)));
#endif
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    auto splits_tensor = std::make_shared<TFTensor>(splits);
    auto enqueue_result = EnqueueTensorAlltoall(
        hvd_context, hvd_tensor, splits_tensor, ready_event_list, node_name, device,
        [context, done](const common::Status& status) {
#if HAVE_GPU
          auto hvd_event = status.event;
          if (hvd_event.event) {
            auto device_context = context->op_device_context();
            if (device_context != nullptr) {
                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
            }
          }
#endif
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
private:
  bool ignore_name_scope_;
}; // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("HorovodAlltoall").Device(DEVICE_CPU),
                        HorovodAlltoallOp);
#if HOROVOD_GPU_ALLTOALL
REGISTER_KERNEL_BUILDER(Name("HorovodAlltoall")
                            .Device(DEVICE_GPU)
                            .HostMemory("splits")
                            .HostMemory("received_splits"),
                        HorovodAlltoallOp);
#endif

REGISTER_OP("HorovodAlltoall")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("ignore_name_scope: bool = False")
    .Input("tensor: T")
    .Input("splits: int32")
    .Output("output: T")
    .Output("received_splits: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      c->set_output(1, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Alltoall on a tensor.

Arguments
    tensor:     A tensor to be distributed with all to all
    splits:     A list of integers in rank order describing how many elements
                in `tensor` to send to each worker.

Output
    output:           The collected tensor data from all workers.
    received_splits:  A list of integers in rank order describing how many
                      elements in `output` have been received from each worker.
)doc");

} // namespace tensorflow
} // namespace horovod
