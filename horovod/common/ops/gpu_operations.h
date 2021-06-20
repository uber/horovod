// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#ifndef HOROVOD_GPU_OPERATIONS_H
#define HOROVOD_GPU_OPERATIONS_H

#include <queue>
#include <unordered_map>
#include <vector>

#if HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using gpuError_t = cudaError_t;
using gpuEvent_t = cudaEvent_t;
using gpuStream_t = cudaStream_t;
#elif HAVE_ROCM
#include <hip/hip_runtime_api.h>
using gpuError_t = hipError_t;
using gpuEvent_t = hipEvent_t;
using gpuStream_t = hipStream_t;
#endif

#include "collective_operations.h"
#include "../thread_pool.h"

namespace horovod {
namespace common {

class GPUContext {
public:
  GPUContext();
  ~GPUContext();

  void Finalize();

  // The GPU stream used for data transfers and within-allreduce operations.
  // A naive implementation would use the TensorFlow StreamExecutor GPU
  // stream. However, the allreduce and allgather require doing memory copies
  // and kernel executions (for accumulation of values on the GPU). However,
  // the subsequent operations must wait for those operations to complete,
  // otherwise MPI (which uses its own stream internally) will begin the data
  // transfers before the GPU calls are complete. In order to wait for those
  // GPU operations, if we were using the TensorFlow stream, we would have to
  // synchronize that stream; however, other TensorFlow threads may be
  // submitting more work to that stream, so synchronizing on it can cause the
  // allreduce to be delayed, waiting for compute totally unrelated to it in
  // other parts of the graph. Overlaying memory transfers and compute during
  // backpropagation is crucial for good performance, so we cannot use the
  // TensorFlow stream, and must use our own stream.
  std::vector<std::unordered_map<int, gpuStream_t>> streams;

  void ErrorCheck(std::string op_name, gpuError_t gpu_result);

  void RecordEvent(std::queue<std::pair<std::string, Event>>& event_queue, std::string name,
                   gpuStream_t& stream);

  Event RecordEvent(gpuStream_t& stream);

  void ReleaseEvent(Event event);

  void WaitForEvents(std::queue<std::pair<std::string, Event>>& event_queue,
                     const std::vector<TensorTableEntry>& entries, Timeline& timeline,
                     const std::function<void()>& error_check_callback = nullptr,
                     bool elastic = false);

  void ClearEvents(std::queue<std::pair<std::string, Event>>& event_queue,
                   const std::vector<TensorTableEntry>& entries, Timeline& timeline,
                   const std::function<void()>& error_check_callback = nullptr,
                   bool elastic = false);

  void StreamCreate(gpuStream_t *stream);
  void StreamSynchronize(gpuStream_t stream);

  int GetDevice();

  void SetDevice(int device);

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count, gpuStream_t stream);
  void MemcpyAsyncH2D(void* dst, const void* src, size_t count, gpuStream_t stream);
  void MemcpyAsyncD2H(void* dst, const void* src, size_t count, gpuStream_t stream);

  void ScaleBufferImpl(const void* fused_input_data, void* buffer_data, int64_t num_elements,
                       double scale_factor, DataType dtype, gpuStream_t stream);

  // Thread pool for finalizer threads
  ThreadPool finalizer_thread_pool;

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

class GPUOpContext {
public:
  GPUOpContext(GPUContext* context,
               HorovodGlobalState* global_state);

  void InitGPU(const std::vector<TensorTableEntry>& entries);

  void InitGPUQueue(const std::vector<TensorTableEntry>& entries, const Response& response);

  Status FinalizeGPUQueue(std::vector<TensorTableEntry>& entries, bool free_host_buffer = true,
                          const std::function<void()>& error_check_callback = nullptr);

  // GPU events are used as an alternative to host-device synchronization (which stalls the GPU pipeline)
  // for the purpose of recording timing on the Horovod timeline.
  //
  // When an event we wish to record occurs (for example, NCCL_ALLREDUCE), the event is enqueued. After the entire
  // operation completes, a background thread is spawned to synchronize on the events in the queue and record
  // timing, while allowing Horovod to continue processing additional tensors.
  //
  // For more information of CUDA Events, see:
  // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
  std::queue<std::pair<std::string, Event>> event_queue;

  gpuStream_t* stream;
  void* host_buffer = nullptr;

private:
  GPUContext* gpu_context_;
  HorovodGlobalState* global_state_;
};

class GPUAllreduce : public AllreduceOp {
public:
  GPUAllreduce(GPUContext* context,
               HorovodGlobalState* global_state);

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
#if HAVE_CUDA
  void MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
                                    void*& buffer_data, size_t& buffer_len) override;

  void MemcpyOutFusionBuffer(const void* buffer_data, std::vector<TensorTableEntry>& entries) override;
#endif

  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e, void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset, TensorTableEntry& e) override;

  void ScaleBuffer(double scale_factor, const std::vector<TensorTableEntry>& entries,
                   const void* fused_input_data, void* buffer_data, int64_t num_elements);

  GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;

};

class GPUAllgather : public AllgatherOp {
public:
  GPUAllgather(GPUContext* context,
               HorovodGlobalState* global_state);

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e, void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset, TensorTableEntry& e,
                                  int64_t entry_offset, size_t entry_size) override;

  GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;
};

class GPUBroadcast : public BroadcastOp {
public:
  GPUBroadcast(GPUContext* context,
               HorovodGlobalState* global_state);

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  struct GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;
};

class GPUAlltoall : public AlltoallOp {
public:
  GPUAlltoall(GPUContext* context,
              HorovodGlobalState* global_state);
  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
protected:
  GPUContext* gpu_context_;
  GPUOpContext gpu_op_context_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_GPU_OPERATIONS_H
