// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

#include "gloo_operations.h"

#include "gloo/allgather.h"
#include "gloo/allgatherv.h"
#include "gloo/allreduce.h"
#include "gloo/alltoallv.h"
#include "gloo/broadcast.h"
#include "gloo/math.h"
#include "gloo/types.h"

#include "../common.h"
#include "../global_state.h"

namespace horovod {
namespace common {

IGlooAlgorithms* GetAlgorithmsForType(DataType dtype,
                                      GlooContext* gloo_context) {
  switch (dtype) {
  case HOROVOD_UINT8:
    return new GlooAlgorithms<u_int8_t>(gloo_context);
  case HOROVOD_INT8:
    return new GlooAlgorithms<int8_t>(gloo_context);
  case HOROVOD_UINT16:
    return new GlooAlgorithms<u_int16_t>(gloo_context);
  case HOROVOD_INT16:
    return new GlooAlgorithms<int16_t>(gloo_context);
  case HOROVOD_INT32:
    return new GlooAlgorithms<int32_t>(gloo_context);
  case HOROVOD_INT64:
    return new GlooAlgorithms<int64_t>(gloo_context);
  case HOROVOD_FLOAT16:
    return new GlooAlgorithms<gloo::float16>(gloo_context);
  case HOROVOD_FLOAT32:
    return new GlooAlgorithms<float>(gloo_context);
  case HOROVOD_FLOAT64:
    return new GlooAlgorithms<double>(gloo_context);
  case HOROVOD_BOOL:
    return new GlooAlgorithms<bool>(gloo_context);
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in Gloo mode.");
  }
}

template <typename T>
GlooAlgorithms<T>::GlooAlgorithms(GlooContext* gloo_context)
    : gloo_context_(gloo_context) {}

template <typename T>
void GlooAlgorithms<T>::Allreduce(void* buffer_data, int num_elements) {
  gloo::AllreduceOptions opts(gloo_context_->ctx);
  opts.setOutput<T>(static_cast<T*>(buffer_data), (size_t) num_elements);

  void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
  opts.setReduceFunction(gloo::AllreduceOptions::Func(func));

  gloo::allreduce(opts);
}

template <typename T>
void GlooAlgorithms<T>::Allgather(void* buffer_data, void* buffer_out,
                                  int* recvcounts, int* displcmnts) {
  // create count index
  std::vector<size_t> counts(recvcounts, recvcounts + gloo_context_->ctx->size);

  gloo::AllgathervOptions opts(gloo_context_->ctx);
  opts.setInput<T>(static_cast<T*>(buffer_data) +
                       displcmnts[gloo_context_->ctx->rank],
                   counts[gloo_context_->ctx->rank]);
  opts.setOutput<T>(static_cast<T*>(buffer_out), counts);

  gloo::allgatherv(opts);
}

template <typename T>
void GlooAlgorithms<T>::Broadcast(void* buffer_data, int num_elements,
                                  int root_rank) {
  gloo::BroadcastOptions opts(gloo_context_->ctx);
  opts.setRoot(root_rank);
  opts.setOutput<T>(static_cast<T*>(buffer_data), (size_t) num_elements);
  gloo::broadcast(opts);
}

template <typename T>
void GlooAlgorithms<T>::Alltoall(void* buffer_data, void* buffer_out,
                                 std::vector<int64_t>& sendcounts,
                                 std::vector<int64_t>& recvcounts) {
  gloo::AlltoallvOptions opts(gloo_context_->ctx);
  opts.setInput<T>(static_cast<T*>(buffer_data), sendcounts);
  opts.setOutput<T>(static_cast<T*>(buffer_out), recvcounts);

  gloo::alltoallv(opts);
}

template <typename T> int GlooAlgorithms<T>::ElementSize() const {
  return sizeof(T);
}

GlooAllreduce::GlooAllreduce(GlooContext* gloo_context,
                             HorovodGlobalState* global_state)
    : AllreduceOp(global_state), gloo_context_(gloo_context) {}

Status GlooAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  WaitForData(entries);
  auto& first_entry = entries[0];

  const void* fused_input_data;
  void* buffer_data;
  int num_elements = (int)NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    size_t buffer_len;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*)first_entry.output->data();
    std::memcpy(buffer_data, first_entry.tensor->data(),
                (size_t)first_entry.tensor->size());
    fused_input_data = buffer_data;
  }

  if (response.prescale_factor() != 1.0) {
    // Execute prescaling op
    ScaleBuffer(response.prescale_factor(), entries, fused_input_data, buffer_data, num_elements);
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, GLOO_ALLREDUCE);
  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(first_entry.tensor->dtype(), gloo_context_));
  gloo_algos->Allreduce(buffer_data, num_elements);
  timeline.ActivityEndAll(entries);

  if (response.postscale_factor() != 1.0) {
    // Execute postscaling op
    ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data, num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool GlooAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

GlooAllgather::GlooAllgather(GlooContext* gloo_context,
                             HorovodGlobalState* global_state)
    : AllgatherOp(global_state), gloo_context_(gloo_context) {}

bool GlooAllgather::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

Status GlooAllgather::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  WaitForData(entries);

  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t*[entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t*[entries.size()];

  int global_size = global_state_->controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status =
      AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    /* Cleanup */
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }   
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts,
                           entry_component_offsets);

  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(first_entry.tensor->dtype(), gloo_context_));
  int element_size = gloo_algos->ElementSize();

  void* sendbuf = nullptr;
  void* buffer_data;

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    sendbuf = buffer_data;
    timeline.ActivityEndAll(entries);
  } else {
    // need to move input data to its corresponding location in the output
    sendbuf = (void*)first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    int buffer_offset = displcmnts[gloo_context_->ctx->rank] * element_size;
    std::memcpy((uint8_t*)buffer_data + buffer_offset, sendbuf,
                (size_t)first_entry.tensor->size());
    sendbuf = buffer_data;
  }

  // call gloo allgather api
  global_state_->timeline.ActivityStartAll(entries, GLOO_ALLGATHER);
  gloo_algos->Allgather(sendbuf, buffer_data, recvcounts, displcmnts);
  global_state_->timeline.ActivityEndAll(entries);

  // if multiple tensors are gathered, restore the sequence from output
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return Status::OK();
}

GlooBroadcast::GlooBroadcast(GlooContext* gloo_context,
                             HorovodGlobalState* global_state)
    : BroadcastOp(global_state), gloo_context_(gloo_context) {}

Status GlooBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  // for gloo broadcast, only output needs to be set if inplace

  void* data_ptr;
  if (global_state_->controller->GetRank() == e.root_rank) {
    data_ptr = (void*)e.tensor->data();
  } else {
    data_ptr = (void*)e.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, GLOO_BCAST);
  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(e.tensor->dtype(), gloo_context_));
  gloo_algos->Broadcast(data_ptr, (int)e.tensor->shape().num_elements(),
                        e.root_rank);
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool GlooBroadcast::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

GlooAlltoall::GlooAlltoall(GlooContext* gloo_context,
                           HorovodGlobalState* global_state)
    : AlltoallOp(global_state), gloo_context_(gloo_context) {}

Status GlooAlltoall::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];

  std::vector<int64_t> sdispls, rdispls;
  std::vector<int64_t> sendcounts, recvcounts;
  Status status = PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_ALLTOALL);

  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(e.tensor->dtype(), gloo_context_));
  gloo_algos->Alltoall((void*)e.tensor->data(), (void*)e.output->data(),
                       sendcounts, recvcounts);

  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool GlooAlltoall::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
