// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020 Intel Corporation
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

#include "ccl_operations.h"

namespace horovod {
namespace common {

// ************************************************************************************
// ************************************************************************************

// We keep a map of CCL specifics like stream and communicator.
// We assume the communication network/communicator per device stays
// unmodified during an enablement of the CCLContext.
struct ccl4hvd {
  ccl::stream stream_;
  ccl::communicator comm_;
};

// We assume there is only a single thread executing the CollOps.
class CCLOpContext {
public:
  // We use this for temporarily storing the queue between calls
  // This safes us from changing the API of existing classes elsewhere
  ccl4hvd* curr_;

private:
  std::unordered_map<int, ccl4hvd> contexts_;
  // CCL's KVS
  std::shared_ptr<ccl::kvs> kvs_;

  // Initialize CCL's kvs and broadcast to all peers.
  // We use HVD's controller for the broadcast.
  void InitKVS(const HorovodGlobalState* global_state) {
    if (global_state->controller->GetRank() == 0) {
      this->kvs_ = ccl::create_main_kvs();
      auto main_addr = this->kvs_->get_address();
      global_state->controller->Bcast((void*)main_addr.data(), main_addr.size(),
                                      0, Communicator::GLOBAL);
    } else {
      ccl::kvs::address_type main_addr;
      global_state->controller->Bcast((void*)main_addr.data(), main_addr.size(),
                                      0, Communicator::GLOBAL);
      this->kvs_ = ccl::create_kvs(main_addr);
    }
  }

public:
  // Return stream/communicator (by const reference) for given TensorTableEntry.
  // We keep one struct per device/host.
  // We memoize them in a map. If for the given e.device we already have a
  // struct we simply return it. Otherwise we initialize kvs if needed, create a
  // new struct, store and return it. Before returning the looked-up/new struct
  // we also set the curr_ pointer.
  const ccl4hvd& GetCCL4HVD(const TensorTableEntry& e,
                            const HorovodGlobalState* global_state) {
    auto resit = this->contexts_.find(e.device);
    if (resit == this->contexts_.end()) {
      // not found
      if (!this->kvs_)
        this->InitKVS(global_state);

      auto rank = global_state->controller->GetRank();
      auto size = global_state->controller->GetSize();

      assert(e.device == CPU_DEVICE_ID);
      auto stream = ccl::create_stream();
      auto device = ccl::create_device();
      auto context = ccl::create_context();
      resit = this->contexts_
                  .emplace(std::make_pair(
                      e.device, ccl4hvd{stream, ccl::create_communicator(
                                                    size, rank, device, context,
                                                    this->kvs_)}))
                  .first;
    }
    // temporarily store the ctxt.
    this->curr_ = &resit->second;
    return *this->curr_;
  }

  // wait for all events in queue
  inline void wait() {}

  typedef void* event_t;

  inline event_t do_memcpy(void* dstBuffer, size_t dstSize,
                           const void* srcBuffer, size_t srcSize) {
    if (dstSize < srcSize) {
      throw std::logic_error("Cannot copy larger buffer into smaller.");
    }
    std::memcpy(dstBuffer, srcBuffer, srcSize);
    return nullptr;
  }
}; // class CCLOpContext

// ************************************************************************************
// ************************************************************************************

CCLContext::CCLContext() : opctxt_(nullptr) {}

void CCLContext::Initialize() {
  ccl::init();
  opctxt_ = NewOpContext();
  enable_cache = GetBoolEnvOrDefault(HOROVOD_CCL_CACHE, false);

  LOG(DEBUG) << "CCL context initialized, enable_cache " << enable_cache;
}

void CCLContext::Finalize() {
  if (opctxt_) {
    delete opctxt_;
    opctxt_ = nullptr;
  }
  LOG(DEBUG) << "CCL context finalized.";
}

CCLOpContext* CCLContext::NewOpContext() { return new CCLOpContext(); }

// ************************************************************************************
// ************************************************************************************

namespace {

inline ccl::datatype GetCCLDataType(const std::shared_ptr<Tensor>& tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return ccl::datatype::uint8;
  case HOROVOD_INT8:
    return ccl::datatype::int8;
  case HOROVOD_UINT16:
    return ccl::datatype::uint16;
  case HOROVOD_INT16:
    return ccl::datatype::int16;
  case HOROVOD_FLOAT32:
    return ccl::datatype::float32;
  case HOROVOD_FLOAT64:
    return ccl::datatype::float64;
  case HOROVOD_INT32:
    return ccl::datatype::int32;
  case HOROVOD_INT64:
    return ccl::datatype::int64;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in CCL.");
  }
}

// ************************************************************************************
// ************************************************************************************

// used in single-rank shortcuts for allgather/broadcast: simply copy input to
// output tensor
Status cpyIn2Out(std::vector<TensorTableEntry>& entries, CCLOpContext* opctxt) {
  for (auto& e : entries) {
    if (e.output->data() != e.tensor->data()) {
      auto sz = e.tensor->size();
      opctxt->do_memcpy(const_cast<void*>(e.output->data()), sz,
                        e.tensor->data(), sz);
    }
  }
  opctxt->wait();
  return Status::OK();
}

// copy a single tensor from a buffer
// do *not* wait for completition if on a device
inline void
memcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                           const void* buffer_data_at_offset,
                           TensorTableEntry& e, size_t entry_size,
                           int64_t entry_offset, CCLOpContext* opctxt) {
  int8_t* outp = reinterpret_cast<int8_t*>(const_cast<void*>(e.output->data()));
  opctxt->do_memcpy(outp + entry_offset, e.output->size() - entry_offset,
                    buffer_data_at_offset, entry_size);
}

} // namespace

// ************************************************************************************
// ************************************************************************************

// Convenience API called by AllreduceOp
// do *not* wait for completition if on a device
void CCLAllreduce::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  this->ccl_context_->opctxt_->do_memcpy(buffer_data_at_offset,
                                         e.tensor->size(), e.tensor->data(),
                                         e.tensor->size());
}

// Convenience API called by AllreduceOp
// do *not* wait for completition if on a device
void CCLAllreduce::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  memcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e,
                             e.tensor->size(), 0, this->ccl_context_->opctxt_);
}

void CCLAllreduce::ScaleBuffer(double scale_factor,
                               const std::vector<TensorTableEntry>& entries,
                               const void* fused_input_data, void* buffer_data,
                               int64_t num_elements) {
  AllreduceOp::ScaleBuffer(scale_factor, entries, fused_input_data, buffer_data,
                           num_elements);
}

Status CCLAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                             const Response& response) {
  WaitForData(entries);

  auto& first_entry = entries[0];
  LOG(DEBUG) << "CCLAllreduce::Execute #entries: " << entries.size()
             << " device " << first_entry.device;

  auto& c4h =
      this->ccl_context_->opctxt_->GetCCL4HVD(first_entry, global_state_);

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    this->ccl_context_->opctxt_->wait();
    timeline.ActivityEndAll(entries);
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = const_cast<void*>(first_entry.output->data());
    buffer_len = (size_t)first_entry.output->size();
  }

  if (response.prescale_factor() != 1.0) {
    // Execute prescaling op
    ScaleBuffer(response.prescale_factor(), entries, fused_input_data,
                buffer_data, num_elements);
    fused_input_data = buffer_data; // for unfused, scale is done out of place
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, CCL_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || fused_input_data == buffer_data
                            ? buffer_data
                            : fused_input_data;
  
  auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();

  if (this->ccl_context_->enable_cache) {
    std::string match_id = "dt_" + DataType_Name(first_entry.tensor->dtype()) +
                           "_len_" + std::to_string(buffer_len);

    if (response.prescale_factor() != 1.0) {
      match_id += "_prescale_" + std::to_string(response.prescale_factor());
    }
    if (response.postscale_factor() != 1.0) {
      match_id += "_postscale_" + std::to_string(response.postscale_factor());
    }
    for (size_t idx = 0; idx < entries.size(); idx++) {
      match_id += "_" + entries[idx].tensor_name;
    }

    attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));
    attr.set<ccl::operation_attr_id::to_cache>(true);

    LOG(DEBUG) << "CCLAllreduce::Execute enable_cache"
               << " buffer_len: " << buffer_len
               << " send_buf: " << sendbuf
               << " recv_buf: " << buffer_data
               << " match_id: " << match_id;
  }
  else {
    attr.set<ccl::operation_attr_id::to_cache>(false);
  }

  ccl::allreduce((void*)sendbuf, buffer_data, num_elements,
                 GetCCLDataType(first_entry.tensor), ccl::reduction::sum,
                 c4h.comm_, c4h.stream_, attr)
      .wait();
  timeline.ActivityEndAll(entries);

  if (response.postscale_factor() != 1.0) {
    // Execute postscaling op
    ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data,
                num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    this->ccl_context_->opctxt_->wait();
    timeline.ActivityEndAll(entries);
  }

  LOG(DEBUG) << "CCLAllreduce::Execute done";

  return Status::OK();
}

// ************************************************************************************
// ************************************************************************************

// Convenience API called by AllGatherOp
// do *not* wait for completition if on a device
void CCLAllgather::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  this->ccl_context_->opctxt_->do_memcpy(buffer_data_at_offset,
                                         e.tensor->size(), e.tensor->data(),
                                         e.tensor->size());
}

// Convenience API called by AllGatherOp
// do *not* wait for completition if on a device
void CCLAllgather::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e,
    int64_t entry_offset, size_t entry_size) {
  memcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e, entry_size,
                             entry_offset, this->ccl_context_->opctxt_);
}

Status CCLAllgather::Execute(std::vector<TensorTableEntry>& entries,
                             const Response& response) {
  WaitForData(entries);

  auto& first_entry = entries[0];
  LOG(DEBUG) << "CCLAllgather::Execute #entries: " << entries.size()
             << " device " << first_entry.device;

  auto& timeline = global_state_->timeline;
  auto& c4h =
      this->ccl_context_->opctxt_->GetCCL4HVD(first_entry, global_state_);

  Status status = Status::OK();
  // shortcut for single rank
  if (global_state_->controller->GetSize() == 1) {
    int64_t** entry_component_sizes = nullptr;
    int* recvcounts = nullptr;
    status =
        AllocateOutput(entries, response, entry_component_sizes, recvcounts);
    return status.ok() ? cpyIn2Out(entries, this->ccl_context_->opctxt_)
                       : status;
  }

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

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (status.ok()) {
    timeline.ActivityEndAll(entries);
    SetDisplacements(recvcounts, displcmnts);
    SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts,
                             entry_component_offsets);

    int element_size =
        global_state_->controller->GetTypeSize(first_entry.tensor->dtype());

    const void* sendbuf = nullptr;
    void* buffer_data;
    int64_t num_elements = NumElements(entries);
    int64_t gather_size = 0;

    if (entries.size() > 1) {
      timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
      MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
      this->ccl_context_->opctxt_->wait();
      timeline.ActivityEndAll(entries);
    } else {
      if (first_entry.tensor->data() == first_entry.output->data()) {
        throw std::logic_error(
            "inplace allgather with single entry not implemented yet.");
      }
      sendbuf = first_entry.tensor->data();
      buffer_data = const_cast<void*>(first_entry.output->data());
    }

    std::vector<size_t> rcounts(global_size);
    for (unsigned int rc = 0; rc < global_size; rc++) {
      rcounts[rc] = recvcounts[rc] * element_size;
      gather_size += rcounts[rc];
    }

    global_state_->timeline.ActivityStartAll(entries, CCL_ALLGATHER);
    ccl::allgatherv(sendbuf != nullptr ? (void*)sendbuf : buffer_data,
                    num_elements * element_size, buffer_data, rcounts,
                    ccl::datatype::int8, c4h.comm_, c4h.stream_)
        .wait();
    global_state_->timeline.ActivityEndAll(entries);

    if (entries.size() > 1) {
      timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
      MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                            buffer_data, element_size, entries);
      this->ccl_context_->opctxt_->wait();
      timeline.ActivityEndAll(entries);
    }
    status = Status::OK();
  }

  /* Cleanup */
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;
  delete[] recvcounts;
  delete[] displcmnts;

  LOG(DEBUG) << "CCLAllgather::Execute done";

  return status;
}

// ************************************************************************************
// ************************************************************************************

Status CCLBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                             const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto& e = entries[0];
  LOG(DEBUG) << "CCLBroadcast::Execute #entries: " << entries.size()
             << " device " << e.device;
  auto& c4h = this->ccl_context_->opctxt_->GetCCL4HVD(e, global_state_);

  global_state_->timeline.ActivityStartAll(entries, CCL_BCAST);

  // shortcut for single rank
  if (global_state_->controller->GetSize() > 1) {
    // On root rank, CCL_Bcast sends data, on other ranks it receives data.
    const bool amroot = global_state_->controller->GetRank() == e.root_rank;
    size_t size = e.tensor->size();
    void* data_ptr = const_cast<void*>((amroot ? e.tensor : e.output)->data());

    ccl::broadcast(data_ptr, size, ccl::datatype::int8, e.root_rank, c4h.comm_,
                   c4h.stream_)
        .wait();
  }

  global_state_->timeline.ActivityEndAll(entries);
  LOG(DEBUG) << "CCLBroadcast::Execute done";

  return Status::OK();
}

// ************************************************************************************
// ************************************************************************************

Status CCLAlltoall::Execute(std::vector<TensorTableEntry>& entries,
                            const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];
  LOG(DEBUG) << "CCLAlltoall::Execute #entries: " << entries.size()
             << " device " << e.device;
  auto& c4h = this->ccl_context_->opctxt_->GetCCL4HVD(e, global_state_);

  global_state_->timeline.ActivityStartAll(entries, CCL_ALLTOALL);

  std::vector<size_t> sdispls, rdispls;
  std::vector<size_t> sendcounts, recvcounts;
  Status status =
      PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  const void* sendbuf = e.tensor->data();
  void* buffer_data = (void*)e.output->data();

  ccl::alltoallv(sendbuf, sendcounts, buffer_data, recvcounts,
                 GetCCLDataType(e.tensor), c4h.comm_, c4h.stream_)
      .wait();

  global_state_->timeline.ActivityEndAll(entries);
  LOG(DEBUG) << "CCLAlltoall::Execute done";

  return Status::OK();
}

} // namespace common
} // namespace horovod
