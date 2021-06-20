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

#ifndef HOROVOD_GLOBAL_STATE_H
#define HOROVOD_GLOBAL_STATE_H

#include <queue>
#include <thread>

#include "fusion_buffer_manager.h"
#include "group_table.h"
#include "parameter_manager.h"
#include "response_cache.h"
#include "tensor_queue.h"
#include "timeline.h"
#include "utils/env_parser.h"

namespace horovod {
namespace common {

// Forward declaration
class Controller;

// The global state shared by threads.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct HorovodGlobalState {
  // An atomic boolean which is set to true when background thread is started.
  // This ensures that only one background thread is spawned.
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // Background thread running MPI communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  std::atomic_bool shut_down{false};

  // Timeline writer.
  Timeline timeline;

  // Flag indicating whether timeline enabled.
  bool timeline_enabled = false;

  // Flag indicating whether running elastic.
  bool elastic_enabled = false;

  // Flag indicating whether to mark cycles in the timeline.
  std::atomic_bool mark_cycles_in_timeline{false};

  ParameterManager parameter_manager;

  // Encapsulates the fusion buffers, handles resizing and auto-tuning of buffer
  // size.
  FusionBufferManager fusion_buffer;

  // Time point when last cycle started.
  std::chrono::steady_clock::time_point last_cycle_start;

  // Whether collective context has been completed on the background thread.
  std::atomic_bool initialization_done{false};

  std::shared_ptr<Controller> controller;

  TensorQueue tensor_queue;

  // Pointer to shared buffer for allgather
  void* shared_buffer = nullptr;

  // Current shared buffer size
  int64_t shared_buffer_size = 0;

  // LRU cache of Responses
  ResponseCache response_cache;

  // Number of responses that can be cached
  uint32_t cache_capacity = 1024;

  // Number of GPU streams to use
  int num_nccl_streams = 1;

  // Index of current GPU stream to use
  int current_nccl_stream = 0;

  // Information on registered groups.
  GroupTable group_table;

  // A LibType indicating what framework we are using to perform CPU operations.
  LibType cpu_operation;

  // A LibType indicating what framework we are using to perform controller
  // operations.
  LibType control_operation;

  // Number of ranks that did Join()
  int joined_size = 0;

  // If a rank is Joined, AllReduce uses temporary 0 tensors for it.
  bool joined = false;

  // Chunk size for MPI send/recv in Adasum allreduce. Some versions of Intel MPI
  // benefit from a smaller chunk size.
  int64_t adasum_mpi_chunk_size = 1<<30;

  // Enable use of batched d2d memcopy kernel on GPU
  bool batch_d2d_memcopies = true;

  // Flag indicating whether to prohibit groups from fusing
  bool disable_group_fusion = false;

  // Flag indicating whether to enable async completion
  bool enable_async_completion = false;

  ~HorovodGlobalState() {
    // Make sure that the destructor of the background thread is safe to
    // call. If a thread is still joinable (not detached or complete) its
    // destructor cannot be called.
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
  }
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_GLOBAL_STATE_H
