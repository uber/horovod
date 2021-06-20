// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Modifications copyright (C) 2019 Intel Corporation
// Modifications copyright Microsoft
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

#include "operations.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common.h"
#include "fusion_buffer_manager.h"
#include "global_state.h"
#include "hashes.h"
#include "logging.h"
#include "message.h"
#include "ops/operation_manager.h"
#include "parameter_manager.h"
#include "timeline.h"
#include "utils/env_parser.h"
#include "nvtx_op_range.h"

#if HAVE_MPI
#define OMPI_SKIP_MPICXX
#include "mpi.h"
#include "mpi/mpi_context.h"
#include "mpi/mpi_controller.h"
#include "ops/mpi_operations.h"
#include "ops/adasum_mpi_operations.h"
#endif

#if HAVE_GPU
#include "ops/gpu_operations.h"
#if HAVE_MPI
#include "ops/mpi_gpu_operations.h"
#endif
#endif

#if HAVE_NCCL
#include "ops/nccl_operations.h"
#if HAVE_MPI
#include "ops/adasum_gpu_operations.h"
#endif
#endif

#if HAVE_DDL && HAVE_MPI
#include "mpi/ddl_mpi_context_manager.h"
#include "ops/ddl_operations.h"
#endif

#if HAVE_CCL
#include "ops/ccl_operations.h"
#endif

#if HAVE_GLOO
#include "gloo/gloo_controller.h"
#include "ops/gloo_operations.h"
#endif

/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * whichever hardware-optimized communication libraries are enabled.
 *
 * The primary logic of the allreduce, allgather and broadcast currently
 * support in MPI, NCCL, CUDA/ROCm, Gloo, oneCCL, DDL. The background thread
 * which facilitates controller operations is run in BackgroundThreadLoop().
 * The provided ops are:
 *      - HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all processes in the global communicator.
 *      - HorovodAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all processes in the
 *          global communicator.
 *      - HorovodBroadcast:
 *          Perform a broadcast on a Tensor, broadcasting Tensor
 *          value from root rank to all other ranks.
 *
 * Additionally, this library provides C APIs to initialize Horovod and query
 * rank, local rank and world size.  These are used in Python directly through
 * ctypes.
 */

namespace horovod {
namespace common {

namespace {

// All the Horovod state that must be stored globally per-process.
HorovodGlobalState horovod_global;

#if HAVE_MPI
MPIContext mpi_context;
#endif

#if HAVE_GLOO
GlooContext gloo_context;
#endif

#if HAVE_GPU
GPUContext gpu_context;
#endif

#if HAVE_NCCL
NCCLContext nccl_context;
#endif

#if HAVE_DDL
DDLContext ddl_context;
#endif

#if HAVE_CCL
CCLContext ccl_context;
#endif

std::unique_ptr<OperationManager> op_manager;

OperationManager* CreateOperationManager(HorovodGlobalState& state) {
  // Order of these operations is very important. Operations will be checked
  // sequentially from the first to the last. The first 'Enabled' operation will
  // be executed.
  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops;
  std::vector<std::shared_ptr<AllreduceOp>> adasum_ops;
  std::vector<std::shared_ptr<AlltoallOp>> alltoall_ops;

#if HAVE_MPI && HAVE_GPU
  if (mpi_context.IsEnabled()) {
#if HOROVOD_GPU_ALLREDUCE == 'M'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new MPI_GPUAllreduce(&mpi_context, &gpu_context, &state)));

#elif HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
    adasum_ops.push_back(std::shared_ptr<AllreduceOp>(new AdasumGpuAllreduceOp(&mpi_context, &nccl_context, &gpu_context, &state)));

    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new NCCLHierarchicalAllreduce(
            &nccl_context, &mpi_context, &gpu_context, &state)));

#elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new DDLAllreduce(&ddl_context, &gpu_context, &state)));
#endif

#if HOROVOD_GPU_ALLGATHER == 'M'
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPI_GPUAllgather(&mpi_context, &gpu_context, &state)));
#endif
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPIHierarchicalAllgather(&mpi_context, &state)));

#if HOROVOD_GPU_ALLTOALL == 'M'
    alltoall_ops.push_back(std::shared_ptr<AlltoallOp>(
        new MPI_GPUAlltoall(&mpi_context, &gpu_context, &state)));
#endif
  }
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new NCCLAllreduce(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_BROADCAST == 'N'
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new NCCLBroadcast(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLGATHER == 'N'
  allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
      new NCCLAllgather(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLTOALL == 'N'
  alltoall_ops.push_back(std::shared_ptr<AlltoallOp>(
      new NCCLAlltoall(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_GLOO
  if (gloo_context.IsEnabled()) {
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new GlooAllreduce(&gloo_context, &state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new GlooAllgather(&gloo_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new GlooBroadcast(&gloo_context, &state)));
    alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new GlooAlltoall(&gloo_context, &state)));
  }
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL) {
    allreduce_ops.push_back(
        std::make_shared<CCLAllreduce>(&ccl_context, &state));
    allgather_ops.push_back(
        std::make_shared<CCLAllgather>(&ccl_context, &state));
    broadcast_ops.push_back(
        std::make_shared<CCLBroadcast>(&ccl_context, &state));
    alltoall_ops.push_back(
        std::make_shared<CCLAlltoall>(&ccl_context, &state));
  }
#endif

#if HAVE_MPI
  if (mpi_context.IsEnabled()){
    adasum_ops.push_back(
        std::shared_ptr<AllreduceOp>(new AdasumMPIAllreduceOp(&mpi_context, &state)));
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new MPIAllreduce(&mpi_context,&state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new MPIAllgather(&mpi_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new MPIBroadcast(&mpi_context, &state)));
    alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new MPIAlltoall(&mpi_context, &state)));
  }
#endif

  std::shared_ptr<JoinOp> join_op(new JoinOp(&state));
  std::shared_ptr<ErrorOp> error_op(new ErrorOp(&state));

  return new OperationManager(&state.parameter_manager, allreduce_ops,
                              allgather_ops, broadcast_ops, alltoall_ops,
                              join_op, adasum_ops, error_op);
}

// Process a Response by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(Response response, HorovodGlobalState& state) {
  std::vector<TensorTableEntry> entries;
  auto& timeline = horovod_global.timeline;
  if (response.response_type() != Response::JOIN) {
    horovod_global.tensor_queue.GetTensorEntriesFromResponse(response, entries,
                                                             state.joined);

    for (auto& e : entries) {
      timeline.Start(e.tensor_name, response.response_type(), e.tensor->size());
    }

    if (entries.size() > 1) {
      auto first_entry = entries[0];
      // Note: it is OK for different entries to come from different frameworks
      // since buffer allocated here is guaranteed to survive at least till the
      // end of this operation.
      Status status = horovod_global.fusion_buffer.InitializeBuffer(
          horovod_global.controller->TensorFusionThresholdBytes(),
          first_entry.device, first_entry.context,
          horovod_global.current_nccl_stream,
          [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
          [&]() { timeline.ActivityEndAll(entries); });
      if (!status.ok()) {
        LOG(DEBUG, horovod_global.controller->GetRank()) << "InitializeBuffer Failed";
        for (auto& e : entries) {
          timeline.End(e.tensor_name, nullptr);
          e.FinishWithCallback(status);
        }
        return;
      }
    }
  }

  Status status;
  try {
    status = op_manager->ExecuteOperation(entries, response);
  } catch (const std::exception& ex) {
    LOG(DEBUG, horovod_global.controller->GetRank()) << "ExecuteOperation Failed";
    status = Status::UnknownError(ex.what());
  }

  if (!status.in_progress()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, status.ok() ? e.output : nullptr);
      e.FinishWithCallback(status);
    }
  }
}

// The background thread loop coordinates all the controller processes and the
// tensor reductions. The design of the communicator mechanism is limited by a
// few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a
//      single thread. Since TensorFlow may use several threads for graph
//      processing, this means we must have our own dedicated thread for
//      dealing with MPI.
//      2. We want to gracefully handle errors, when all processes do not
//      properly agree upon what should happen (such as mismatched types or
//      shapes). To do so requires every process to know about the shapes
//      and types of the relevant tensors on the other processes.
//      3. The reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never
//      make progress if we have a thread pool limit.
bool RunLoopOnce(HorovodGlobalState& state);

void BackgroundThreadLoop(HorovodGlobalState& state) {
#if HAVE_CCL
  // Initialize ccl context
  if (state.cpu_operation == LibType::CCL) {
    ccl_context.Initialize();
  }
#endif

#if HAVE_MPI
  // Initialize mpi context
#if HAVE_DDL
  // If DDL is enabled, let DDL ops manage MPI environment.
  auto mpi_ctx_manager = DDL_MPIContextManager(ddl_context, gpu_context);
#else
  // Otherwise, let MPI ops be in charge.
  auto mpi_ctx_manager = MPIContextManager();
#endif
  mpi_context.Initialize(state.controller->GetRanks(), mpi_ctx_manager);
#endif

#if HAVE_GLOO
#if HAVE_MPI
    if (mpi_context.IsEnabled()) {
      // Initialize gloo context if mpi context is available
      gloo_context.InitializeFromMPI(mpi_context, ParseGlooIface());
    }
    else
#endif
    {
      gloo_context.Initialize(ParseGlooIface());
    }
#endif
  // Initialize controller
  state.controller->Initialize();

  bool is_coordinator = state.controller->IsCoordinator();
  bool is_homogeneous = state.controller->IsHomogeneous();
  int size = state.controller->GetSize();
  int local_size = state.controller->GetLocalSize();
  int local_rank = state.controller->GetLocalRank();

  // Set background thread affinity
  parse_and_set_affinity(std::getenv(HOROVOD_THREAD_AFFINITY), local_size, local_rank);

#if HAVE_GPU
  // Set number of GPU streams to use
  auto horovod_num_nccl_streams =
      std::getenv(HOROVOD_NUM_NCCL_STREAMS);
  if (horovod_num_nccl_streams != nullptr &&
      std::stol(horovod_num_nccl_streams, nullptr, 10) > 0) {
    state.num_nccl_streams = std::atoi(horovod_num_nccl_streams);
  }

#if HAVE_NCCL
  nccl_context.nccl_comms.resize(state.num_nccl_streams);
#endif
  gpu_context.streams.resize(state.num_nccl_streams);

  // Create finalizer thread pool (one thread per stream)
  gpu_context.finalizer_thread_pool.create(state.num_nccl_streams);
#endif

#if HAVE_NVTX
  if (GetBoolEnvOrDefault(HOROVOD_DISABLE_NVTX_RANGES, false)) {
    NvtxOpRange::nvtx_ops_handle.Disable();
    horovod_global.timeline.DisableNvtx();
  }
#endif // HAVE_NVTX

  // Open the timeline file on coordinator.
  bool should_enable_timeline = false;
  auto timeline_env = std::getenv(HOROVOD_TIMELINE);
  if (timeline_env) {
    if (is_coordinator) {
      auto horovod_timeline = std::string(timeline_env);
      if (horovod_timeline != "DYNAMIC") {
        state.timeline.Initialize(horovod_timeline,
                                  static_cast<unsigned int>(size));
      } else {
        state.timeline.Initialize("",
                                  static_cast<unsigned int>(size));
      }
    }
    should_enable_timeline = true;
  }
  state.controller->SetTimelineEnabled(should_enable_timeline);

  SetBoolFromEnv(HOROVOD_ELASTIC, state.elastic_enabled, true);

  ParseStallInspectorFromEnv(state.controller->GetStallInspector());
  bool mark_cycles = false;
  SetBoolFromEnv(HOROVOD_TIMELINE_MARK_CYCLES, mark_cycles,
                 true);
  state.controller->SetMarkCyclesInTimelinePending(mark_cycles);
  state.mark_cycles_in_timeline = mark_cycles;

  // Override Tensor Fusion threshold, if it's set.
  state.parameter_manager.SetTensorFusionThresholdBytes(128 * 1024 * 1024);
  auto horovod_fusion_threshold = std::getenv(HOROVOD_FUSION_THRESHOLD);
  if (horovod_fusion_threshold != nullptr) {
    int64_t threshold = std::strtol(horovod_fusion_threshold, nullptr, 10);
    state.parameter_manager.SetTensorFusionThresholdBytes(threshold, true);
  }

  // Override the cycle time.
  state.parameter_manager.SetCycleTimeMs(1);
  auto horovod_cycle_time = std::getenv(HOROVOD_CYCLE_TIME);
  if (horovod_cycle_time != nullptr) {
    state.parameter_manager.SetCycleTimeMs(
        std::strtof(horovod_cycle_time, nullptr), true);
  }

  // Override response cache capacity, if it's set.
  state.parameter_manager.SetCacheEnabled(true);
  auto horovod_cache_capacity = std::getenv(HOROVOD_CACHE_CAPACITY);
  if (horovod_cache_capacity != nullptr) {
    uint32_t cache_capacity = std::strtol(horovod_cache_capacity, nullptr, 10);
    state.cache_capacity = cache_capacity;
    state.parameter_manager.SetCacheEnabled(cache_capacity > 0, true);
  }
  state.response_cache.set_capacity(
      (int)state.parameter_manager.CacheEnabled() * state.cache_capacity);

  // Set flag for hierarchical allgather. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allgather =
      std::getenv(HOROVOD_HIERARCHICAL_ALLGATHER);
  state.parameter_manager.SetHierarchicalAllgather(false);
  if (horovod_hierarchical_allgather != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allgather, nullptr, 10) > 0 &&
                 (size != local_size);
    state.parameter_manager.SetHierarchicalAllgather(value, true);
  }
  // Set flag for hierarchical allreduce. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allreduce =
      std::getenv(HOROVOD_HIERARCHICAL_ALLREDUCE);
  state.parameter_manager.SetHierarchicalAllreduce(false);
  if (horovod_hierarchical_allreduce != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allreduce, nullptr, 10) > 0 &&
                 (size != local_size);
    state.parameter_manager.SetHierarchicalAllreduce(value, true);
  }

#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
  // Hierarchical allreduce is not supported without NCCL or DDL
  state.parameter_manager.SetHierarchicalAllreduce(false, true);
#endif

  // Issue warning if hierarchical allreduce is enabled in heterogeneous cluster
  if (is_coordinator &&
      (state.parameter_manager.HierarchicalAllreduce() ||
       state.parameter_manager.HierarchicalAllgather()) &&
      !is_homogeneous) {
    std::cerr
        << "WARNING: Using different number of ranks per node might cause "
           "performance loss in hierarchical allgather and "
           "hierarchical allreduce. Consider assigning the same "
           "number of ranks to each node, or disabling hierarchical "
           "allgather and hierarchical allreduce.";
  }

  // Set flag to control use of batched memcopy kernel on GPU
  auto horovod_batch_d2d_memcopies =
      std::getenv(HOROVOD_BATCH_D2D_MEMCOPIES);
  if (horovod_batch_d2d_memcopies != nullptr &&
      std::strtol(horovod_batch_d2d_memcopies, nullptr, 10) == 0) {
    state.batch_d2d_memcopies = false;
  }

  // Check if group fusion should be disabled
  SetBoolFromEnv(HOROVOD_DISABLE_GROUP_FUSION, state.disable_group_fusion, true);

  // Check if async completion should be enabled
  SetBoolFromEnv(HOROVOD_ENABLE_ASYNC_COMPLETION, state.enable_async_completion, true);

  // Enable auto-tuning.
  auto horovod_autotune = std::getenv(HOROVOD_AUTOTUNE);
  if (horovod_autotune != nullptr &&
      std::strtol(horovod_autotune, nullptr, 10) > 0) {
    auto horovod_autotune_log = std::getenv(HOROVOD_AUTOTUNE_LOG);
    state.parameter_manager.Initialize(state.controller->GetRank(), RANK_ZERO,
                                       horovod_autotune_log != nullptr
                                           ? std::string(horovod_autotune_log)
                                           : "");
    state.parameter_manager.SetAutoTuning(true);
  }

  // Set chunk size for MPI based Adasum allreduce algorithms
  auto horovod_adasum_mpi_chunk_size = std::getenv(HOROVOD_ADASUM_MPI_CHUNK_SIZE);
  if (horovod_adasum_mpi_chunk_size != nullptr) {
    state.adasum_mpi_chunk_size = std::strtol(horovod_adasum_mpi_chunk_size, nullptr, 10);
  }

  op_manager.reset(CreateOperationManager(state));

  // Signal that initialization is completed.
  state.initialization_done = true;
  LOG(INFO, horovod_global.controller->GetRank()) << "Horovod Initialized";

  // Iterate until shutdown.
  try {
    while (RunLoopOnce(state));
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Horovod background loop uncaught exception: " << ex.what();
  }

    // Finalize all contexts
#if HAVE_NCCL
  nccl_context.ShutDown();
#endif

#if HAVE_GLOO
  gloo_context.Finalize();
#endif

  LOG(DEBUG, horovod_global.controller->GetRank()) << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

  // Notify all outstanding operations that Horovod has been shut down
  // and finalize tensor queue.
  horovod_global.tensor_queue.FinalizeTensorQueue(SHUT_DOWN_ERROR);

#if HAVE_GPU
  gpu_context.Finalize();
#endif

#if HAVE_MPI
  mpi_context.Finalize(mpi_ctx_manager);
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL){
    ccl_context.Finalize();
  }
#endif

}

bool RunLoopOnce(HorovodGlobalState& state) {
  // This delay determines thread frequency and communication message latency
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = state.last_cycle_start +
                        std::chrono::microseconds(long(
                            state.parameter_manager.CycleTimeMs() * 1000.)) -
                        start_time;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

  if (state.mark_cycles_in_timeline) {
    // Mark start of the new cycle.
    state.timeline.MarkCycleStart();
  }

  auto response_list =
      state.controller->ComputeResponseList(horovod_global.shut_down, state);

  state.mark_cycles_in_timeline =
      state.controller->MarkCyclesInTimelinePending();

  // Get tensor name and size data for autotuning.
  int64_t total_tensor_size = 0;
  std::vector<std::string> tensor_names;
  if (state.parameter_manager.IsAutoTuning()) {
    total_tensor_size = horovod_global.tensor_queue.GetTensorDataForAutotuner(
        response_list, tensor_names);
  }

  // Perform the collective operation. All nodes should end up performing
  // the same operation.
  int rank = state.controller->GetRank();
  for (auto& response : response_list.responses()) {
    if (!state.group_table.empty()) {
      // Deregister any completed groups
      state.group_table.DeregisterGroups(response.tensor_names());
    }

    LOG(TRACE, rank) << "Performing " << response.tensor_names_string();
    LOG(TRACE, rank) << "Processing " << response.tensor_names().size()
                     << " tensors";
    PerformOperation(response, horovod_global);
    LOG(TRACE, rank) << "Finished performing "
                     << response.tensor_names_string();
  }

  if (state.parameter_manager.IsAutoTuning()) {
    bool should_sync =
        state.parameter_manager.Update(tensor_names, total_tensor_size);

    if (should_sync) {
      state.controller->SynchronizeParameters();
    }
  }

  return !response_list.shutdown();
}

// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce(const int* ranks, int nranks) {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.control_operation = ParseControllerOpsFromEnv();
    horovod_global.cpu_operation = ParseCPUOpsFromEnv();
#if HAVE_MPI
    // Enable mpi is it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::MPI ||
        horovod_global.control_operation == LibType::MPI) {
      mpi_context.Enable();
    }

    if (horovod_global.control_operation == LibType::MPI){
      horovod_global.controller.reset(new MPIController(
          horovod_global.response_cache,
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, horovod_global.group_table,
          mpi_context));
      horovod_global.controller->SetRanks(ranks, nranks);
    }
#endif

#if HAVE_GLOO
    // Enable gloo is it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::GLOO ||
        horovod_global.control_operation == LibType::GLOO) {
      gloo_context.Enable();
    }

    if (horovod_global.control_operation == LibType::GLOO) {
      horovod_global.controller.reset(new GlooController(
          horovod_global.response_cache,
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, horovod_global.group_table,
          gloo_context));
    }
#endif
    // Reset initialization flag
    horovod_global.initialization_done = false;
    horovod_global.background_thread = std::thread(
        BackgroundThreadLoop, std::ref(horovod_global));
  }

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  LOG(DEBUG) << "Background thread init done";
}

} // namespace

Status CheckInitialized() {
  if (!horovod_global.initialization_done) {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

void horovod_init(const int* ranks, int nranks) {
  InitializeHorovodOnce(ranks, nranks);
}

#if HAVE_MPI
void horovod_init_comm(MPI_Comm comm) {
  MPI_Comm_dup(comm, &mpi_context.mpi_comm);
  InitializeHorovodOnce(nullptr, 0);
}
#endif

void horovod_shutdown() {
  if (horovod_global.background_thread.joinable()) {
    horovod_global.timeline.Shutdown();
    horovod_global.shut_down = true;
    horovod_global.background_thread.join();

    // Reset the initialization flag to allow restarting with horovod_init(...)
    horovod_global.initialize_flag.clear();
    horovod_global.shut_down = false;
    horovod_global.initialization_done = false;
  }
}

bool horovod_is_initialized() {
  return horovod_global.initialization_done;
}

int horovod_start_timeline(const char* file_name, bool mark_cycles) {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  if (!horovod_global.controller->TimelineEnabled()) {
    return -2;
  }
  bool is_coordinator = horovod_global.controller->IsCoordinator();
  if (is_coordinator) {
    horovod_global.timeline.Initialize(std::string(file_name), horovod_global.controller->GetSize());
    horovod_global.timeline.SetPendingTimelineFile(std::string(file_name));
  }
  horovod_global.controller->SetMarkCyclesInTimelinePending(mark_cycles);
  return 1;
}

int horovod_stop_timeline() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  if(!horovod_global.controller->TimelineEnabledPending()){
    LOG(INFO) << " Timeline is already stopped. Please start timeline before stopping it.";
    return 1;
  }
  bool is_coordinator = horovod_global.controller->IsCoordinator();
  if (is_coordinator) {
      horovod_global.timeline.SetPendingTimelineFile(std::string(""));
  }
  return 1;
}

int horovod_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetRank();
}

int horovod_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetLocalRank();
}

int horovod_cross_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetCrossRank();
}

int horovod_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetSize();
}

int horovod_local_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetLocalSize();
}

int horovod_cross_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetCrossSize();
}

bool horovod_is_homogeneous() {
  return horovod_global.controller->IsHomogeneous();
}

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }

#if HAVE_MPI
  auto mpiController =
      std::dynamic_pointer_cast<MPIController>(horovod_global.controller);
  return mpiController->IsMpiThreadsSupported() ? 1 : 0;
#endif

  return -1;
}

bool horovod_mpi_enabled() {
#if HAVE_MPI
  return mpi_context.IsEnabled();
#else
  return false;
#endif
}

bool horovod_mpi_built() {
#if HAVE_MPI
  return true;
#else
  return false;
#endif
}

bool horovod_gloo_enabled() {
#if HAVE_GLOO
  return gloo_context.IsEnabled();
#else
  return false;
#endif
}

bool horovod_gloo_built() {
#if HAVE_GLOO
  return true;
#else
  return false;
#endif
}

int horovod_nccl_built() {
#if HAVE_NCCL
  return NCCL_VERSION_CODE;
#else
  return 0;
#endif
}

bool horovod_ddl_built() {
#if HAVE_DDL
  return true;
#else
  return false;
#endif
}

bool horovod_ccl_built() {
#if HAVE_CCL
  return true;
#else
  return false;
#endif
}

bool horovod_cuda_built() {
#if HAVE_CUDA
  return true;
#else
  return false;
#endif
}

bool horovod_rocm_built() {
#if HAVE_ROCM
  return true;
#else
  return false;
#endif
}

int horovod_reduce_op_average() {
  return ReduceOp::AVERAGE;
}

int horovod_reduce_op_sum() {
  return ReduceOp::SUM;
}

int horovod_reduce_op_adasum() {
  return ReduceOp::ADASUM;
}

}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              ReadyEventList ready_event_list,
                              std::string name, const int device,
                              StatusCallback callback,
                              ReduceOp reduce_op,
                              double prescale_factor,
                              double postscale_factor) {
  // Wrap inputs in std::vector and pass onto multi tensor implementation
  std::vector<std::shared_ptr<OpContext>> contexts;
  std::vector<std::shared_ptr<Tensor>> tensors;
  std::vector<std::shared_ptr<Tensor>> outputs;
  std::vector<ReadyEventList> ready_event_lists;
  std::vector<std::string> names;
  std::vector<StatusCallback> callbacks;

  contexts.emplace_back(std::move(context));
  tensors.emplace_back(std::move(tensor));
  outputs.emplace_back(std::move(output));
  ready_event_lists.emplace_back(std::move(ready_event_list));
  names.emplace_back(std::move(name));
  callbacks.emplace_back(std::move(callback));

  return EnqueueTensorAllreduces(contexts, tensors, outputs, ready_event_lists,
                                 names, device, callbacks, reduce_op,
                                 prescale_factor, postscale_factor);
}

Status EnqueueTensorAllreduces(std::vector<std::shared_ptr<OpContext>>& contexts,
                               std::vector<std::shared_ptr<Tensor>>& tensors,
                               std::vector<std::shared_ptr<Tensor>>& outputs,
                               std::vector<ReadyEventList>& ready_event_lists,
                               std::vector<std::string>& names,
                               const int device,
                               std::vector<StatusCallback>& callbacks,
                               ReduceOp reduce_op,
                               double prescale_factor,
                               double postscale_factor) {
  Status status;

  if (reduce_op == ReduceOp::AVERAGE) {
#if !HAVE_ROCM
    // Averaging happens via postscale_factor
    postscale_factor /= horovod_global.controller->GetSize();
#else
    LOG(ERROR, horovod_global.controller->GetRank()) << "Enqueuing AVERAGE allreduce is not allowed.";
    return status.Aborted("AVERAGE not allowed.");
#endif
  } else if (reduce_op == ReduceOp::ADASUM) {
#if HAVE_NCCL && !HAVE_ROCM
    if (device != CPU_DEVICE_ID) {
      // Averaging by local size happens via postscale_factor
      postscale_factor /= horovod_global.controller->GetLocalSize();
    }
#endif
  }

  std::vector<Request> messages;
  std::vector<TensorTableEntry> entries;
  messages.reserve(tensors.size());
  entries.reserve(tensors.size());

  for (int n = 0; n < tensors.size(); ++n) {
    Request message;
    message.set_request_rank(horovod_global.controller->GetRank());
    message.set_tensor_name(names[n]);
    message.set_tensor_type(tensors[n]->dtype());
    message.set_device(device);
    message.set_prescale_factor(prescale_factor);
    message.set_postscale_factor(postscale_factor);

    if (reduce_op == ReduceOp::ADASUM) {
      message.set_request_type(Request::ADASUM);
    } else {
      message.set_request_type(Request::ALLREDUCE);
    }

    message.set_tensor_shape(tensors[n]->shape().to_vector());
    messages.push_back(std::move(message));

    TensorTableEntry e;
    e.tensor_name = names[n];
    e.context = std::move(contexts[n]);
    // input and output can be the same, only move when safe
    if (tensors[n] != outputs[n]) {
      e.tensor = std::move(tensors[n]);
      e.output = std::move(outputs[n]);
    } else {
      e.tensor = tensors[n];
      e.output = outputs[n];
    }
    e.ready_event_list = std::move(ready_event_lists[n]);
    e.device = device;
    e.callback = std::move(callbacks[n]);

    entries.push_back(std::move(e));
  }

  // Start appropriate NVTX range
  if (tensors.size() == 1) {
    auto& e = entries[0];
    e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodAllreduce, e.tensor->size());
  } else {
    auto total_size =
        std::accumulate(entries.begin(), entries.end(), 0ll,
                        [](int64_t size_sum, const TensorTableEntry& e) {
                          return size_sum + e.tensor->size();
                        });
    SharedNvtxOpRange range;
    range.Start(RegisteredNvtxOp::HorovodGroupedAllreduce, total_size);
    for (auto& e : entries) {
      e.nvtx_op_range = range;
    }
  }

  std::string tensors_enqueued;
  for (const auto& n : names) {
    tensors_enqueued += n + "; ";
  }
  LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << tensors_enqueued;

  // Only create groups larger than 1 tensor, unless disable_group_fusion is requested.
  // In that case, even single tensor groups are created to enforce disabling fusion.
  if (tensors.size() > 1 || horovod_global.disable_group_fusion) {
    auto group_id = horovod_global.group_table.RegisterGroup(std::move(names));
    for (auto& message : messages) {
      message.set_group_id(group_id);
    }
  }

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  status = horovod_global.tensor_queue.AddToTensorQueueMulti(entries, messages);

  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              ReadyEventList ready_event_list,
                              const std::string& name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event_list = ready_event_list;
  e.device = device;
  e.callback = callback;
  e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodAllgather, e.tensor->size());

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              ReadyEventList ready_event_list,
                              const std::string& name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(Request::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event_list = ready_event_list;
  e.device = device;
  e.callback = callback;
  e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodBroadcast, e.tensor->size());

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAlltoall(std::shared_ptr<OpContext> context,
                             std::shared_ptr<Tensor> tensor,
                             std::shared_ptr<Tensor> splits,
                             ReadyEventList ready_event_list,
                             const std::string& name, const int device,
                             StatusCallback callback) {
  // Check arguments
  if (splits->shape().dims() > 1) {
    return Status::InvalidArgument("alltoall expects a 1D splits tensor");
  }
  if (splits->dtype() != HOROVOD_INT32) {
    return Status::InvalidArgument("alltoall expects splits to contain 32-bit integer elements.");
  }

  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLTOALL);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event_list = ready_event_list;
  e.device = device;
  e.callback = callback;
  e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodAlltoall, e.tensor->size());

  int64_t splits_first_dim = splits->shape().dim_size(0);
  int64_t tensor_first_dim = tensor->shape().dim_size(0);
  int world_size = horovod_global.controller->GetSize();
  if (splits_first_dim == world_size) {
    auto splits_data = static_cast<const int32_t*>(splits->data());
    auto sum = std::accumulate(splits_data, splits_data + splits_first_dim, 0);
    if (sum > tensor_first_dim) {
      return Status::InvalidArgument("Sum of splits entries is greater than the first dimension of tensor.");
    }
    e.splits.assign(splits_data,
                    splits_data + splits->shape().num_elements());
  } else if (splits_first_dim == 0) {
    if (tensor_first_dim % world_size != 0) {
      return Status::InvalidArgument("splits not provided, but first dimension of tensor is not an even "
                                     "multiple of the number of workers.");
    }
    e.splits.resize(world_size, tensor_first_dim / world_size);
  } else {
      return Status::InvalidArgument("Number of entries in splits does not equal number of workers.");
  }

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueJoin(std::shared_ptr<OpContext> context,
                   ReadyEventList ready_event_list,
                   const std::string& name, const int device,
                   StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
  message.set_device(device);
  message.set_request_type(Request::JOIN);

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.ready_event_list = ready_event_list;
  e.device = device;
  e.callback = callback;

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

} // namespace common
} // namespace horovod
