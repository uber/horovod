// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Modifications copyright (C) 2019 Intel Corporation
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
#include <cassert>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#define OMPI_SKIP_MPICXX

#include "common.h"
#include "fusion_buffer_manager.h"
#include "global_state.h"
#include "half.h"
#include "hashes.h"
#include "logging.h"
#include "message.h"
#include "message_table.h"
#include "mpi.h"
#include "mpi_context.h"
#include "mpi_controller.h"
#include "operations.h"
#include "ops/mpi_operations.h"
#include "ops/operation_manager.h"
#include "parameter_manager.h"
#include "timeline.h"
#include "utils/env_parser.h"

#if HAVE_CUDA
#include "ops/cuda_operations.h"
#include "ops/mpi_cuda_operations.h"
#endif

#if HAVE_NCCL
#include "ops/nccl_operations.h"
#endif

#if HAVE_DDL
#include "ddl_mpi_context_manager.h"
#include "ops/ddl_operations.h"
#endif

#if HAVE_MLSL
#include "ops/mlsl_operations.h"
#endif

#if HAVE_GLOO
#include "ops/gloo_operations.h"
#endif

/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * whichever hardware-optimized communication libraries enabled.
 *
 * The primary logic of the allreduce, allgather and broadcast currently
 * support in MPI, NCCL, CUDA, Gloo, MLSL, DDL. The background thread which
 * facilitates controller operations is run in BackgroundThreadLoop().
 * The provided ops are:
 *      – HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all processes in the global communicator.
 *      – HorovodAllgather:
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

MPIContext mpi_context;

#if HAVE_GLOO
GlooContext gloo_context;
#endif

#if HAVE_CUDA
CUDAContext cuda_context;
#endif

#if HAVE_NCCL
NCCLContext nccl_context;
#endif

#if HAVE_DDL
DDLContext ddl_context;
#endif

#if HAVE_MLSL
MLSLContext mlsl_context;
#endif

std::unique_ptr<OperationManager> op_manager;

const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Horovod has not been initialized; use hvd.init().");

const Status SHUT_DOWN_ERROR = Status::UnknownError(
    "Horovod has been shut down. This was caused by an exception on one of the "
    "ranks or an attempt to allreduce, allgather or broadcast a tensor after "
    "one of the ranks finished execution. If the shutdown was caused by an "
    "exception, you should see the exception in the log before the first "
    "shutdown message.");

const Status DUPLICATE_NAME_ERROR = Status::InvalidArgument(
    "Requested to allreduce, allgather, or broadcast a tensor with the same "
    "name as another tensor that is currently being processed.  If you want "
    "to request another tensor, use a different tensor name.");

OperationManager* CreateOperationManager(HorovodGlobalState& state) {
  // Order of these operations is very important. Operations will be checked
  // sequentially from the first to the last. The first 'Enabled' operation will
  // be executed.
  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops;

#if HAVE_CUDA
#if HOROVOD_GPU_ALLREDUCE == 'M'
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new MPI_CUDAAllreduce(&mpi_context, &cuda_context, &state)));

#else
#if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
  LOG(INFO) << "NCCL enabled.";
  allreduce_ops.push_back(
      std::shared_ptr<AllreduceOp>(new NCCLHierarchicalAllreduce(
          &nccl_context, &mpi_context, &cuda_context, &state)));
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new NCCLAllreduce(&nccl_context, &mpi_context, &cuda_context, &state)));

#elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
  LOG(INFO) << "DDL enabled.";
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new DDLAllreduce(&ddl_context, &cuda_context, &state)));
#endif

  allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
      new MPIHierarchicalAllgather(&mpi_context, &state)));
#endif
#endif

#if HAVE_GLOO
  if (state.cpu_operation == LibType::GLOO) {
    LOG(INFO) << "Gloo enabled.";
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new GlooAllreduce(&gloo_context, &state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new GlooAllgather(&gloo_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new GlooBroadcast(&gloo_context, &state)));
  }
#endif

#if HAVE_MLSL
  if (state.cpu_operation == LibType::MLSL) {
    LOG(INFO) << "MLSL enabled.";
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new MLSLAllreduce(&mlsl_context, &state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new MLSLAllgather(&mlsl_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new MLSLBroadcast(&mlsl_context, &state)));
  }
#endif

  // Default operations, always enabled but last to be checked.
  allreduce_ops.push_back(
      std::shared_ptr<AllreduceOp>(new MPIAllreduce(&mpi_context, &state)));
  allgather_ops.push_back(
      std::shared_ptr<AllgatherOp>(new MPIAllgather(&mpi_context, &state)));
  broadcast_ops.push_back(
      std::shared_ptr<BroadcastOp>(new MPIBroadcast(&mpi_context, &state)));

  std::shared_ptr<ErrorOp> error_op(new ErrorOp(&state));

  return new OperationManager(&state.parameter_manager, allreduce_ops,
                              allgather_ops, broadcast_ops, error_op);
}

// Helper function to get list of allreduced tensor names and total size for
// use with the autotuner.
int64_t GetTensorDataForAutotuner(const ResponseList& response_list,
                                  const TensorTable& tensor_table,
                                  std::vector<std::string>& tensor_names) {
  int64_t total_tensor_size = 0;
  for (auto& response : response_list.responses()) {
    if (response.response_type() == Response::ResponseType::ALLREDUCE) {
      for (auto& tensor_name : response.tensor_names()) {
        tensor_names.push_back(tensor_name);
        LOG(TRACE) << "Looking for tensor with name " << tensor_name;
        auto& entry = tensor_table.at(tensor_name);
        LOG(TRACE) << "Found tensor with name " << tensor_name;
        total_tensor_size += entry.tensor->size();
      }
    }
  }
  return total_tensor_size;
}

// Process a Response by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(TensorTable& tensor_table, Response response) {
  std::vector<TensorTableEntry> entries;
  // Reserve to save re-allocation costs, as we know the size before.
  entries.reserve(response.tensor_names().size());
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(horovod_global.mutex);
    for (auto& name : response.tensor_names()) {
      // We should never fail at finding this key in the tensor table.
      auto iter = tensor_table.find(name);
      assert(iter != tensor_table.end());

      assert(response.response_type() == Response::ALLREDUCE ||
             response.response_type() == Response::ALLGATHER ||
             response.response_type() == Response::BROADCAST ||
             response.response_type() == Response::ERROR);

      entries.push_back(iter->second);

      // Clear the tensor table of this tensor and its callbacks; the rest of
      // this function takes care of it.
      tensor_table.erase(iter);
    }
  }

  auto& timeline = horovod_global.timeline;
  for (auto& e : entries) {
    timeline.Start(e.tensor_name, response.response_type());
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
      for (auto& e : entries) {
        timeline.End(e.tensor_name, nullptr);
        e.callback(status);
      }
      return;
    }
  }

  // On GPU data readiness is signalled by ready_event.
  std::vector<TensorTableEntry> waiting_tensors;
  for (auto& e : entries) {
    if (e.ready_event != nullptr) {
      timeline.ActivityStart(e.tensor_name, WAIT_FOR_DATA);
      waiting_tensors.push_back(e);
    }
  }
  while (!waiting_tensors.empty()) {
    for (auto it = waiting_tensors.begin(); it != waiting_tensors.end();) {
      if (it->ready_event->Ready()) {
        timeline.ActivityEnd(it->tensor_name);
        timeline.ActivityStart(it->tensor_name, WAIT_FOR_OTHER_TENSOR_DATA);
        it = waiting_tensors.erase(it);
      } else {
        ++it;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }
  for (auto& e : entries) {
    if (e.ready_event != nullptr) {
      timeline.ActivityEnd(e.tensor_name);
    }
  }

  Status status;
  try {
    status = op_manager->ExecuteOperation(entries, response);
  } catch (const std::exception& ex) {
    status = Status::UnknownError(ex.what());
  }

  if (!status.in_progress()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, status.ok() ? e.output : nullptr);
      e.callback(status);
    }
  }
}

// The background thread loop coordinates all the controller processes and the
// tensor reductions. The design of the communicator mechanism is limited by a
// few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a
//      single thread. Since TensorFlow may use several threads for graph
//      processing, this means we must have our own dedicated thread for dealing
//      with MPI.
//      2. We want to gracefully handle errors, when all processes do not
//      properly agree upon what should happen (such as mismatched types or
//      shapes). To do so requires the every process to know about the shapes
//      and types of the relevant tensors on the other processes.
//      3. The reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never make
//      progress if we have a thread pool limit.
bool RunLoopOnce(HorovodGlobalState& state, bool is_coordinator);

void BackgroundThreadLoop(HorovodGlobalState& state) {
  state.cpu_operation = ParseCPUOpsFromEnv();

  // Initialize mlsl context
#if HAVE_MLSL
  if (state.cpu_operation == LibType::MLSL) {
    mlsl_context.Init();
  }
#endif

  // Initialize mpi context
#if HAVE_DDL
  // If DDL is enabled, let DDL ops manage MPI environment.
  auto mpi_ctx_manager = DDL_MPIContextManager(ddl_context, cuda_context);
#else
  // Otherwise, let MPI manager be in charge.
  auto mpi_ctx_manager = MPIContextManager();
#endif
  mpi_context.Initialize(state.controller->GetRanks(), mpi_ctx_manager);

  // Initialize controller
  state.controller->Initialize();

  bool is_coordinator = state.controller->IsCoordinator();
  bool is_homogeneous = state.controller->IsHomogeneous();
  int size = state.controller->GetSize();
  int local_size = state.controller->GetLocalSize();

#if HAVE_MLSL
  mlsl_context.Setup(size);
#endif

#if HAVE_CUDA
  // Set number of CUDA streams to use
  auto horovod_num_nccl_streams = std::getenv(HOROVOD_NUM_NCCL_STREAMS);
  if (horovod_num_nccl_streams != nullptr &&
      std::stol(horovod_num_nccl_streams, nullptr, 10) > 0) {
    state.num_nccl_streams = std::atoi(horovod_num_nccl_streams);
  }

#if HAVE_NCCL
  nccl_context.nccl_comms.resize(state.num_nccl_streams);
#endif
  cuda_context.streams.resize(state.num_nccl_streams);
#endif

#if HAVE_GLOO
  if (state.cpu_operation == LibType::GLOO) {
    gloo_context.InitializeFromMPI(mpi_context.mpi_comm, ParseGlooIface());
  }
#endif

  // Open the timeline file on coordinator.
  auto horovod_timeline = std::getenv(HOROVOD_TIMELINE);
  if (is_coordinator && horovod_timeline != nullptr) {
    state.timeline.Initialize(std::string(horovod_timeline),
                              static_cast<unsigned int>(size));
  }
  if (horovod_timeline != nullptr) {
    state.timeline_enabled = true;
  }

  ParseStallInspectorFromEnv(state.controller->GetStallInspector());

  SetBoolFromEnv(HOROVOD_TIMELINE_MARK_CYCLES, state.mark_cycles_in_timeline,
                 true);

  // Override Tensor Fusion threshold, if it's set.
  state.parameter_manager.SetTensorFusionThresholdBytes(64 * 1024 * 1024);
  auto horovod_fusion_threshold = std::getenv(HOROVOD_FUSION_THRESHOLD);
  if (horovod_fusion_threshold != nullptr) {
    int64_t threshold = std::strtol(horovod_fusion_threshold, nullptr, 10);
    state.parameter_manager.SetTensorFusionThresholdBytes(threshold, true);
  }

  // Override the cycle time.
  state.parameter_manager.SetCycleTimeMs(5);
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

  // Initialize the tensor count table. No tensors are available yet.
  if (is_coordinator) {
    state.message_table = std::shared_ptr<MessageTable>(new MessageTable());
  }

  op_manager.reset(CreateOperationManager(state));

  // Signal that initialization is completed.
  state.initialization_done = true;
  LOG(INFO, horovod_global.controller->GetRank()) << "Horovod Initialized";

  // Iterate until shutdown.
  while (RunLoopOnce(state, is_coordinator))
    ;

    // Finalize all contexts
#if HAVE_NCCL
  nccl_context.ShutDown();
#endif

#if HAVE_GLOO
  if (state.cpu_operation == LibType::GLOO) {
    gloo_context.Finalize();
  }
#endif

  LOG(DEBUG, state.controller->GetRank()) << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

  // Notify all outstanding operations that Horovod has been shut down
  // and clear up the tensor table and message queue.
  std::vector<StatusCallback> callbacks;
  {
    std::lock_guard<std::mutex> guard(state.mutex);
    for (auto& e : state.tensor_table) {
      callbacks.emplace_back(e.second.callback);
    }
    state.tensor_table.clear();
    while (!state.message_queue.empty()) {
      state.message_queue.pop();
    }
  }
  for (auto& cb : callbacks) {
    cb(SHUT_DOWN_ERROR);
  }

  mpi_context.Finalize(mpi_ctx_manager);

#if HAVE_MLSL
  if (state.cpu_operation == LibType::MLSL) {
    mlsl_context.Finalize();
  }
#endif
}

bool RunLoopOnce(HorovodGlobalState& state, bool is_coordinator) {
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

  auto response_list = state.controller->ComputeResponseList();

  // Get tensor name and size data for autotuning.
  int64_t total_tensor_size = 0;
  std::vector<std::string> tensor_names;
  if (state.parameter_manager.IsAutoTuning()) {
    std::lock_guard<std::mutex> guard(state.mutex);
    total_tensor_size = GetTensorDataForAutotuner(
        response_list, state.tensor_table, tensor_names);
  }

  // Perform the collective operation. All nodes should end up performing
  // the same operation.
  int rank = state.controller->GetRank();
  for (auto& response : response_list.responses()) {
    LOG(TRACE, rank) << "Performing " << response.tensor_names_string();
    LOG(DEBUG, rank) << "Processing " << response.tensor_names().size()
                     << " tensors";
    PerformOperation(state.tensor_table, response);
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
    horovod_global.controller = std::shared_ptr<Controller>(
        new MPIController(horovod_global, mpi_context));
    horovod_global.controller->SetRank(ranks, nranks);

    // Reset initialization flag
    horovod_global.initialization_done = false;
    horovod_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(horovod_global));
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

void horovod_init_comm(MPI_Comm comm) {
  MPI_Comm_dup(comm, &mpi_context.mpi_comm);
  InitializeHorovodOnce(nullptr, 0);
}

void horovod_shutdown() {
  if (horovod_global.background_thread.joinable()) {
    horovod_global.shut_down = true;
    horovod_global.background_thread.join();
    // Reset the initialization flag to allow restarting with horovod_init(...)
    horovod_global.initialize_flag.clear();
    horovod_global.shut_down = false;
  }
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

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  auto mpiController =
      std::dynamic_pointer_cast<MPIController>(horovod_global.controller);
  return mpiController->IsMpiThreadsSupported() ? 1 : 0;
}
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLREDUCE);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  if (horovod_global.tensor_table.find(name) !=
      horovod_global.tensor_table.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
  LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  return Status::OK();
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
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
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  if (horovod_global.tensor_table.find(name) !=
      horovod_global.tensor_table.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
  LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  return Status::OK();
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
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
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  if (horovod_global.tensor_table.find(name) !=
      horovod_global.tensor_table.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
  LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  return Status::OK();
}

} // namespace common
} // namespace horovod
