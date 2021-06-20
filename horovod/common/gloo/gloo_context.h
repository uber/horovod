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

#ifndef HOROVOD_GLOO_CONTEXT_H
#define HOROVOD_GLOO_CONTEXT_H

#include "gloo/context.h"

#include "../common.h"
#include "../logging.h"

#if HAVE_MPI
#include "../mpi/mpi_context.h"
#endif

// Horovod Gloo rendezvous knobs.
#define HOROVOD_GLOO_TIMEOUT_SECONDS "HOROVOD_GLOO_TIMEOUT_SECONDS"
#define HOROVOD_GLOO_RENDEZVOUS_ADDR "HOROVOD_GLOO_RENDEZVOUS_ADDR"
#define HOROVOD_GLOO_RENDEZVOUS_PORT "HOROVOD_GLOO_RENDEZVOUS_PORT"
#define HOROVOD_GLOO_GLOBAL_PREFIX "global"
#define HOROVOD_GLOO_LOCAL_PREFIX "local_"
#define HOROVOD_GLOO_CROSS_PREFIX "cross_"
#define HOROVOD_GLOO_GET_RANK_AND_SIZE "rank_and_size"
#define HOROVOD_HOSTNAME "HOROVOD_HOSTNAME"
#define HOROVOD_RANK "HOROVOD_RANK"
#define HOROVOD_SIZE "HOROVOD_SIZE"
#define HOROVOD_LOCAL_RANK "HOROVOD_LOCAL_RANK"
#define HOROVOD_LOCAL_SIZE "HOROVOD_LOCAL_SIZE"
#define HOROVOD_CROSS_RANK "HOROVOD_CROSS_RANK"
#define HOROVOD_CROSS_SIZE "HOROVOD_CROSS_SIZE"

namespace horovod {
namespace common {

struct GlooContext {

#if HAVE_MPI
  void InitializeFromMPI(MPIContext& mpi_ctx, const std::string& gloo_iface);
#endif

  void Initialize(const std::string& gloo_iface);

  void Finalize();

  std::shared_ptr<gloo::Context> GetGlooContext(Communicator communicator);

  void Enable() {
    enabled_ = true;
    LOG(DEBUG) << "Gloo context enabled.";
  }

  bool IsEnabled() { return enabled_; }

  std::shared_ptr<gloo::Context> ctx = nullptr; // Global context
  std::shared_ptr<gloo::Context> cross_ctx = nullptr;
  std::shared_ptr<gloo::Context> local_ctx = nullptr;

private:
  // Flag indicating whether gloo is enabled.
  bool enabled_ = false;
  bool reset_ = false;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_GLOO_CONTEXT_H
