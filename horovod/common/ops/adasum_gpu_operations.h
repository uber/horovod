// Copyright 2019 Microsoft. All Rights Reserved.
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

#ifndef HOROVOD_ADASUM_GPU_OPERATIONS_H
#define HOROVOD_ADASUM_GPU_OPERATIONS_H

#include "adasum/adasum_mpi.h"
#include "nccl_operations.h"
#include <array>

namespace horovod {
namespace common {

class AdasumGpuAllreduceOp : public AdasumMPI, public NCCLAllreduce {
public:
  AdasumGpuAllreduceOp(MPIContext* mpi_context, NCCLContext* nccl_context,
                       GPUContext* gpu_context,
                       HorovodGlobalState* global_state);

  ~AdasumGpuAllreduceOp();

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  Status NcclHierarchical(std::vector<TensorTableEntry>& entries,
                          const Response& response);

  // Get host buffer
  uint8_t* GetHostBuffer(uint64_t buffer_length);

private:
  uint64_t current_host_buffer_length;
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_ADASUM_GPU_OPERATIONS_H
