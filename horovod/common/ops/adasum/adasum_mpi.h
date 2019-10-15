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

#ifndef HOROVOD_ADASUM_MPI_H
#define HOROVOD_ADASUM_MPI_H

#include "mpi.h"

#include "../../mpi/mpi_context.h"
#include "adasum.h"


namespace horovod {
namespace common {

class AdasumMPI : public Adasum<MPI_Comm> {
public:
  AdasumMPI(MPIContext* mpi_context, HorovodGlobalState* global_state);
  
  ~AdasumMPI();
  
protected:
  void PointToPointSend(void* input_data_buffer,
                        int64_t buffer_length,
                        DataType horovod_datatype,
                        int dest_rank,
                        int tag,
                        MPI_Comm communicator) override;

  void PointToPointRecv(void* output_data_buffer,
                        int64_t buffer_length,
                        DataType horovod_datatype,
                        int src_rank,
                        int tag,
                        MPI_Comm communicator) override;

  void PointToPointSendRecv(void* input_data_buffer,
                            int64_t input_buffer_length,
                            DataType input_horovod_datatype,
                            int dst_rank,
                            int send_tag,
                            void* output_data_buffer,
                            int64_t output_buffer_length,
                            DataType output_horovod_datatype,
                            int src_rank,
                            int recv_tag,
                            MPI_Comm communicator) override;

  int GetLocalRankWithComm(MPI_Comm local_comm) override;

  int GetSizeWithComm(MPI_Comm comm) override;

  void SumAllreduceWithComm(std::vector<TensorTableEntry>& entries,
                            void* data,
                            int num_elements,
                            DataType horovod_datatype,
                            MPI_Comm comm,
                            HorovodGlobalState *global_state) override;

  MPIContext* mpi_context_;
  // MPI communicators used to do adasum
  MPI_Comm* reduction_comms_ = nullptr;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_ADASUM_MPI_H
