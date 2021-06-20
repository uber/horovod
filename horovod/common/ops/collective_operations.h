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

#ifndef HOROVOD_COLLECTIVE_OPERATIONS_H
#define HOROVOD_COLLECTIVE_OPERATIONS_H

#include <iostream>

#include "../common.h"
#include "../controller.h"
#include "../global_state.h"
#include "../half.h"
#include "../operations.h"
#include "../parameter_manager.h"

#if __AVX__ && __F16C__
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace horovod {
namespace common {

class HorovodOp {
public:
  HorovodOp(HorovodGlobalState* global_state);

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

protected:
  int64_t NumElements(std::vector<TensorTableEntry>& entries);

  virtual void WaitForData(std::vector<TensorTableEntry>& entries);

  HorovodGlobalState* global_state_;
};

class AllreduceOp : public HorovodOp {
public:
  AllreduceOp(HorovodGlobalState* global_state);

  virtual ~AllreduceOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

  virtual bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const = 0;

protected:
  virtual void
  MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                       const void*& fused_input_data, void*& buffer_data,
                       size_t& buffer_len);

  virtual void MemcpyOutFusionBuffer(const void* buffer_data,
                                     std::vector<TensorTableEntry>& entries);

  virtual void
  MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                            const TensorTableEntry& e,
                            void* buffer_data_at_offset);

  virtual void
  MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                             const void* buffer_data_at_offset,
                             TensorTableEntry& e);

  virtual void
  ScaleBuffer(double scale_factor, const std::vector<TensorTableEntry>& entries,
              const void* fused_input_data, void* buffer_data, int64_t num_elements);

};

template <typename T, typename TS>
void ScaleBufferCPUImpl(const T* input, T* output, int64_t num_elements, TS scale_factor) {
  for (int64_t i = 0; i < num_elements; ++i) {
    output[i] = scale_factor * input[i];
  }
}

// Specialization for float16
template <> inline
void ScaleBufferCPUImpl(const unsigned short* input, unsigned short* output, int64_t num_elements, float scale_factor) {
  int64_t i = 0;

#if __AVX__ && __F16C__
  if (is_avx_and_f16c()) {
    __m256 scale_factor_m256 = _mm256_broadcast_ss(&scale_factor);
    for (; i < (num_elements / 8) * 8; i += 8) {
      // convert input to m256
      __m256 input_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(input + i)));

      // scale and store result in output_m256
      __m256 output_m256 = _mm256_mul_ps(input_m256, scale_factor_m256);

      // convert back and store in output
      __m128i output_m128i = _mm256_cvtps_ph(output_m256, 0);
      _mm_storeu_si128((__m128i*)(output + i), output_m128i);
    }
  }
#endif

  for (; i < num_elements; ++i) {
    float in_float;
    HalfBits2Float(input + i, &in_float);
    float out_float = scale_factor * in_float;
    Float2HalfBits(&out_float, output + i);
  }

}

class AllgatherOp : public HorovodOp {
public:
  AllgatherOp(HorovodGlobalState* global_state);

  virtual ~AllgatherOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

  virtual bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const = 0;

protected:
  virtual Status AllocateOutput(std::vector<TensorTableEntry>& entries,
                                const Response& response,
                                int64_t**& entry_component_sizes,
                                int*& recvcounts);

  virtual void SetDisplacements(const int* recvcounts, int*& displcmnts);

  virtual void
  SetEntryComponentOffsets(const std::vector<TensorTableEntry>& entries,
                           const int64_t* const* entry_component_sizes,
                           const int* recvcounts,
                           int64_t**& entry_component_offsets);

  virtual void
  MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                       const int* displcmnts, int element_size,
                       void*& buffer_data);

  virtual void
  MemcpyOutFusionBuffer(const int64_t* const* entry_component_offsets,
                        const int64_t* const* entry_component_sizes,
                        const void* buffer_data, int element_size,
                        std::vector<TensorTableEntry>& entries);

  virtual void
  MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                            const TensorTableEntry& e,
                            void* buffer_data_at_offset);

  virtual void
  MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                             const void* buffer_data_at_offset,
                             TensorTableEntry& e,
                             int64_t entry_offset,
                             size_t entry_size);
};

class BroadcastOp : public HorovodOp {
public:
  BroadcastOp(HorovodGlobalState* global_state);

  virtual ~BroadcastOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

  virtual bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const = 0;
};

class AlltoallOp : public HorovodOp {
public:
  AlltoallOp(HorovodGlobalState* global_state);

  virtual ~AlltoallOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

  virtual bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const = 0;

protected:
  template <typename T>
  Status PrepareOutputAndParams(TensorTableEntry& e,
                                std::vector<T>& sdispls,
                                std::vector<T>& rdispls,
                                std::vector<T>& sendcounts,
                                std::vector<T>& recvcounts) {
    auto world_size = global_state_->controller->GetSize();

    const auto& splits = e.splits;
    std::vector<int32_t> recvsplits;
    // Perform alltoall of splits to get expected receive splits
    global_state_->controller->AlltoallGetRecvSplits(splits, recvsplits);

    // Every tensor participating in Alltoall operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }
    int64_t slice_num_elements = slice_shape.num_elements();

    // Prepare send/recvcounts and displacements for Alltoallv
    sdispls.resize(world_size);
    rdispls.resize(world_size);
    sendcounts.resize(world_size);
    recvcounts.resize(world_size);

    size_t output_first_dim = 0;
    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = splits[i] * slice_num_elements;
      recvcounts[i] = recvsplits[i] * slice_num_elements;
      output_first_dim += recvsplits[i];
    }

    for (int i = 1; i < world_size; ++i) {
      sdispls[i] = sdispls[i-1] + sendcounts[i-1];
      rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    }

    // Allocate output
    TensorShape output_shape;
    output_shape.AddDim(output_first_dim);
    output_shape.AppendShape(slice_shape);
    Status status = e.context->AllocateOutput(output_shape, &e.output);
    if (!status.ok()) {
      return status;
    }

    // Allocate and fill received_splits output
    TensorShape received_splits_shape;
    received_splits_shape.AddDim(recvsplits.size());
    Status rstatus = e.context->AllocateOutput(1, received_splits_shape,
                                               &e.received_splits);
    if (!rstatus.ok()) {
      return rstatus;
    }
    auto* target_pointer = reinterpret_cast<int32_t*>(
        const_cast<void*>(e.received_splits->data()));
    std::copy(recvsplits.cbegin(), recvsplits.cend(), target_pointer);

    return Status::OK();
  }
};

class JoinOp : public HorovodOp {
public:
  JoinOp(HorovodGlobalState* global_state);

  virtual ~JoinOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response);
};

class ErrorOp : public HorovodOp {
public:
  ErrorOp(HorovodGlobalState* global_state);

  virtual ~ErrorOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries, const Response& response);
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_COLLECTIVE_OPERATIONS_H
