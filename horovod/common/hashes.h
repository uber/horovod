// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef HOROVOD_HASHES_H
#define HOROVOD_HASHES_H

#include <functional>

#include "common.h"

// Golden ratio from http://burtleburtle.net/bob/hash/doobs.html
#define GOLDEN_RATIO 0x9e3779b9

namespace std {

namespace {

template <typename T>
inline std::size_t hash_one(const T& element, std::size_t seed) {
  return seed ^
         (std::hash<T>()(element) + GOLDEN_RATIO + (seed << 6) + (seed >> 2));
}

} // namespace

template <typename T> struct hash<std::vector<T>> {
  using argument_type = std::vector<T>;
  using result_type = std::size_t;

  result_type operator()(argument_type const& in) const {
    size_t size = in.size();
    result_type seed = 0;
    for (size_t i = 0; i < size; i++)
      seed = hash_one<T>(in[i], seed);
    return seed;
  }
};

template <typename U, typename V> struct hash<std::tuple<U, V>> {
  using argument_type = std::tuple<U, V>;
  using result_type = std::size_t;

  result_type operator()(argument_type const& in) const {
    result_type seed = 0;
    seed = hash_one<U>(std::get<0>(in), seed);
    seed = hash_one<V>(std::get<1>(in), seed);
    return seed;
  }
};

template <typename U, typename V, typename W> struct hash<std::tuple<U, V, W>> {
  using argument_type = std::tuple<U, V, W>;
  using result_type = std::size_t;

  result_type operator()(argument_type const& in) const {
    result_type seed = 0;
    seed = hash_one<U>(std::get<0>(in), seed);
    seed = hash_one<V>(std::get<1>(in), seed);
    seed = hash_one<W>(std::get<2>(in), seed);
    return seed;
  }
};

template <typename U, typename V> struct hash<std::pair<U, V>> {
  using argument_type = std::tuple<U, V>;
  using result_type = std::size_t;

  result_type operator()(argument_type const& in) const {
    result_type seed = 0;
    seed = hash_one<U>(std::get<0>(in), seed);
    seed = hash_one<V>(std::get<1>(in), seed);
    return seed;
  }
};

template <> struct hash<horovod::common::Framework> {
  std::size_t operator()(horovod::common::Framework const& in) const {
    return (std::size_t)in;
  }
};

} // namespace std

#endif // HOROVOD_HASHES_H
