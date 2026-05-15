#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace amio::util {

template <class T, class ScoreT> struct Scored {
  T id{};
  ScoreT score{};
};

template <class T, class ScoreT>
inline void topk_inplace(std::vector<Scored<T, ScoreT>> &v, size_t k) {
  if (k >= v.size())
    return;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end(),
                   [](auto &a, auto &b) { return a.score < b.score; });
  v.resize(k);
  std::sort(v.begin(), v.end(),
            [](auto &a, auto &b) { return a.score < b.score; });
}

} // namespace amio::util
