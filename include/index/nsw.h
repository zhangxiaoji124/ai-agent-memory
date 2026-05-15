#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace amio::index {

/// 单层 NSW（可逐步升级为多层 HNSW）。
class NswIndex {
public:
  NswIndex(size_t dim, size_t m, size_t ef_construction, uint64_t seed);

  size_t dim() const { return dim_; }
  size_t size() const { return vectors_.size(); }
  bool empty() const { return vectors_.empty(); }
  uint64_t entry_point() const { return entry_point_; }

  /// 插入：`id` 必须等于当前 size()（连续 ID）。
  void insert(uint64_t id, std::vector<float> vec);

  /// 近似检索 Top-K（best-first on graph）。
  std::vector<std::pair<uint64_t, float>> search(const std::vector<float> &query,
                                                 size_t k, size_t ef) const;

  /// 导出邻接表（Layer0）。
  const std::vector<std::vector<uint64_t>> &graph() const { return graph_; }
  const std::vector<std::vector<float>> &vectors() const { return vectors_; }

private:
  size_t dim_ = 0;
  size_t m_ = 0;
  size_t ef_construction_ = 0;

  uint64_t entry_point_ = 0;
  bool has_entry_ = false;

  std::vector<std::vector<float>> vectors_;
  std::vector<std::vector<uint64_t>> graph_;
  mutable std::mt19937_64 rng_;

  static float l2_sq(const std::vector<float> &a, const std::vector<float> &b);
  std::vector<std::pair<uint64_t, float>> search_layer(uint64_t ep,
                                                       const std::vector<float> &q,
                                                       size_t ef) const;
  void add_edge(uint64_t a, uint64_t b);
};

} // namespace amio::index
