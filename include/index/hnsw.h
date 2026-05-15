#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace amio::index {

/// 多层 HNSW（最多 8 层，与 `NodeBlock` 中邻居表对齐）。
class HnswIndex {
public:
  /// `m`：高层邻居上限；Layer0 使用 `2*m`。
  HnswIndex(size_t dim, size_t m, size_t ef_construction, uint64_t seed);

  size_t dim() const { return dim_; }
  size_t size() const { return vectors_.size(); }
  uint64_t entry_point() const { return entry_point_; }
  int max_layer() const { return max_layer_; }

  /// 插入：`id` 必须等于当前 `size()`（连续 ID）。
  void insert(uint64_t id, std::vector<float> vec);

  /// 近似检索 Top-K。
  std::vector<std::pair<uint64_t, float>> search(const std::vector<float> &query,
                                                 size_t k, size_t ef) const;

  /// 节点 `id` 的最高层编号（0..7）。
  int node_max_layer(uint64_t id) const;

  /// 第 `layer` 层的邻居列表（只读）。
  const std::vector<uint64_t> &neighbors_at(int layer, uint64_t id) const;

private:
  static constexpr int kMaxLayers = 8;

  size_t dim_ = 0;
  size_t m_ = 0;
  size_t ef_construction_ = 0;

  int max_layer_ = -1;
  uint64_t entry_point_ = 0;

  std::vector<std::vector<float>> vectors_;
  std::vector<int> node_level_; // 每个节点的最高层
  /// `layer_neighbors_[l][id]`：第 l 层上节点 id 的邻居。
  std::vector<std::vector<std::vector<uint64_t>>> layer_neighbors_;

  mutable std::mt19937_64 rng_;

  static float l2_sq(const std::vector<float> &a, const std::vector<float> &b);
  int random_level();
  size_t max_neighbors_for_layer(int layer) const;

  void ensure_layer_sizes(int layer, uint64_t node_id);
  std::vector<std::pair<uint64_t, float>> search_layer(uint64_t ep, const std::vector<float> &q,
                                                       size_t ef, int layer) const;
  static uint64_t pick_closest(const std::vector<std::pair<uint64_t, float>> &w);
  std::vector<uint64_t> select_neighbors(const std::vector<std::pair<uint64_t, float>> &w,
                                         int layer) const;
  void add_edge(int layer, uint64_t a, uint64_t b);
};

} // namespace amio::index
