#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/storage_layout.h"
#include "policy/agent_io_policy.h"
#include "cache/isvm_scorer.h"

namespace amio {
struct IoMetrics;
}

namespace amio::prefetch {
class IoUringBackend;
}

namespace amio::cache {

/// 图感知缓存：pinned 常驻 + hot LRU + PAIC 预取有效性追踪（简化版）。
///
/// PAIC 追踪：
///  - TopologyPrefetcher 提交预取时调用 record_prefetch_submitted()，
///    将这批 id 写入 prefetch_pending_ 集合。
///  - get_or_load() 缓存未命中从磁盘加载时，若 id 在 prefetch_pending_ 中
///    则标记 HotEntry.is_prefetch_loaded=true（预取与需求协同）。
///  - get() 命中已标记条目时，计入 io_metrics_->useful_prefetch_demand_hits
///    并清除标记（需求已消费该预取）。
///  - evict_if_needed_locked() 淘汰仍标记条目时，计入
///    io_metrics_->wasted_prefetch_evictions（条目在需求到来前被逐出）。
///  - KV Group：以 layer0 邻居子集为粒度缓存元数据，配合 ISVM 做 size-aware 驱逐。
class GraphAwareCache {
public:
  /// @param dynamic_capacity_bytes  Dynamic/hot LRU 池上限
  /// @param pinned_capacity_bytes   Static 常驻 pin 池上限（仅淘汰 hot，不淘汰 pinned）
  explicit GraphAwareCache(size_t dynamic_capacity_bytes,
                           size_t pinned_capacity_bytes = 0,
                           const policy::AgentIoPolicy *policy = nullptr,
                           ::amio::IoMetrics *io_metrics = nullptr);

  GraphAwareCache(const GraphAwareCache &) = delete;
  GraphAwareCache &operator=(const GraphAwareCache &) = delete;

  bool contains(uint64_t node_id) const;
  bool is_pinned(uint64_t node_id) const;

  /// 缓存命中时返回 true 并输出 block；否则返回 false。
  bool get(uint64_t node_id, amio::index::NodeBlock *out);

  /// 命中则返回；未命中则从磁盘读取并回填热点区。
  bool get_or_load(amio::index::IndexFile *file, uint64_t node_id,
                   amio::index::NodeBlock *out);

  /// 若 `uring` 可用且 `ok()`，优先走 io_uring 同步读；否则 `pread`。
  bool get_or_load(amio::index::IndexFile *file, prefetch::IoUringBackend *uring,
                   uint64_t node_id, amio::index::NodeBlock *out);

  void insert_hot(uint64_t node_id, const amio::index::NodeBlock &b);
  void pin(uint64_t node_id, const amio::index::NodeBlock &b);

  double hit_rate() const;

  uint64_t hits_total() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t misses_total() const { return misses_.load(std::memory_order_relaxed); }

  uint64_t pinned_bytes_used() const;
  uint64_t static_pins_count() const { return static_pins_count_; }

  /// 将命中/未命中计数清零（保留缓存内容），用于按查询粒度统计。
  void reset_access_counters();

  /// PAIC：记录预取提交的节点 id，供后续需求命中时统计有效性。
  /// 集合上限 kMaxPrefetchPending，超出时清空以防无界增长。
  void record_prefetch_submitted(const std::vector<uint64_t> &ids);

  void set_isvm_kv_cache_enabled(bool enabled);

private:
  static constexpr size_t kMaxPrefetchPending = 32768;

  size_t dynamic_capacity_bytes_ = 0;
  size_t pinned_capacity_bytes_ = 0;
  size_t hot_used_bytes_ = 0;
  size_t pinned_used_bytes_ = 0;
  uint64_t static_pins_count_ = 0;

  mutable std::shared_mutex mu_;
  std::unordered_map<uint64_t, amio::index::NodeBlock> pinned_;

  struct HotEntry {
    amio::index::NodeBlock block{};
    uint64_t tick = 0;
    bool is_prefetch_loaded = false; // PAIC: 该条目由预取协同触发加载
    uint32_t kv_group_bytes = 0;     // GroupCache: layer0 邻居组字节估算
    int64_t ev_key = 0;              // 在 ev_index_ 中的稳定驱逐键（16*C - tick）
  };
  std::unordered_map<uint64_t, HotEntry> hot_;
  // 驱逐索引：(ev_key, node_id) 升序；ev_key 最大者最该被驱逐（rbegin）。
  // 把 O(n) 全表扫描换成 O(log n)，同时保持 ISVM 评分的 argmax 顺序不变。
  std::set<std::pair<int64_t, uint64_t>> ev_index_;
  uint64_t tick_ = 0;

  /// 计算与时间无关的 ISVM 分量 C，并组合成稳定驱逐键 16*C - tick。
  /// 驱逐时 tick_ 对所有条目相同，故 argmax(score) ≡ argmax(16*C - tick)。
  int64_t ev_key_for(bool is_prefetch_loaded, uint32_t kv_group_bytes, uint64_t tick) const;

  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};

  // PAIC 预取 pending 集合，独立锁以减少与主缓存锁的竞争
  std::mutex prefetch_pending_mu_;
  std::unordered_set<uint64_t> prefetch_pending_;

  void evict_if_needed_locked();
  void insert_hot_locked(uint64_t node_id, const amio::index::NodeBlock &b,
                         bool is_prefetch_loaded = false);
  bool consume_prefetch_pending(uint64_t id);

  const policy::AgentIoPolicy *policy_ = nullptr;
  ::amio::IoMetrics *io_metrics_ = nullptr;
  std::unordered_map<uint64_t, uint32_t> cold_misses_;
  IsvmScorer isvm_{};
};

} // namespace amio::cache
