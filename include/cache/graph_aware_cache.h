#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "index/storage_layout.h"
#include "policy/agent_io_policy.h"

namespace amio {
struct IoMetrics;
}

namespace amio::prefetch {
class IoUringBackend;
}

namespace amio::cache {

/// 图感知缓存（骨架）：pinned 常驻 + hot LRU（后续可替换 LFU）。
class GraphAwareCache {
public:
  /// @param dynamic_capacity_bytes  Dynamic/hot LRU 池上限
  /// @param pinned_capacity_bytes   Static 常驻 pin 池上限（仅淘汰 hot，不淘汰 pinned）
  explicit GraphAwareCache(size_t dynamic_capacity_bytes,
                           size_t pinned_capacity_bytes = 0,
                           const policy::AgentIoPolicy *policy = nullptr,
                           ::amio::IoMetrics *io_metrics = nullptr);

  // 不可拷贝
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

  uint64_t hits_total() const {
    return hits_.load(std::memory_order_relaxed);
  }
  uint64_t misses_total() const {
    return misses_.load(std::memory_order_relaxed);
  }

  uint64_t pinned_bytes_used() const;
  uint64_t static_pins_count() const { return static_pins_count_; }

  /// 将命中/未命中计数清零（保留缓存内容），用于按查询粒度统计。
  void reset_access_counters();

private:
  size_t dynamic_capacity_bytes_ = 0;
  size_t pinned_capacity_bytes_ = 0;
  size_t hot_used_bytes_ = 0;
  size_t pinned_used_bytes_ = 0;
  uint64_t static_pins_count_ = 0;

  mutable std::shared_mutex mu_;
  std::unordered_map<uint64_t, amio::index::NodeBlock> pinned_;

  // 简易 LRU：使用 `tick` + 最小 tick 淘汰（骨架实现，O(n)）。
  struct HotEntry {
    amio::index::NodeBlock block{};
    uint64_t tick = 0;
  };
  std::unordered_map<uint64_t, HotEntry> hot_;
  uint64_t tick_ = 0;

  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};

  void evict_if_needed_locked();
  void insert_hot_locked(uint64_t node_id, const amio::index::NodeBlock &b);

  const policy::AgentIoPolicy *policy_ = nullptr;
  ::amio::IoMetrics *io_metrics_ = nullptr;
  std::unordered_map<uint64_t, uint32_t> cold_misses_;
};

} // namespace amio::cache