#pragma once

#include <atomic>
#include <cstdint>

namespace amio {

/// 磁盘检索路径上的 I/O 与预取观测（线程安全计数，供评测日志使用）。
struct IoMetrics {
  std::atomic<uint64_t> disk_sync_block_reads{0};
  std::atomic<uint64_t> disk_sync_via_uring{0};
  std::atomic<uint64_t> disk_sync_via_pread{0};
  std::atomic<uint64_t> prefetch_blocks_submitted{0};
  // 单次查询中 visited_count 越过 θ·ef_search 阈值触发阶段切换的次数（每查询通常 ≤1）
  std::atomic<uint64_t> theta_phase_switches{0};

  // PAIC 简化版：预取有效性可量化指标
  // useful_prefetch_demand_hits: 预取加载的缓存条目被后续需求访问命中的次数（有效预取）
  // wasted_prefetch_evictions:   预取加载的缓存条目在被需求访问前被驱逐的次数（无效预取）
  std::atomic<uint64_t> useful_prefetch_demand_hits{0};
  std::atomic<uint64_t> wasted_prefetch_evictions{0};

  void reset() {
    disk_sync_block_reads.store(0, std::memory_order_relaxed);
    disk_sync_via_uring.store(0, std::memory_order_relaxed);
    disk_sync_via_pread.store(0, std::memory_order_relaxed);
    prefetch_blocks_submitted.store(0, std::memory_order_relaxed);
    theta_phase_switches.store(0, std::memory_order_relaxed);
    useful_prefetch_demand_hits.store(0, std::memory_order_relaxed);
    wasted_prefetch_evictions.store(0, std::memory_order_relaxed);
  }
};

struct IoMetricsSnapshot {
  uint64_t disk_sync_block_reads = 0;
  uint64_t disk_sync_read_bytes = 0;
  uint64_t disk_sync_via_uring = 0;
  uint64_t disk_sync_via_pread = 0;
  uint64_t prefetch_blocks_submitted = 0;
  uint64_t theta_phase_switches = 0;
  uint64_t useful_prefetch_demand_hits = 0;
  uint64_t wasted_prefetch_evictions = 0;
};

inline IoMetricsSnapshot snapshot(const IoMetrics &m, uint64_t block_bytes) {
  const uint64_t blocks = m.disk_sync_block_reads.load(std::memory_order_relaxed);
  return IoMetricsSnapshot{
      blocks,
      blocks * block_bytes,
      m.disk_sync_via_uring.load(std::memory_order_relaxed),
      m.disk_sync_via_pread.load(std::memory_order_relaxed),
      m.prefetch_blocks_submitted.load(std::memory_order_relaxed),
      m.theta_phase_switches.load(std::memory_order_relaxed),
      m.useful_prefetch_demand_hits.load(std::memory_order_relaxed),
      m.wasted_prefetch_evictions.load(std::memory_order_relaxed),
  };
}

} // namespace amio
