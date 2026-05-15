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

  void reset() {
    disk_sync_block_reads.store(0, std::memory_order_relaxed);
    disk_sync_via_uring.store(0, std::memory_order_relaxed);
    disk_sync_via_pread.store(0, std::memory_order_relaxed);
    prefetch_blocks_submitted.store(0, std::memory_order_relaxed);
  }
};

struct IoMetricsSnapshot {
  uint64_t disk_sync_block_reads = 0;
  uint64_t disk_sync_read_bytes = 0;
  uint64_t disk_sync_via_uring = 0;
  uint64_t disk_sync_via_pread = 0;
  uint64_t prefetch_blocks_submitted = 0;
};

inline IoMetricsSnapshot snapshot(const IoMetrics &m, uint64_t block_bytes) {
  const uint64_t blocks = m.disk_sync_block_reads.load(std::memory_order_relaxed);
  return IoMetricsSnapshot{
      blocks,
      blocks * block_bytes,
      m.disk_sync_via_uring.load(std::memory_order_relaxed),
      m.disk_sync_via_pread.load(std::memory_order_relaxed),
      m.prefetch_blocks_submitted.load(std::memory_order_relaxed),
  };
}

} // namespace amio
