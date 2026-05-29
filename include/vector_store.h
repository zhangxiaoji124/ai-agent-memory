#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cache/graph_aware_cache.h"
#include "index/external_vectors.h"
#include "index/hnsw.h"
#include "index/pq.h"
#include "index/storage_layout.h"
#include "policy/agent_io_policy.h"
#include "prefetch/io_uring_backend.h"
#include "prefetch/topology_prefetcher.h"
#include "write/compaction.h"
#include "write/index_merger.h"
#include "write/memtable.h"
#include "write/noblsm_commit.h"
#include "write/nvtable.h"
#include "write/remap_com.h"
#include "write/wal.h"
#include "util/io_metrics.h"
#include "runtime/memory_partition.h"

namespace amio {

enum class AgentPolicyMode {
  /// 仅使用 `AgentIoPolicy::from_config` 的默认策略，不合并学习 JSON。
  Builtin,
  /// 在默认策略基础上合并 `agent_policy_path` 指向的 JSON。
  Learned,
};

struct SearchResult {
  uint64_t id = 0;
  float distance = 0.0f;
};

struct Config {
  std::string index_path = "data/sift_base.index";
  /// Dynamic 热缓存池（MB）；0 时由 apply_partition_to_config 填充。
  size_t cache_size_mb = 512;
  /// Static 常驻 pin 池（MB）。
  size_t static_cache_mb = 0;
  uint64_t ram_budget_bytes = 0;
  runtime::PartitionDecision partition{};
  bool enable_static_subgraph_pin = true;
  size_t prefetch_depth = 1;
  bool enable_wal = true;
  bool enable_compaction = true;
  bool enable_nvtable = true;
  bool enable_noblsm = true;
  bool enable_isvm_kv_cache = true;
  bool enable_memtable_search = true;
  bool force_disable_uring = false;
  size_t memtable_limit_mb = 64;
  size_t ef_search = 128;
  /// PQ 检索：>0 时若存在 `<index>.pq` 码本则启用 ADC 粗筛 + 全精度重排。
  size_t pq_rerank_mult = 8; // 重排候选数 = k * 该倍数
  /// 可选：离线学习的 Agent I/O 策略 JSON（见 `learning/train_agent_policy.py`）。
  std::string agent_policy_path;
  /// `Learned` 且路径非空时合并 JSON；`Builtin` 时忽略路径（基线对照）。
  AgentPolicyMode agent_policy_mode = AgentPolicyMode::Builtin;
};

/// 集成内存 HNSW、MemTable/WAL、磁盘 mmap/O_DIRECT 与 io_uring 预取。
class VectorStore {
public:
  explicit VectorStore(const Config &cfg);
  ~VectorStore();

  VectorStore(const VectorStore &) = delete;
  VectorStore &operator=(const VectorStore &) = delete;

  /// 若 `index_path` 可打开则加载头并初始化；失败时可纯内存运行（单测）。
  bool open();

  std::vector<SearchResult> search(const std::vector<float> &query, size_t k);
  bool insert(uint64_t id, const std::vector<float> &vec);

  /// 磁盘近似检索：按 `IndexFileHeader.max_layer` 做上层贪心 + Layer0 best-first。
  std::vector<SearchResult> search_disk(const std::vector<float> &query, size_t k);

  bool io_uring_active() const { return static_cast<bool>(uring_ && uring_->ok()); }

  /// 清零 I/O 与缓存访问计数（不清理缓存内容），用于逐查询统计。
  void reset_search_observability();

  IoMetricsSnapshot io_metrics_snapshot() const {
    return snapshot(io_metrics_, amio::index::kBlockSize);
  }

  uint64_t cache_hits_total() const { return cache_.hits_total(); }
  uint64_t cache_misses_total() const { return cache_.misses_total(); }
  uint64_t static_pins_count() const { return cache_.static_pins_count(); }

  /// 等待 compaction 队列清空（测试 / 关闭前刷盘）。
  void flush_writes();

  const policy::AgentIoPolicy &effective_io_policy() const { return io_policy_; }
  const runtime::PartitionDecision &partition() const { return cfg_.partition; }
  bool uses_external_vectors() const {
    return ext_vecs_ && ext_vecs_->ok();
  }
  bool uses_pq() const { return pq_ && pq_->ok() && !pq_codes_.empty(); }

private:
  Config cfg_;
  index::IndexFile file_;
  index::IndexFileHeader header_{};
  std::unique_ptr<index::ExternalVectorStore> ext_vecs_;

  // PQ：码本（驻留内存）+ 全部码（每节点 m 字节，驻留内存）。启用时检索走 ADC 粗筛 + 重排。
  std::unique_ptr<index::ProductQuantizer> pq_;
  std::vector<uint8_t> pq_codes_;

  mutable std::mutex index_mu_;
  index::HnswIndex index_;

  policy::AgentIoPolicy io_policy_{};
  IoMetrics io_metrics_{};
  cache::GraphAwareCache cache_;
  prefetch::TopologyPrefetcher prefetch_;
  std::unique_ptr<prefetch::IoUringBackend> uring_;

  write::MemTable memtable_;
  write::NVTable nvtable_;
  write::RemapComTracker remap_;
  write::NobLsmCommitTracker noblsm_;
  std::unique_ptr<write::Wal> wal_;
  std::unique_ptr<write::CompactionWorker> compaction_;
  uint64_t next_node_id_ = 0;
  bool index_writable_ = false;

  void submit_compaction_batch(write::CompactionWorker::Batch batch);
  void on_compaction_applied(const write::IndexMergeStats &st);

  static float l2_sq(const std::vector<float> &a, const std::vector<float> &b);
  static float l2_sq_block(const std::vector<float> &q,
                           const amio::index::NodeBlock &b, size_t dim);
};

} // namespace amio
