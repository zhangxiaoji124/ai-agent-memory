#pragma once

#include <cstddef>
#include <cstdint>

#include "index/storage_layout.h"
#include "policy/agent_io_policy.h"
#include "prefetch/io_uring_backend.h"
#include "util/io_metrics.h"

// Forward declaration — 避免循环依赖（graph_aware_cache.h → io_uring_backend.h）
namespace amio::cache {
class GraphAwareCache;
}

namespace amio::prefetch {

class TopologyPrefetcher {
public:
  TopologyPrefetcher(size_t prefetch_depth,
                     const policy::AgentIoPolicy *policy = nullptr,
                     IoMetrics *io_metrics = nullptr);

  /// 非拥有指针：通常指向 `VectorStore` 持有的 `IoUringBackend`。
  void set_io_backend(IoBackend *backend) { io_ = backend; }

  /// 非拥有指针：用于 PAIC 预取提交记录（record_prefetch_submitted）。
  void set_cache(cache::GraphAwareCache *cache) { cache_ = cache; }

  // phase2=true 表示搜索已进入细粒度探索期（θ 阈值触发后），使用更激进的预取扇出。
  void on_visit_node(const amio::index::NodeBlock &node, uint8_t layer, bool phase2 = false);

private:
  size_t depth_ = 0;
  const policy::AgentIoPolicy *policy_ = nullptr;
  IoMetrics *io_metrics_ = nullptr;
  IoBackend *io_ = nullptr;
  cache::GraphAwareCache *cache_ = nullptr;

  static bool use_legacy_prefetch(const policy::AgentIoPolicy &p);
};

} // namespace amio::prefetch
