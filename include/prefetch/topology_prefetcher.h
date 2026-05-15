#pragma once

#include <cstddef>
#include <cstdint>

#include "index/storage_layout.h"
#include "policy/agent_io_policy.h"
#include "prefetch/io_uring_backend.h"
#include "util/io_metrics.h"

namespace amio::prefetch {

class TopologyPrefetcher {
public:
  TopologyPrefetcher(size_t prefetch_depth,
                     const policy::AgentIoPolicy *policy = nullptr,
                     IoMetrics *io_metrics = nullptr);

  /// 非拥有指针：通常指向 `VectorStore` 持有的 `IoUringBackend`。
  void set_io_backend(IoBackend *backend) { io_ = backend; }

  void on_visit_node(const amio::index::NodeBlock &node, uint8_t layer);

private:
  size_t depth_ = 0;
  const policy::AgentIoPolicy *policy_ = nullptr;
  IoMetrics *io_metrics_ = nullptr;
  IoBackend *io_ = nullptr;

  static bool use_legacy_prefetch(const policy::AgentIoPolicy &p);
};

} // namespace amio::prefetch
