#include "runtime/static_subgraph.h"

#include <queue>
#include <unordered_set>
#include <vector>

#include "cache/graph_aware_cache.h"
#include "index/storage_layout.h"
#include "prefetch/io_uring_backend.h"

namespace amio::runtime {

uint64_t pin_static_navigation_subgraph(index::IndexFile &file, cache::GraphAwareCache &cache,
                                        prefetch::IoUringBackend *uring, uint64_t entry_point,
                                        uint8_t max_layer, uint64_t total_nodes,
                                        uint64_t byte_budget) {
  if (total_nodes == 0 || byte_budget < index::kBlockSize) {
    return 0;
  }
  if (entry_point >= total_nodes) {
    entry_point = 0;
  }

  uint64_t pinned_bytes = 0;
  std::unordered_set<uint64_t> seen;
  seen.reserve(4096);
  std::queue<uint64_t> q;
  q.push(entry_point);
  seen.insert(entry_point);

  const int top_layer = std::min(7, static_cast<int>(max_layer));

  while (!q.empty() && pinned_bytes + index::kBlockSize <= byte_budget) {
    const uint64_t id = q.front();
    q.pop();

    index::NodeBlock b{};
    bool loaded = false;
    if (uring && uring->ok()) {
      loaded = uring->read_node_sync(id, &b);
    }
    if (!loaded) {
      loaded = file.pread_node(id, &b);
    }
    if (!loaded) {
      continue;
    }

    if (!cache.is_pinned(id)) {
      cache.pin(id, b);
      pinned_bytes += index::kBlockSize;
    }

    for (int layer = top_layer; layer >= 1; layer--) {
      const uint8_t cnt = b.neighbor_counts[static_cast<size_t>(layer)];
      const size_t n = cnt > 32 ? 32 : cnt;
      for (size_t j = 0; j < n; j++) {
        const uint64_t nb = static_cast<uint64_t>(b.neighbors[static_cast<size_t>(layer)][j]);
        if (nb >= total_nodes) {
          continue;
        }
        if (seen.insert(nb).second) {
          q.push(nb);
        }
      }
    }
  }
  return pinned_bytes;
}

} // namespace amio::runtime
