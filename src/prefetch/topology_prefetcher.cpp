#include "prefetch/topology_prefetcher.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "util/macros.h"

namespace amio::prefetch {

TopologyPrefetcher::TopologyPrefetcher(size_t prefetch_depth,
                                       const policy::AgentIoPolicy *policy,
                                       IoMetrics *io_metrics)
    : depth_(prefetch_depth), policy_(policy), io_metrics_(io_metrics) {}

bool TopologyPrefetcher::use_legacy_prefetch(const policy::AgentIoPolicy &p) {
  return !p.use_layer_aware_prefetch && !p.sort_prefetch_by_disk_offset &&
         p.prefetch_depth_upper == p.prefetch_depth_layer0 &&
         p.max_neighbor_fanout_layer0 == 32;
}

void TopologyPrefetcher::on_visit_node(const amio::index::NodeBlock &node,
                                         uint8_t layer) {
  if (!io_)
    return;

  const policy::AgentIoPolicy &pol =
      policy_ ? *policy_ : policy::AgentIoPolicy{};

  if (!policy_) {
    if (depth_ == 0)
      return;
    std::vector<uint64_t> ids;
    ids.reserve(256);
    const size_t d = depth_;
    for (size_t rep = 0; rep < d; rep++) {
      AMIO_UNUSED(rep);
      for (size_t l = 0; l < 8; l++) {
        const uint8_t cnt = node.neighbor_counts[l];
        const size_t n = cnt > 32 ? 32 : cnt;
        for (size_t i = 0; i < n; i++)
          ids.push_back(static_cast<uint64_t>(node.neighbors[l][i]));
      }
    }
    if (io_metrics_)
      io_metrics_->prefetch_blocks_submitted.fetch_add(ids.size(),
                                                       std::memory_order_relaxed);
    (void)io_->submit_prefetch(ids);
    return;
  }

  if (use_legacy_prefetch(pol)) {
    const size_t d = pol.prefetch_depth_layer0;
    if (d == 0)
      return;
    std::vector<uint64_t> ids;
    ids.reserve(256);
    for (size_t rep = 0; rep < d; rep++) {
      AMIO_UNUSED(rep);
      for (size_t l = 0; l < 8; l++) {
        const uint8_t cnt = node.neighbor_counts[l];
        const size_t n = cnt > 32 ? 32 : cnt;
        for (size_t i = 0; i < n; i++)
          ids.push_back(static_cast<uint64_t>(node.neighbors[l][i]));
      }
    }
    if (io_metrics_)
      io_metrics_->prefetch_blocks_submitted.fetch_add(ids.size(),
                                                       std::memory_order_relaxed);
    (void)io_->submit_prefetch(ids);
    return;
  }

  const size_t d =
      layer == 0 ? pol.prefetch_depth_layer0 : pol.prefetch_depth_upper;
  if (d == 0)
    return;

  const size_t max_fan =
      layer == 0 ? std::min<size_t>(32, pol.max_neighbor_fanout_layer0) : 32u;
  std::vector<std::pair<uint64_t, uint64_t>> tagged;
  tagged.reserve(256);

  for (size_t rep = 0; rep < d; rep++) {
    AMIO_UNUSED(rep);
    if (pol.use_layer_aware_prefetch) {
      const uint8_t cnt = node.neighbor_counts[layer];
      const size_t n = std::min(static_cast<size_t>(cnt), max_fan);
      for (size_t i = 0; i < n; i++) {
        const uint64_t nid = static_cast<uint64_t>(node.neighbors[layer][i]);
        const uint64_t off = node.neighbor_offsets[layer][i];
        tagged.push_back({off, nid});
      }
    } else {
      for (size_t l = 0; l < 8; l++) {
        const uint8_t cnt = node.neighbor_counts[l];
        const size_t n = l == 0 ? std::min(static_cast<size_t>(cnt), max_fan)
                                : (cnt > 32 ? 32 : cnt);
        for (size_t i = 0; i < n; i++) {
          const uint64_t nid = static_cast<uint64_t>(node.neighbors[l][i]);
          const uint64_t off = node.neighbor_offsets[l][i];
          tagged.push_back({off, nid});
        }
      }
    }
  }

  if (pol.sort_prefetch_by_disk_offset)
    std::sort(tagged.begin(), tagged.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<uint64_t> ids;
  ids.reserve(tagged.size());
  for (const auto &pr : tagged)
    ids.push_back(pr.second);
  if (io_metrics_)
    io_metrics_->prefetch_blocks_submitted.fetch_add(ids.size(),
                                                     std::memory_order_relaxed);
  (void)io_->submit_prefetch(ids);
}

} // namespace amio::prefetch
