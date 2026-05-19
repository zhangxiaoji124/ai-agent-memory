#pragma once

#include <cstdint>

namespace amio::index {
class IndexFile;
}
namespace amio::cache {
class GraphAwareCache;
}
namespace amio::prefetch {
class IoUringBackend;
}

namespace amio::runtime {

/// 从磁盘索引提取入口及高层导航子图并 pin 到 Static 缓存池（GoVector 静态区）。
uint64_t pin_static_navigation_subgraph(index::IndexFile &file, cache::GraphAwareCache &cache,
                                        prefetch::IoUringBackend *uring, uint64_t entry_point,
                                        uint8_t max_layer, uint64_t total_nodes,
                                        uint64_t byte_budget);

} // namespace amio::runtime
