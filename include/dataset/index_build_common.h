#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "index/hnsw.h"
#include "index/storage_layout.h"

namespace amio::dataset {

/// 从内存图构造单个 NodeBlock。`node_id` 为磁盘全局 id；图内用 `node_id - id_base` 索引。
index::NodeBlock make_node_block(const index::HnswIndex &idx, uint32_t dim, uint64_t node_id);

/// 将 HNSW 图写入 NodeBlock 序列；dim>128 时不写内联向量（走外置向量区）。
bool write_graph_nodes(index::IndexFile &f, const index::HnswIndex &idx, uint32_t dim,
                       uint64_t total_nodes);

/// 增量写回：写单个节点块；v2 时追加外置向量行。
bool write_single_node(index::IndexFile &f, index::IndexFileHeader *header,
                       const index::HnswIndex &idx, uint32_t dim, uint64_t node_id,
                       const std::vector<float> &vec);

/// 根据 dim 选择 v1 内联或 v2 外置向量区并完成索引头。
bool finalize_index_vectors(
    index::IndexFile &f, index::IndexFileHeader *h, uint32_t dim,
    index::VectorEncoding enc,
    const std::function<bool(uint64_t, std::vector<float> *)> &fill_float);

} // namespace amio::dataset
