#pragma once

#include <cstdint>
#include <vector>

#include "index/hnsw.h"
#include "index/storage_layout.h"
#include "write/compaction.h"
#include "write/remap_com.h"

namespace amio::write {

struct IndexMergeStats {
  uint64_t nodes_merged = 0;
  uint64_t blocks_written = 0;
  uint64_t blocks_remapped = 0;
  bool ok = false;
};

/// 将 compaction 批次合并进内存 HNSW 并增量写回 `.index` 文件。
/// 要求 batch 内 node_id 与当前 `header.total_nodes` 起连续递增。
IndexMergeStats merge_batch_into_index(index::IndexFile &file, index::IndexFileHeader &header,
                                       index::HnswIndex &graph,
                                       const CompactionWorker::Batch &batch,
                                       RemapComTracker *remap = nullptr);

/// 全量 RemapCom 压缩：unchanged NodeBlock 复用原物理偏移写回新文件。
bool compact_index_with_remap(const std::string &src_path, const std::string &dst_path,
                              std::string *report = nullptr);

} // namespace amio::write
