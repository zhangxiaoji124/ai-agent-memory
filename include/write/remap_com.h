#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "index/storage_layout.h"

namespace amio::write {

/// RemapCom 简化：4KB 块 UDB（未变更数据块）识别，合并时跳过重写。
enum class BlockDisposition : uint8_t {
  Changed = 0,
  Unchanged = 1,
};

class RemapComTracker {
public:
  BlockDisposition classify(const index::NodeBlock &prev, const index::NodeBlock &next) const;

  /// 记录一次 UDB 重映射（逻辑 node_id 复用原物理偏移，零拷贝语义）。
  void record_remap(uint64_t node_id, uint64_t physical_offset);

  uint64_t unchanged_blocks() const { return unchanged_blocks_; }
  uint64_t remapped_blocks() const { return remapped_blocks_; }

  const std::vector<std::pair<uint64_t, uint64_t>> &remap_table() const { return remap_table_; }

private:
  uint64_t unchanged_blocks_ = 0;
  uint64_t remapped_blocks_ = 0;
  std::vector<std::pair<uint64_t, uint64_t>> remap_table_;
};

} // namespace amio::write
