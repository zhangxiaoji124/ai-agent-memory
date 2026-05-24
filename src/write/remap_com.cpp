#include "write/remap_com.h"

namespace amio::write {

BlockDisposition RemapComTracker::classify(const index::NodeBlock &prev,
                                           const index::NodeBlock &next) const {
  if (std::memcmp(&prev, &next, sizeof(index::NodeBlock)) == 0) {
    return BlockDisposition::Unchanged;
  }
  return BlockDisposition::Changed;
}

void RemapComTracker::record_remap(uint64_t node_id, uint64_t physical_offset) {
  remap_table_.push_back({node_id, physical_offset});
  remapped_blocks_++;
  unchanged_blocks_++;
}

} // namespace amio::write
