#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace amio::write {

/// NVTable 简化：WAL-free 内存链表，吸收 MemTable flush 批次后再统一 compaction。
class NVTable {
public:
  using Record = std::pair<uint64_t, std::vector<float>>;

  explicit NVTable(size_t byte_limit = 64 * 1024 * 1024);

  /// 追加一批记录；返回 true 表示超过 byte_limit 建议触发 L0 flush。
  bool append_slice(std::vector<Record> slice);

  std::vector<Record> drain_all();

  bool empty() const;
  size_t slice_count() const;
  size_t bytes_used() const;

private:
  size_t limit_ = 0;
  size_t bytes_used_ = 0;
  mutable std::mutex mu_;
  std::vector<std::vector<Record>> chain_;
};

} // namespace amio::write
