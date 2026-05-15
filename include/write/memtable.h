#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace amio::write {

class MemTable {
public:
  explicit MemTable(size_t size_limit_bytes);

  /// 返回 true 表示达到阈值，需要 flush。
  bool insert(uint64_t node_id, std::vector<float> vec);
  std::optional<std::vector<float>> get(uint64_t node_id) const;

  std::vector<std::pair<uint64_t, std::vector<float>>> drain();

  std::vector<std::pair<uint64_t, float>> brute_force_search(const std::vector<float> &query,
                                                             size_t k) const;

private:
  size_t limit_ = 0;
  mutable std::mutex mu_;
  std::map<uint64_t, std::vector<float>> data_;
  size_t used_ = 0;
};

} // namespace amio::write