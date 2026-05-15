#include "write/memtable.h"

#include <algorithm>

#include "util/topk.h"

namespace amio::write {

static float l2_sq(const std::vector<float> &a, const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  float s = 0.0f;
  for (size_t i = 0; i < n; i++) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

MemTable::MemTable(size_t size_limit_bytes) : limit_(size_limit_bytes) {}

bool MemTable::insert(uint64_t node_id, std::vector<float> vec) {
  std::lock_guard lk(mu_);
  const size_t bytes = vec.size() * sizeof(float);
  auto it = data_.find(node_id);
  if (it != data_.end()) {
    used_ -= it->second.size() * sizeof(float);
    it->second = std::move(vec);
  } else {
    data_.emplace(node_id, std::move(vec));
  }
  used_ += bytes;
  return used_ >= limit_;
}

std::optional<std::vector<float>> MemTable::get(uint64_t node_id) const {
  std::lock_guard lk(mu_);
  auto it = data_.find(node_id);
  if (it == data_.end())
    return std::nullopt;
  return it->second;
}

std::vector<std::pair<uint64_t, std::vector<float>>> MemTable::drain() {
  std::lock_guard lk(mu_);
  std::vector<std::pair<uint64_t, std::vector<float>>> out;
  out.reserve(data_.size());
  for (auto &kv : data_) {
    out.push_back({kv.first, std::move(kv.second)});
  }
  data_.clear();
  used_ = 0;
  return out;
}

std::vector<std::pair<uint64_t, float>>
MemTable::brute_force_search(const std::vector<float> &query, size_t k) const {
  std::lock_guard lk(mu_);
  std::vector<amio::util::Scored<uint64_t, float>> scored;
  scored.reserve(data_.size());
  for (const auto &kv : data_) {
    scored.push_back({kv.first, l2_sq(query, kv.second)});
  }
  amio::util::topk_inplace(scored, k);
  std::vector<std::pair<uint64_t, float>> out;
  out.reserve(scored.size());
  for (const auto &s : scored)
    out.push_back({s.id, s.score});
  return out;
}

} // namespace amio::write