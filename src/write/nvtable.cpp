#include "write/nvtable.h"

namespace amio::write {

NVTable::NVTable(size_t byte_limit) : limit_(byte_limit) {}

bool NVTable::append_slice(std::vector<Record> slice) {
  if (slice.empty()) {
    return false;
  }
  size_t add = 0;
  for (const auto &r : slice) {
    add += sizeof(uint64_t) + r.second.size() * sizeof(float);
  }
  std::lock_guard lk(mu_);
  chain_.push_back(std::move(slice));
  bytes_used_ += add;
  return bytes_used_ >= limit_;
}

std::vector<NVTable::Record> NVTable::drain_all() {
  std::lock_guard lk(mu_);
  std::vector<Record> out;
  size_t n = 0;
  for (const auto &sl : chain_) {
    n += sl.size();
  }
  out.reserve(n);
  for (auto &sl : chain_) {
    for (auto &r : sl) {
      out.push_back({r.first, std::move(r.second)});
    }
  }
  chain_.clear();
  bytes_used_ = 0;
  return out;
}

bool NVTable::empty() const {
  std::lock_guard lk(mu_);
  return chain_.empty();
}

size_t NVTable::slice_count() const {
  std::lock_guard lk(mu_);
  return chain_.size();
}

size_t NVTable::bytes_used() const {
  std::lock_guard lk(mu_);
  return bytes_used_;
}

} // namespace amio::write
