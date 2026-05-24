#include "cache/graph_aware_cache.h"

#include <algorithm>
#include <limits>
#include <mutex>

#include "prefetch/io_uring_backend.h"
#include "util/io_metrics.h"

namespace amio::cache {

GraphAwareCache::GraphAwareCache(size_t dynamic_capacity_bytes, size_t pinned_capacity_bytes,
                                 const policy::AgentIoPolicy *policy,
                                 ::amio::IoMetrics *io_metrics)
    : dynamic_capacity_bytes_(dynamic_capacity_bytes),
      pinned_capacity_bytes_(pinned_capacity_bytes),
      policy_(policy),
      io_metrics_(io_metrics) {}

bool GraphAwareCache::is_pinned(uint64_t node_id) const {
  std::shared_lock lk(mu_);
  return pinned_.find(node_id) != pinned_.end();
}

uint64_t GraphAwareCache::pinned_bytes_used() const {
  std::shared_lock lk(mu_);
  return pinned_used_bytes_;
}

bool GraphAwareCache::contains(uint64_t node_id) const {
  std::shared_lock lk(mu_);
  if (pinned_.find(node_id) != pinned_.end())
    return true;
  return hot_.find(node_id) != hot_.end();
}

bool GraphAwareCache::get(uint64_t node_id, amio::index::NodeBlock *out) {
  if (!out)
    return false;
  std::unique_lock lk(mu_);
  auto itp = pinned_.find(node_id);
  if (itp != pinned_.end()) {
    *out = itp->second;
    hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  auto ith = hot_.find(node_id);
  if (ith != hot_.end()) {
    ith->second.tick = ++tick_;
    // PAIC: 需求命中预取协同加载的条目 → 有效预取
    if (ith->second.is_prefetch_loaded) {
      ith->second.is_prefetch_loaded = false;
      if (io_metrics_)
        io_metrics_->useful_prefetch_demand_hits.fetch_add(1, std::memory_order_relaxed);
    }
    *out = ith->second.block;
    hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  misses_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool GraphAwareCache::get_or_load(amio::index::IndexFile *file, uint64_t node_id,
                                  amio::index::NodeBlock *out) {
  if (get(node_id, out))
    return true;
  amio::index::NodeBlock b{};
  if (!file || !file->pread_node(node_id, &b))
    return false;
  if (io_metrics_) {
    io_metrics_->disk_sync_block_reads.fetch_add(1, std::memory_order_relaxed);
    io_metrics_->disk_sync_via_pread.fetch_add(1, std::memory_order_relaxed);
  }
  const bool was_prefetch_pending = consume_prefetch_pending(node_id);
  std::unique_lock lk(mu_);
  const uint32_t need =
      policy_ ? std::max(1u, policy_->hot_insert_min_prior_misses) : 1u;
  uint32_t c = ++cold_misses_[node_id];
  if (cold_misses_.size() > 200000)
    cold_misses_.clear();
  if (c >= need) {
    insert_hot_locked(node_id, b, was_prefetch_pending);
    cold_misses_.erase(node_id);
  }
  *out = b;
  return true;
}

bool GraphAwareCache::get_or_load(amio::index::IndexFile *file,
                                  prefetch::IoUringBackend *uring, uint64_t node_id,
                                  amio::index::NodeBlock *out) {
  if (get(node_id, out))
    return true;
  amio::index::NodeBlock b{};
  bool ok = false;
  if (uring && uring->ok() && uring->read_node_sync(node_id, &b)) {
    ok = true;
    if (io_metrics_) {
      io_metrics_->disk_sync_block_reads.fetch_add(1, std::memory_order_relaxed);
      io_metrics_->disk_sync_via_uring.fetch_add(1, std::memory_order_relaxed);
    }
  } else if (file && file->pread_node(node_id, &b)) {
    ok = true;
    if (io_metrics_) {
      io_metrics_->disk_sync_block_reads.fetch_add(1, std::memory_order_relaxed);
      io_metrics_->disk_sync_via_pread.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (!ok)
    return false;
  const bool was_prefetch_pending = consume_prefetch_pending(node_id);
  std::unique_lock lk(mu_);
  const uint32_t need =
      policy_ ? std::max(1u, policy_->hot_insert_min_prior_misses) : 1u;
  uint32_t c = ++cold_misses_[node_id];
  if (cold_misses_.size() > 200000)
    cold_misses_.clear();
  if (c >= need) {
    insert_hot_locked(node_id, b, was_prefetch_pending);
    cold_misses_.erase(node_id);
  }
  *out = b;
  return true;
}

void GraphAwareCache::pin(uint64_t node_id, const amio::index::NodeBlock &b) {
  std::unique_lock lk(mu_);
  if (pinned_.find(node_id) == pinned_.end()) {
    if (pinned_capacity_bytes_ > 0 &&
        pinned_used_bytes_ + amio::index::kBlockSize > pinned_capacity_bytes_) {
      return;
    }
    pinned_used_bytes_ += amio::index::kBlockSize;
    static_pins_count_++;
  }
  pinned_[node_id] = b;
}

void GraphAwareCache::insert_hot(uint64_t node_id, const amio::index::NodeBlock &b) {
  std::unique_lock lk(mu_);
  insert_hot_locked(node_id, b);
}

void GraphAwareCache::insert_hot_locked(uint64_t node_id, const amio::index::NodeBlock &b,
                                        bool is_prefetch_loaded) {
  HotEntry e;
  e.block = b;
  e.tick = ++tick_;
  e.is_prefetch_loaded = is_prefetch_loaded;
  if (hot_.find(node_id) == hot_.end()) {
    hot_used_bytes_ += amio::index::kBlockSize;
  }
  hot_[node_id] = e;
  evict_if_needed_locked();
}

void GraphAwareCache::evict_if_needed_locked() {
  if (dynamic_capacity_bytes_ == 0)
    return;
  while (hot_used_bytes_ > dynamic_capacity_bytes_ && !hot_.empty()) {
    uint64_t victim = 0;
    uint64_t best = std::numeric_limits<uint64_t>::max();
    for (const auto &kv : hot_) {
      if (kv.second.tick < best) {
        best = kv.second.tick;
        victim = kv.first;
      }
    }
    auto it = hot_.find(victim);
    // PAIC: 驱逐仍标记为预取加载的条目 → 无效预取
    if (it != hot_.end() && it->second.is_prefetch_loaded && io_metrics_) {
      io_metrics_->wasted_prefetch_evictions.fetch_add(1, std::memory_order_relaxed);
    }
    hot_.erase(victim);
    hot_used_bytes_ -= amio::index::kBlockSize;
  }
}

double GraphAwareCache::hit_rate() const {
  const uint64_t h = hits_.load(std::memory_order_relaxed);
  const uint64_t m = misses_.load(std::memory_order_relaxed);
  const uint64_t t = h + m;
  if (t == 0)
    return 0.0;
  return static_cast<double>(h) / static_cast<double>(t);
}

void GraphAwareCache::reset_access_counters() {
  hits_.store(0, std::memory_order_relaxed);
  misses_.store(0, std::memory_order_relaxed);
}

void GraphAwareCache::record_prefetch_submitted(const std::vector<uint64_t> &ids) {
  if (ids.empty())
    return;
  std::lock_guard lk(prefetch_pending_mu_);
  if (prefetch_pending_.size() >= kMaxPrefetchPending) {
    prefetch_pending_.clear();
  }
  for (uint64_t id : ids) {
    prefetch_pending_.insert(id);
  }
}

bool GraphAwareCache::consume_prefetch_pending(uint64_t id) {
  std::lock_guard lk(prefetch_pending_mu_);
  return prefetch_pending_.erase(id) > 0;
}

} // namespace amio::cache
