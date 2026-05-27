#include "cache/graph_aware_cache.h"

#include <algorithm>
#include <limits>
#include <mutex>

#include "cache/isvm_scorer.h"
#include "prefetch/io_uring_backend.h"
#include "util/io_metrics.h"

namespace amio::cache {

static uint32_t estimate_kv_group_bytes(const amio::index::NodeBlock &b) {
  const uint8_t cnt = b.neighbor_counts[0];
  return static_cast<uint32_t>(cnt) * static_cast<uint32_t>(sizeof(uint32_t));
}

int64_t GraphAwareCache::ev_key_for(bool is_prefetch_loaded, uint32_t kv_group_bytes,
                                    uint64_t tick) const {
  // C = ISVM 评分中与时间无关的分量（tick_age=0 时即为该分量）。
  // 真实驱逐分 score = tick_age/16 + C；驱逐时 tick_ 对所有条目相同，
  // 故 argmax(score) ≡ argmax(16*C - tick)（floor 引入 ≤1 桶误差，对启发式可忽略）。
  const int32_t c = isvm_.eviction_score(is_prefetch_loaded, /*tick_age=*/0, kv_group_bytes);
  return static_cast<int64_t>(c) * 16 - static_cast<int64_t>(tick);
}

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
    // O(1) 命中：只更新 tick / 预取标志；ev_index_ 中的键此刻变陈旧，
    // 不在此处做 set 增删，改由驱逐时惰性校正（lazy priority queue），
    // 避免每次命中付 O(log n) 的有序集合手术。
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
  auto existing = hot_.find(node_id);
  if (existing != hot_.end()) {
    // 覆盖已有条目：先摘掉旧的驱逐索引项。
    ev_index_.erase({existing->second.ev_key, node_id});
  } else {
    hot_used_bytes_ += amio::index::kBlockSize;
  }
  HotEntry e;
  e.block = b;
  e.tick = ++tick_;
  e.is_prefetch_loaded = is_prefetch_loaded;
  e.kv_group_bytes = estimate_kv_group_bytes(b);
  e.ev_key = ev_key_for(e.is_prefetch_loaded, e.kv_group_bytes, e.tick);
  hot_[node_id] = e;
  ev_index_.insert({e.ev_key, node_id});
  evict_if_needed_locked();
}

void GraphAwareCache::set_isvm_kv_cache_enabled(bool enabled) {
  std::unique_lock lk(mu_);
  isvm_.set_enabled(enabled);
}

void GraphAwareCache::evict_if_needed_locked() {
  if (dynamic_capacity_bytes_ == 0)
    return;
  // O(log n) 惰性驱逐：ev_index_ 中 ev_key 最大者（rbegin）是候选淘汰项；
  // 因命中不再实时更新键，候选可能“陈旧”（实际更晚被访问、更不该淘汰），
  // 此时用当前 tick/标志重算真实键、回插再重试；命中而陈旧的条目至多被校正一次。
  while (hot_used_bytes_ > dynamic_capacity_bytes_ && !ev_index_.empty()) {
    auto worst = std::prev(ev_index_.end());
    const int64_t key_stored = worst->first;
    const uint64_t victim = worst->second;
    auto it = hot_.find(victim);
    if (it == hot_.end()) {
      // 孤儿索引项（理论上不出现）：直接丢弃。
      ev_index_.erase(worst);
      continue;
    }
    const int64_t key_now =
        ev_key_for(it->second.is_prefetch_loaded, it->second.kv_group_bytes, it->second.tick);
    if (key_now != key_stored) {
      // 陈旧：命中后真实键已变，校正后重试（真正的最大值可能在别处）。
      ev_index_.erase(worst);
      it->second.ev_key = key_now;
      ev_index_.insert({key_now, victim});
      continue;
    }
    // 校验通过 → 确为 ISVM 评分最高者，执行淘汰。
    // PAIC: 驱逐仍标记为预取加载的条目 → 无效预取
    if (it->second.is_prefetch_loaded && io_metrics_) {
      io_metrics_->wasted_prefetch_evictions.fetch_add(1, std::memory_order_relaxed);
    }
    ev_index_.erase(worst);
    hot_.erase(it);
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
