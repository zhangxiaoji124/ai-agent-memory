#include "write/compaction.h"

namespace amio::write {

CompactionWorker::CompactionWorker(ApplyFn apply) : apply_(std::move(apply)) {
  th_ = std::thread([this] { loop(); });
}

CompactionWorker::~CompactionWorker() {
  {
    std::lock_guard lk(mu_);
    stop_.store(true, std::memory_order_relaxed);
  }
  cv_has_work_.notify_all();
  if (th_.joinable())
    th_.join();
}

void CompactionWorker::submit(Batch batch) {
  std::unique_lock lk(mu_);
  // Back-pressure: block if queue is full
  cv_has_space_.wait(lk, [this] {
    return stop_.load(std::memory_order_relaxed) || q_.size() < kMaxQueueDepth;
  });
  if (!stop_.load(std::memory_order_relaxed))
    q_.push_back(std::move(batch));
  lk.unlock();
  cv_has_work_.notify_one();
}

void CompactionWorker::flush() {
  std::unique_lock lk(mu_);
  cv_idle_.wait(lk, [this] { return q_.empty(); });
}

size_t CompactionWorker::queue_depth() const {
  std::lock_guard lk(mu_);
  return q_.size();
}

void CompactionWorker::loop() {
  while (true) {
    Batch batch;
    {
      std::unique_lock lk(mu_);
      cv_has_work_.wait(lk, [this] {
        return stop_.load(std::memory_order_relaxed) || !q_.empty();
      });
      if (q_.empty()) {
        // stop_ must be true
        break;
      }
      batch = std::move(q_.front());
      q_.erase(q_.begin());
    }
    cv_has_space_.notify_one();

    if (!batch.empty() && apply_) {
      apply_(batch);
    }

    {
      std::lock_guard lk(mu_);
      if (q_.empty())
        cv_idle_.notify_all();
    }
  }
}

} // namespace amio::write
