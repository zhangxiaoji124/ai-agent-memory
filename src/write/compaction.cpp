#include "write/compaction.h"

#include <chrono>

namespace amio::write {

CompactionWorker::CompactionWorker(ApplyFn apply) : apply_(std::move(apply)) {
  th_ = std::thread([this] { loop(); });
}

CompactionWorker::~CompactionWorker() {
  stop_.store(true, std::memory_order_relaxed);
  if (th_.joinable())
    th_.join();
}

void CompactionWorker::submit(Batch batch) {
  std::lock_guard lk(mu_);
  q_.push_back(std::move(batch));
}

void CompactionWorker::loop() {
  while (!stop_.load(std::memory_order_relaxed)) {
    Batch batch;
    {
      std::lock_guard lk(mu_);
      if (!q_.empty()) {
        batch = std::move(q_.front());
        q_.erase(q_.begin());
      }
    }
    if (!batch.empty() && apply_) {
      apply_(batch);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

} // namespace amio::write