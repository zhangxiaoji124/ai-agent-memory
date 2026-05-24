#include "write/noblsm_commit.h"

namespace amio::write {

NobLsmCommitTracker::NobLsmCommitTracker(std::chrono::milliseconds grace) : grace_(grace) {}

void NobLsmCommitTracker::check_commit(const std::string &path) {
  if (path.empty()) {
    return;
  }
  std::lock_guard lk(mu_);
  pending_.emplace(path, Pending{});
}

void NobLsmCommitTracker::mark_data_written(const std::string &path) {
  if (path.empty()) {
    return;
  }
  std::lock_guard lk(mu_);
  auto it = pending_.find(path);
  if (it == pending_.end()) {
    pending_.emplace(path, Pending{});
    it = pending_.find(path);
  }
  it->second.data_written = true;
  it->second.data_written_at = std::chrono::steady_clock::now();
}

bool NobLsmCommitTracker::is_committed(const std::string &path) const {
  std::lock_guard lk(mu_);
  auto it = pending_.find(path);
  if (it == pending_.end() || !it->second.data_written) {
    return false;
  }
  const auto elapsed = std::chrono::steady_clock::now() - it->second.data_written_at;
  return elapsed >= grace_;
}

std::vector<std::string> NobLsmCommitTracker::poll_committed() {
  std::vector<std::string> out;
  std::lock_guard lk(mu_);
  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.data_written &&
        std::chrono::steady_clock::now() - it->second.data_written_at >= grace_) {
      out.push_back(it->first);
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }
  return out;
}

size_t NobLsmCommitTracker::pending_count() const {
  std::lock_guard lk(mu_);
  return pending_.size();
}

} // namespace amio::write
