#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace amio::write {

/// 后台 compaction：把 MemTable 批次合并进主索引（通过回调实现，骨架）。
class CompactionWorker {
public:
  using Batch = std::vector<std::pair<uint64_t, std::vector<float>>>;
  using ApplyFn = std::function<void(const Batch &)>;

  explicit CompactionWorker(ApplyFn apply);
  ~CompactionWorker();

  CompactionWorker(const CompactionWorker &) = delete;
  CompactionWorker &operator=(const CompactionWorker &) = delete;

  void submit(Batch batch);

private:
  ApplyFn apply_;
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::vector<Batch> q_;
  std::thread th_;

  void loop();
};

} // namespace amio::write