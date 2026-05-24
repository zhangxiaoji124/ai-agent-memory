#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace amio::write {

/// 后台 compaction：把 MemTable 批次合并进主索引（通过回调实现）。
/// 使用条件变量实现零延迟唤醒；submit() 超过 kMaxQueueDepth 时阻塞以限制内存。
class CompactionWorker {
public:
  using Batch = std::vector<std::pair<uint64_t, std::vector<float>>>;
  using ApplyFn = std::function<void(const Batch &)>;

  /// 队列深度上限：防止写入速率远超 compaction 速率时无限堆积。
  static constexpr size_t kMaxQueueDepth = 16;

  explicit CompactionWorker(ApplyFn apply);
  ~CompactionWorker();

  CompactionWorker(const CompactionWorker &) = delete;
  CompactionWorker &operator=(const CompactionWorker &) = delete;

  /// 提交一个批次。若队列已满则阻塞直到有空位（背压）。
  void submit(Batch batch);

  /// 同步等待队列中所有批次处理完毕后返回（用于测试 / 关闭前刷盘）。
  void flush();

  size_t queue_depth() const;

private:
  ApplyFn apply_;
  std::atomic<bool> stop_{false};
  mutable std::mutex mu_;
  std::condition_variable cv_has_work_;
  std::condition_variable cv_has_space_;
  std::condition_variable cv_idle_;
  std::vector<Batch> q_;
  std::thread th_;

  void loop();
};

} // namespace amio::write