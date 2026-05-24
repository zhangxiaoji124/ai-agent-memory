#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace amio::write {

/// NobLSM 简化：异步提交跟踪，避免每次 WAL/索引刷盘 fsync 阻塞。
/// 用户态模拟 check_commit / is_committed：数据写完后登记，经 grace 期视为已提交。
class NobLsmCommitTracker {
public:
  explicit NobLsmCommitTracker(std::chrono::milliseconds grace = std::chrono::milliseconds(50));

  /// 登记待提交文件（如 WAL 段、索引快照路径）。
  void check_commit(const std::string &path);

  /// 标记数据已异步写完（对应内核 writeback 完成）。
  void mark_data_written(const std::string &path);

  /// 是否已过 grace 期可安全删除旧文件 / 截断 WAL。
  bool is_committed(const std::string &path) const;

  /// 返回已提交路径列表并从 pending 移除。
  std::vector<std::string> poll_committed();

  size_t pending_count() const;

private:
  struct Pending {
    std::chrono::steady_clock::time_point data_written_at{};
    bool data_written = false;
  };

  mutable std::mutex mu_;
  std::chrono::milliseconds grace_;
  std::unordered_map<std::string, Pending> pending_;
};

} // namespace amio::write
