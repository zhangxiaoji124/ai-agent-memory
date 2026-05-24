#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace amio::write {

class Wal {
public:
  explicit Wal(const std::string &path);
  ~Wal();

  Wal(const Wal &) = delete;
  Wal &operator=(const Wal &) = delete;

  bool ok() const { return ok_; }

  bool append(uint64_t node_id, const std::vector<float> &vec);

  /// NobLSM：异步提交完成后截断 WAL（非阻塞语义）。
  bool truncate();

private:
  int fd_ = -1;
  bool ok_ = false;
  std::mutex mu_;
};

} // namespace amio::write