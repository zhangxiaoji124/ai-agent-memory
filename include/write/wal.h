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

private:
  int fd_ = -1;
  bool ok_ = false;
  std::mutex mu_;
};

} // namespace amio::write