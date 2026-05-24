#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "index/storage_layout.h"

namespace amio::prefetch {

class IoBackend {
public:
  virtual ~IoBackend() = default;
  virtual bool submit_prefetch(const std::vector<uint64_t> &node_ids) = 0;
  // 非阻塞收割已完成的 io_uring CQE，释放 slot 供后续预取复用。
  // 非 io_uring 后端默认为空操作。
  virtual void drain_completions_nonblocking(int max_events = 32) { (void)max_events; }
};

class NoopBackend final : public IoBackend {
public:
  bool submit_prefetch(const std::vector<uint64_t> &node_ids) override;
};

/// Linux + liburing + O_DIRECT fd。
class IoUringBackend final : public IoBackend {
public:
  IoUringBackend(int odirect_fd, uint32_t ring_depth);
  ~IoUringBackend() override;

  IoUringBackend(const IoUringBackend &) = delete;
  IoUringBackend &operator=(const IoUringBackend &) = delete;

  bool ok() const;

  bool submit_prefetch(const std::vector<uint64_t> &node_ids) override;

  bool read_node_sync(uint64_t node_id, amio::index::NodeBlock *out);

  void drain_completions_nonblocking(int max_events);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace amio::prefetch
