#include "prefetch/io_uring_backend.h"

#include <cstring>

#include "util/macros.h"

#if defined(AMIO_HAS_URING) && AMIO_HAS_URING && defined(__linux__)
#include <liburing.h>
#include <cstdlib>
#include <unistd.h>
#endif

namespace amio::prefetch {

bool NoopBackend::submit_prefetch(const std::vector<uint64_t> &node_ids) {
  AMIO_UNUSED(node_ids);
  return true;
}

#if defined(AMIO_HAS_URING) && AMIO_HAS_URING && defined(__linux__)

struct IoUringBackend::Impl {
  int fd = -1;
  io_uring ring {};
  bool ring_inited = false;
  uint32_t depth = 0;
  struct Slot {
    void *buf = nullptr;
    bool busy = false;
    uint64_t node_id = 0;
  };
  std::vector<Slot> slots;
};

IoUringBackend::IoUringBackend(int odirect_fd, uint32_t ring_depth)
    : impl_(new Impl()) {
  if (odirect_fd < 0 || ring_depth == 0) {
    impl_.reset();
    return;
  }
  impl_->fd = odirect_fd;
  impl_->depth = ring_depth;
  if (io_uring_queue_init(ring_depth, &impl_->ring, 0) != 0) {
    impl_.reset();
    return;
  }
  impl_->ring_inited = true;
  impl_->slots.resize(ring_depth);
  for (auto &s : impl_->slots) {
    if (::posix_memalign(&s.buf, amio::index::kBlockSize, amio::index::kBlockSize) != 0) {
      io_uring_queue_exit(&impl_->ring);
      impl_->ring_inited = false;
      impl_.reset();
      return;
    }
  }
}

IoUringBackend::~IoUringBackend() {
  if (!impl_)
    return;
  for (auto &s : impl_->slots) {
    if (s.buf)
      std::free(s.buf);
  }
  if (impl_->ring_inited) {
    io_uring_queue_exit(&impl_->ring);
  }
}

bool IoUringBackend::ok() const { return static_cast<bool>(impl_ && impl_->ring_inited); }

bool IoUringBackend::read_node_sync(uint64_t node_id,
                                     amio::index::NodeBlock *out) {
  if (!ok() || !out)
    return false;
  void *buf = nullptr;
  if (::posix_memalign(&buf, amio::index::kBlockSize, amio::index::kBlockSize) != 0)
    return false;
  const off_t off = static_cast<off_t>(amio::index::node_offset(node_id));
  io_uring_sqe *sqe = io_uring_get_sqe(&impl_->ring);
  if (!sqe) {
    std::free(buf);
    return false;
  }
  io_uring_prep_read(sqe, impl_->fd, buf, static_cast<unsigned>(amio::index::kBlockSize),
                     off);
  io_uring_submit(&impl_->ring);
  io_uring_cqe *cqe = nullptr;
  io_uring_wait_cqe(&impl_->ring, &cqe);
  const int res = cqe ? cqe->res : -1;
  if (cqe)
    io_uring_cqe_seen(&impl_->ring, cqe);
  if (res != static_cast<int>(amio::index::kBlockSize)) {
    std::free(buf);
    return false;
  }
  std::memcpy(out, buf, amio::index::kBlockSize);
  std::free(buf);
  return true;
}

bool IoUringBackend::submit_prefetch(const std::vector<uint64_t> &node_ids) {
  if (!ok())
    return true;
  uint32_t submitted = 0;
  for (uint64_t id : node_ids) {
    int slot_idx = -1;
    for (size_t i = 0; i < impl_->slots.size(); i++) {
      if (!impl_->slots[i].busy) {
        slot_idx = static_cast<int>(i);
        break;
      }
    }
    if (slot_idx < 0)
      break;
    auto &slot = impl_->slots[static_cast<size_t>(slot_idx)];
    slot.busy = true;
    slot.node_id = id;
    const off_t off = static_cast<off_t>(amio::index::node_offset(id));
    io_uring_sqe *sqe = io_uring_get_sqe(&impl_->ring);
    if (!sqe) {
      slot.busy = false;
      break;
    }
    io_uring_prep_read(sqe, impl_->fd, slot.buf,
                       static_cast<unsigned>(amio::index::kBlockSize), off);
    io_uring_sqe_set_data64(sqe,
                            (static_cast<uint64_t>(slot_idx) << 32) ^
                                static_cast<uint64_t>(id + 0x9e3779b97f4a7c15ULL));
    submitted++;
  }
  if (submitted > 0)
    (void)io_uring_submit(&impl_->ring);
  return true;
}

void IoUringBackend::drain_completions_nonblocking(int max_events) {
  if (!ok() || max_events <= 0)
    return;
  int got = 0;
  io_uring_cqe *cqe = nullptr;
  while (got < max_events &&
         io_uring_peek_cqe(&impl_->ring, &cqe) == 0 && cqe != nullptr) {
    io_uring_cqe_seen(&impl_->ring, cqe);
    (void)(cqe->res);
    // 简化：不在此处解析 user_data；仅释放队列上的完成项。
    got++;
  }
}

#else

struct IoUringBackend::Impl {};

IoUringBackend::IoUringBackend(int, uint32_t) : impl_(nullptr) {}
IoUringBackend::~IoUringBackend() = default;
bool IoUringBackend::ok() const { return false; }
bool IoUringBackend::read_node_sync(uint64_t, amio::index::NodeBlock *) { return false; }
bool IoUringBackend::submit_prefetch(const std::vector<uint64_t> &) { return true; }
void IoUringBackend::drain_completions_nonblocking(int) {}

#endif

} // namespace amio::prefetch
