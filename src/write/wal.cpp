#include "write/wal.h"

#include <cstdint>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace amio::write {

static int open_wal(const char *path) {
#if defined(_WIN32)
  int mode = 0;
#if defined(_S_IREAD) && defined(_S_IWRITE)
  mode = _S_IREAD | _S_IWRITE;
#elif defined(S_IREAD) && defined(S_IWRITE)
  mode = S_IREAD | S_IWRITE;
#else
  mode = 0666;
#endif
  return _open(path, _O_BINARY | _O_CREAT | _O_APPEND | _O_WRONLY, mode);
#else
  return ::open(path, O_CREAT | O_APPEND | O_WRONLY, 0644);
#endif
}

Wal::Wal(const std::string &path) {
  fd_ = open_wal(path.c_str());
  ok_ = fd_ >= 0;
}

Wal::~Wal() {
  if (fd_ < 0)
    return;
#if defined(_WIN32)
  _close(fd_);
#else
  ::close(fd_);
#endif
  fd_ = -1;
}

static bool write_all(int fd, const void *buf, size_t len) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
#if defined(_WIN32)
    int r = _write(fd, p + done, static_cast<unsigned int>(len - done));
#else
    ssize_t r = ::write(fd, p + done, len - done);
#endif
    if (r <= 0)
      return false;
    done += static_cast<size_t>(r);
  }
  return true;
}

bool Wal::append(uint64_t node_id, const std::vector<float> &vec) {
  if (!ok_)
    return false;
  std::lock_guard lk(mu_);
  const uint32_t dim = static_cast<uint32_t>(vec.size());
  if (!write_all(fd_, &node_id, sizeof(node_id)))
    return false;
  if (!write_all(fd_, &dim, sizeof(dim)))
    return false;
  if (!vec.empty()) {
    return write_all(fd_, vec.data(), vec.size() * sizeof(float));
  }
  return true;
}

bool Wal::truncate() {
  if (!ok_)
    return false;
  std::lock_guard lk(mu_);
#if defined(_WIN32)
  return _chsize_s(fd_, 0) == 0;
#else
  return ::ftruncate(fd_, 0) == 0;
#endif
}

} // namespace amio::write