#include "index/storage_layout.h"

#include <cerrno>
#include <cstring>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace amio::index {

IndexFile::~IndexFile() { close(); }

void IndexFile::close() {
  if (fd_ < 0)
    return;
#if defined(_WIN32)
  _close(fd_);
#else
  ::close(fd_);
#endif
  fd_ = -1;
}

static int open_rdonly(const char *path) {
#if defined(_WIN32)
  return _open(path, _O_BINARY | _O_RDONLY);
#else
  return ::open(path, O_RDONLY);
#endif
}

static int open_create_trunc_fd(const char *path) {
#if defined(_WIN32)
  int mode = 0;
#if defined(_S_IREAD) && defined(_S_IWRITE)
  mode = _S_IREAD | _S_IWRITE;
#elif defined(S_IREAD) && defined(S_IWRITE)
  mode = S_IREAD | S_IWRITE;
#else
  mode = 0666;
#endif
  return _open(path, _O_BINARY | _O_CREAT | _O_TRUNC | _O_RDWR, mode);
#else
  return ::open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
#endif
}

bool IndexFile::open_readonly(const std::string &path) {
  close();
  fd_ = open_rdonly(path.c_str());
  return fd_ >= 0;
}

bool IndexFile::open_create_trunc(const std::string &path) {
  close();
  fd_ = open_create_trunc_fd(path.c_str());
  return fd_ >= 0;
}

static bool pread_all(int fd, uint64_t off, void *buf, size_t len) {
  uint8_t *p = reinterpret_cast<uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
#if defined(_WIN32)
    if (_lseeki64(fd, static_cast<__int64>(off + done), SEEK_SET) < 0)
      return false;
    int r = _read(fd, p + done, static_cast<unsigned int>(len - done));
#else
    ssize_t r = ::pread(fd, p + done, len - done, static_cast<off_t>(off + done));
#endif
    if (r <= 0)
      return false;
    done += static_cast<size_t>(r);
  }
  return true;
}

static bool pwrite_all(int fd, uint64_t off, const void *buf, size_t len) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
#if defined(_WIN32)
    if (_lseeki64(fd, static_cast<__int64>(off + done), SEEK_SET) < 0)
      return false;
    int r = _write(fd, p + done, static_cast<unsigned int>(len - done));
#else
    ssize_t r = ::pwrite(fd, p + done, len - done, static_cast<off_t>(off + done));
#endif
    if (r <= 0)
      return false;
    done += static_cast<size_t>(r);
  }
  return true;
}

bool IndexFile::read_header(IndexFileHeader *out) const {
  if (!out || fd_ < 0)
    return false;
  IndexFileHeader h{};
  if (!pread_all(fd_, 0, &h, sizeof(h)))
    return false;
  if (h.magic != kMagic)
    return false;
  *out = h;
  return true;
}

bool IndexFile::write_header(const IndexFileHeader &h) {
  if (fd_ < 0)
    return false;
  return pwrite_all(fd_, 0, &h, sizeof(h));
}

bool IndexFile::pread_node(uint64_t node_id, NodeBlock *out) const {
  if (!out || fd_ < 0)
    return false;
  return pread_all(fd_, node_offset(node_id), out, sizeof(NodeBlock));
}

bool IndexFile::pwrite_node(uint64_t node_id, const NodeBlock &b) {
  if (fd_ < 0)
    return false;
  return pwrite_all(fd_, node_offset(node_id), &b, sizeof(NodeBlock));
}

} // namespace amio::index