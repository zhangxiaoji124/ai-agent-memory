#include "index/external_vectors.h"

#include <cmath>
#include <cstring>
#include <functional>

#include "util/simd_distance.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace amio::index {

ExternalVectorStore::~ExternalVectorStore() { detach(); }

void ExternalVectorStore::unmap_region() {
#if defined(_WIN32)
  if (map_base_) {
    UnmapViewOfFile(map_base_);
    map_base_ = nullptr;
  }
  if (map_handle_) {
    CloseHandle(static_cast<HANDLE>(map_handle_));
    map_handle_ = nullptr;
  }
#else
  if (map_base_ && map_base_ != MAP_FAILED) {
    munmap(map_base_, map_size_);
  }
  map_base_ = nullptr;
  map_size_ = 0;
#endif
}

void ExternalVectorStore::detach() {
  unmap_region();
  if (owns_fd_ && fd_ >= 0) {
#if defined(_WIN32)
    _close(fd_);
#else
    ::close(fd_);
#endif
  }
  fd_ = -1;
  owns_fd_ = false;
  ok_ = false;
  path_.clear();
}

bool ExternalVectorStore::map_region() {
  unmap_region();
  if (fd_ < 0 || stride_bytes_ == 0) {
    return false;
  }
#if defined(_WIN32)
  HANDLE hf = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
  if (hf == INVALID_HANDLE_VALUE) {
    return false;
  }
  LARGE_INTEGER li{};
  if (!GetFileSizeEx(hf, &li)) {
    return false;
  }
  const uint64_t need = base_offset_ + stride_bytes_;
  if (static_cast<uint64_t>(li.QuadPart) < need) {
    return false;
  }
  HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!hm) {
    return false;
  }
  void *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
  if (!p) {
    CloseHandle(hm);
    return false;
  }
  map_handle_ = hm;
  map_base_ = p;
  map_size_ = static_cast<uint64_t>(li.QuadPart);
#else
  off_t end = lseek(fd_, 0, SEEK_END);
  if (end < 0) {
    return false;
  }
  const uint64_t need = base_offset_ + stride_bytes_;
  if (static_cast<uint64_t>(end) < need) {
    return false;
  }
  map_size_ = static_cast<uint64_t>(end);
  map_base_ = mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (map_base_ == MAP_FAILED) {
    map_base_ = nullptr;
    return false;
  }
#endif
  return true;
}

bool ExternalVectorStore::attach(int index_fd, const IndexFileHeader &header,
                                 const std::string &path_for_mmap) {
  detach();
  if (!header_uses_external_vectors(header)) {
    return false;
  }
  fd_ = index_fd;
  owns_fd_ = false;
  path_ = path_for_mmap;
  dim_ = header.dim;
  encoding_ = static_cast<VectorEncoding>(header.vector_encoding);
  base_offset_ = header.vector_section_offset;
  stride_bytes_ = header.vector_stride_bytes;
  if (!map_region()) {
    detach();
    return false;
  }
  ok_ = true;
  return true;
}

const uint8_t *ExternalVectorStore::payload(uint64_t node_id) const {
  if (!ok_ || !map_base_) {
    return nullptr;
  }
  const uint64_t off = base_offset_ + node_id * stride_bytes_;
  if (off + stride_bytes_ > map_size_) {
    return nullptr;
  }
  return static_cast<const uint8_t *>(map_base_) + off;
}

bool ExternalVectorStore::read_float_vector(uint64_t node_id, std::vector<float> *out) const {
  if (!out || dim_ == 0) {
    return false;
  }
  const uint8_t *p = payload(node_id);
  if (!p) {
    return false;
  }
  out->resize(dim_);
  if (encoding_ == VectorEncoding::Float32) {
    std::memcpy(out->data(), p, dim_ * sizeof(float));
    return true;
  }
  for (uint32_t i = 0; i < dim_; i++) {
    (*out)[i] = static_cast<float>(p[i]);
  }
  return true;
}

float ExternalVectorStore::l2_sq(const std::vector<float> &query, uint64_t node_id) const {
  if (query.size() != dim_) {
    return 1e30f;
  }
  const uint8_t *p = payload(node_id);
  if (!p) {
    return 1e30f;
  }
  if (encoding_ == VectorEncoding::Float32) {
    return amio::util::l2_sq_f32(query.data(), reinterpret_cast<const float *>(p), dim_);
  }
  float s = 0.0f;
  for (uint32_t i = 0; i < dim_; i++) {
    const float d = query[i] - static_cast<float>(p[i]);
    s += d * d;
  }
  return s;
}

bool ExternalVectorStore::write_vector_section(
    int fd, const IndexFileHeader &h_in, uint64_t total_nodes, VectorEncoding enc,
    const std::function<bool(uint64_t, std::vector<float> *)> &fill_float) {
  if (fd < 0 || total_nodes == 0 || h_in.dim == 0) {
    return false;
  }
  const uint64_t stride =
      enc == VectorEncoding::UInt8 ? static_cast<uint64_t>(h_in.dim)
                                   : static_cast<uint64_t>(h_in.dim) * sizeof(float);
  const uint64_t base = default_vector_section_offset(total_nodes);

  std::vector<float> row;
  row.reserve(h_in.dim);
  std::vector<uint8_t> row_u8;
  if (enc == VectorEncoding::UInt8) {
    row_u8.resize(h_in.dim);
  }

  for (uint64_t id = 0; id < total_nodes; id++) {
    if (!fill_float(id, &row)) {
      return false;
    }
    const uint64_t off = base + id * stride;
#if defined(_WIN32)
    if (_lseeki64(fd, static_cast<__int64>(off), SEEK_SET) < 0) {
      return false;
    }
    if (enc == VectorEncoding::UInt8) {
      for (uint32_t i = 0; i < h_in.dim; i++) {
        row_u8[i] = static_cast<uint8_t>(row[i]);
      }
      if (_write(fd, reinterpret_cast<const char *>(row_u8.data()),
                 static_cast<unsigned int>(h_in.dim)) !=
          static_cast<int>(h_in.dim)) {
        return false;
      }
    } else {
      if (_write(fd, reinterpret_cast<const char *>(row.data()),
                 static_cast<unsigned int>(h_in.dim * sizeof(float))) !=
          static_cast<int>(h_in.dim * sizeof(float))) {
        return false;
      }
    }
#else
    if (enc == VectorEncoding::UInt8) {
      for (uint32_t i = 0; i < h_in.dim; i++) {
        row_u8[i] = static_cast<uint8_t>(row[i]);
      }
      if (pwrite(fd, row_u8.data(), h_in.dim, static_cast<off_t>(off)) !=
          static_cast<ssize_t>(h_in.dim)) {
        return false;
      }
    } else {
      const size_t nbytes = static_cast<size_t>(h_in.dim) * sizeof(float);
      if (pwrite(fd, row.data(), nbytes, static_cast<off_t>(off)) !=
          static_cast<ssize_t>(nbytes)) {
        return false;
      }
    }
#endif
  }

  IndexFileHeader h = h_in;
  h.version = kIndexVersion2;
  h.vector_section_offset = base;
  h.vector_stride_bytes = static_cast<uint32_t>(stride);
  h.vector_encoding = static_cast<uint8_t>(enc);

#if defined(_WIN32)
  if (_lseeki64(fd, 0, SEEK_SET) < 0) {
    return false;
  }
  return _write(fd, reinterpret_cast<const char *>(&h), static_cast<unsigned int>(sizeof(h))) ==
         static_cast<int>(sizeof(h));
#else
  return pwrite(fd, &h, sizeof(h), 0) == static_cast<ssize_t>(sizeof(h));
#endif
}

} // namespace amio::index
