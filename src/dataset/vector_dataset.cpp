#include "dataset/vector_dataset.h"

#include <cstring>
#include <fstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace amio::dataset {

VectorDataset::~VectorDataset() { close(); }

void VectorDataset::unmap_file() {
#if defined(_WIN32)
  if (map_base_) {
    UnmapViewOfFile(map_base_);
    map_base_ = nullptr;
  }
  if (map_handle_) {
    CloseHandle(static_cast<HANDLE>(map_handle_));
    map_handle_ = nullptr;
  }
  if (file_handle_) {
    CloseHandle(static_cast<HANDLE>(file_handle_));
    file_handle_ = nullptr;
  }
#else
  if (map_base_ && map_base_ != MAP_FAILED) {
    munmap(map_base_, map_size_);
  }
  map_base_ = nullptr;
  map_size_ = 0;
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
#endif
}

bool VectorDataset::map_file() {
  unmap_file();
#if defined(_WIN32)
  HANDLE hf =
      CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hf == INVALID_HANDLE_VALUE) {
    return false;
  }
  LARGE_INTEGER li{};
  if (!GetFileSizeEx(hf, &li)) {
    CloseHandle(hf);
    return false;
  }
  map_size_ = static_cast<uint64_t>(li.QuadPart);
  if (map_size_ == 0) {
    CloseHandle(hf);
    return false;
  }
  HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!hm) {
    CloseHandle(hf);
    return false;
  }
  void *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
  if (!p) {
    CloseHandle(hm);
    CloseHandle(hf);
    return false;
  }
  file_handle_ = hf;
  map_handle_ = hm;
  map_base_ = p;
#else
  fd_ = open(path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    return false;
  }
  struct stat st {};
  if (fstat(fd_, &st) != 0 || st.st_size <= 0) {
    close(fd_);
    fd_ = -1;
    return false;
  }
  map_size_ = static_cast<uint64_t>(st.st_size);
  map_base_ = mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (map_base_ == MAP_FAILED) {
    map_base_ = nullptr;
    close(fd_);
    fd_ = -1;
    return false;
  }
#endif
  return true;
}

bool VectorDataset::open(const std::string &path, runtime::VectorFileKind kind) {
  close();
  path_ = path;
  kind_ = kind;
  if (kind_ == runtime::VectorFileKind::Unknown) {
    kind_ = runtime::detect_kind_by_extension(path);
  }
  if (kind_ == runtime::VectorFileKind::Ivecs || kind_ == runtime::VectorFileKind::Unknown) {
    kind_ = runtime::VectorFileKind::Fvecs;
  }

  uint32_t d = 0;
  uint64_t n = 0;
  if (!runtime::probe_vector_file(path_, kind_, &d, &n)) {
    return false;
  }
  dim_ = d;
  num_vectors_ = n;
  record_bytes_ = 4 + ((kind_ == runtime::VectorFileKind::Bvecs) ? dim_ : dim_ * 4u);

  if (!map_file()) {
    return false;
  }
  ok_ = true;
  return true;
}

void VectorDataset::close() {
  unmap_file();
  ok_ = false;
  dim_ = 0;
  num_vectors_ = 0;
  record_bytes_ = 0;
  path_.clear();
}

const uint8_t *VectorDataset::record_ptr(uint64_t i) const {
  if (!ok_ || !map_base_ || i >= num_vectors_) {
    return nullptr;
  }
  const auto off = i * record_bytes_;
  if (off + record_bytes_ > map_size_) {
    return nullptr;
  }
  return static_cast<const uint8_t *>(map_base_) + off;
}

bool VectorDataset::get_view(uint64_t i, VectorRecordView *out) const {
  if (!out) {
    return false;
  }
  const uint8_t *rec = record_ptr(i);
  if (!rec) {
    return false;
  }
  uint32_t d = 0;
  std::memcpy(&d, rec, 4);
  if (d != dim_) {
    return false;
  }
  out->dim = dim_;
  out->payload = rec + 4;
  out->enc = encoding();
  return true;
}

bool VectorDataset::get_float(uint64_t i, std::vector<float> *out) const {
  if (!out) {
    return false;
  }
  VectorRecordView v{};
  if (!get_view(i, &v)) {
    return false;
  }
  out->resize(dim_);
  if (v.enc == VectorEncoding::Float32) {
    std::memcpy(out->data(), v.payload, dim_ * sizeof(float));
    return true;
  }
  const auto *u = static_cast<const uint8_t *>(v.payload);
  for (uint32_t j = 0; j < dim_; j++) {
    (*out)[j] = static_cast<float>(u[j]);
  }
  return true;
}

bool VectorDataset::iterate_batch(
    uint64_t begin, uint64_t count,
    const std::function<bool(uint64_t, const std::vector<float> &)> &fn) const {
  if (!fn || !ok_) {
    return false;
  }
  const uint64_t end = std::min(num_vectors_, begin + count);
  std::vector<float> buf;
  for (uint64_t i = begin; i < end; i++) {
    if (!get_float(i, &buf)) {
      return false;
    }
    if (!fn(i, buf)) {
      return false;
    }
  }
  return true;
}

bool VectorDataset::load_subset_float(const std::string &path, uint64_t max_vectors,
                                      std::vector<std::vector<float>> *out) {
  if (!out) {
    return false;
  }
  VectorDataset ds;
  if (!ds.open(path)) {
    return false;
  }
  const uint64_t n = std::min(max_vectors, ds.size());
  out->clear();
  out->reserve(static_cast<size_t>(n));
  std::vector<float> row;
  for (uint64_t i = 0; i < n; i++) {
    if (!ds.get_float(i, &row)) {
      return false;
    }
    out->push_back(row);
  }
  return true;
}

} // namespace amio::dataset
