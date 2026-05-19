#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "runtime/memory_partition.h"

namespace amio::dataset {

enum class VectorEncoding { Float32, UInt8 };

struct VectorRecordView {
  const void *payload = nullptr;
  uint32_t dim = 0;
  VectorEncoding enc = VectorEncoding::Float32;
};

/// 统一向量文件访问：fvecs / bvecs（大文件 mmap，小文件可内存映射或缓冲）。
class VectorDataset {
public:
  VectorDataset() = default;
  ~VectorDataset();

  VectorDataset(const VectorDataset &) = delete;
  VectorDataset &operator=(const VectorDataset &) = delete;

  bool open(const std::string &path, runtime::VectorFileKind kind = runtime::VectorFileKind::Unknown);

  void close();

  bool ok() const { return ok_; }

  runtime::VectorFileKind kind() const { return kind_; }
  VectorEncoding encoding() const {
    return kind_ == runtime::VectorFileKind::Bvecs ? VectorEncoding::UInt8
                                                   : VectorEncoding::Float32;
  }

  uint32_t dim() const { return dim_; }
  uint64_t size() const { return num_vectors_; }
  uint64_t record_bytes() const { return record_bytes_; }

  /// 将第 i 条解码为 float（bvec 做 uint8→float 提升）。
  bool get_float(uint64_t i, std::vector<float> *out) const;

  bool get_view(uint64_t i, VectorRecordView *out) const;

  /// 批量回调；返回 false 则中止。
  bool iterate_batch(uint64_t begin, uint64_t count,
                     const std::function<bool(uint64_t, const std::vector<float> &)> &fn) const;

  /// 载入至多 max_vectors 条到内存（仅适合子集评测）。
  static bool load_subset_float(const std::string &path, uint64_t max_vectors,
                                std::vector<std::vector<float>> *out);

private:
  bool ok_ = false;
  runtime::VectorFileKind kind_ = runtime::VectorFileKind::Unknown;
  uint32_t dim_ = 0;
  uint64_t num_vectors_ = 0;
  uint64_t record_bytes_ = 0;
  std::string path_;

  void *map_base_ = nullptr;
  uint64_t map_size_ = 0;
#if defined(_WIN32)
  void *file_handle_ = nullptr;
  void *map_handle_ = nullptr;
#else
  int fd_ = -1;
#endif

  bool map_file();
  void unmap_file();

  const uint8_t *record_ptr(uint64_t i) const;
};

} // namespace amio::dataset
