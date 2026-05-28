#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "index/storage_layout.h"

namespace amio::index {

/// 向量载荷：v1 内联于 NodeBlock；v2 起存于索引文件尾部连续区（支持 dim>128，如 960）。
enum class VectorEncoding : uint8_t {
  Float32 = 0,
  UInt8 = 1,
};

inline bool header_uses_external_vectors(const IndexFileHeader &h) {
  return h.version >= 2 && h.vector_section_offset != 0 && h.vector_stride_bytes > 0;
}

inline uint64_t default_vector_section_offset(uint64_t total_nodes) {
  return kHeaderSize + total_nodes * kBlockSize;
}

class ExternalVectorStore {
public:
  ExternalVectorStore() = default;
  ~ExternalVectorStore();

  ExternalVectorStore(const ExternalVectorStore &) = delete;
  ExternalVectorStore &operator=(const ExternalVectorStore &) = delete;

  /// 绑定已打开的索引 fd 与文件头（mmap 向量区）。
  bool attach(int index_fd, const IndexFileHeader &header, const std::string &path_for_mmap);

  void detach();

  bool ok() const { return ok_; }
  uint32_t dim() const { return dim_; }
  VectorEncoding encoding() const { return encoding_; }
  uint64_t stride_bytes() const { return stride_bytes_; }

  float l2_sq(const std::vector<float> &query, uint64_t node_id) const;

  bool read_float_vector(uint64_t node_id, std::vector<float> *out) const;

  /// 建索引时写入向量区并填充 header 中 vector_* 字段（dim 任意，如 960）。
  static bool write_vector_section(int fd, const IndexFileHeader &h_in, uint64_t total_nodes,
                                   VectorEncoding enc,
                                   const std::function<bool(uint64_t, std::vector<float> *)>
                                       &fill_float);

private:
  bool ok_ = false;
  uint32_t dim_ = 0;
  VectorEncoding encoding_ = VectorEncoding::Float32;
  uint64_t base_offset_ = 0;
  uint64_t stride_bytes_ = 0;
  uint64_t map_size_ = 0;

  void *map_base_ = nullptr;
  int fd_ = -1;
  bool owns_fd_ = false;
  std::string path_;

#if defined(_WIN32)
  void *map_handle_ = nullptr;
#endif

  bool map_region();
  void unmap_region();
  const uint8_t *payload(uint64_t node_id) const;
};

} // namespace amio::index
