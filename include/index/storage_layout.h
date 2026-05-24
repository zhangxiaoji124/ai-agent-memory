#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace amio::index {

constexpr size_t kBlockSize = 4096;
constexpr uint64_t kMagic = 0x5844494D45474D41ULL; // "AGMEMIDX" (little-endian-ish)
constexpr uint32_t kIndexVersion1 = 1;           // 向量内联 NodeBlock（dim≤128）
constexpr uint32_t kIndexVersion2 = 2;           // 向量外置连续区（任意 dim，如 960）
constexpr uint32_t kMaxInlineVectorDim = 128;

#pragma pack(push, 1)
struct alignas(4096) IndexFileHeader {
  uint64_t magic = kMagic;
  uint32_t version = 1;
  uint64_t total_nodes = 0;
  uint8_t max_layer = 0;
  uint8_t _pad0[7]{};
  uint32_t ef_construction = 200;
  uint32_t m = 16;
  uint32_t dim = 128;
  uint64_t entry_point = 0;
  uint64_t high_layer_end_offset = kBlockSize;
  /// v2：向量区在文件中的起始偏移；0 表示 v1 内联于 NodeBlock。
  uint64_t vector_section_offset = 0;
  /// 每条向量字节数（float32: dim*4；uint8: dim）。
  uint32_t vector_stride_bytes = 0;
  /// 0=float32, 1=uint8（见 VectorEncoding）。
  uint8_t vector_encoding = 0;
  uint8_t _pad_hdr[3]{};
  uint8_t _reserved[4024]{};
};
static_assert(sizeof(IndexFileHeader) == kBlockSize);

struct alignas(4096) NodeBlock {
  uint64_t node_id = 0;
  uint8_t layer = 0;
  uint8_t neighbor_counts[8]{};
  uint16_t vec_dim = 0;
  uint8_t _reserved_header[45]{};
  float vector[128]{};
  uint32_t neighbors[8][32]{};
  uint64_t neighbor_offsets[8][32]{};
  uint8_t _padding[448]{};
};
static_assert(sizeof(NodeBlock) == kBlockSize);
#pragma pack(pop)

inline uint64_t node_offset(uint64_t node_id) {
  return static_cast<uint64_t>(kBlockSize) +
         node_id * static_cast<uint64_t>(kBlockSize);
}

class IndexFile {
public:
  IndexFile() = default;
  ~IndexFile();
  IndexFile(const IndexFile &) = delete;
  IndexFile &operator=(const IndexFile &) = delete;

  bool open_readonly(const std::string &path);
  /// 读写打开已有索引（compaction 增量写回）。
  bool open_readwrite(const std::string &path);
  bool open_create_trunc(const std::string &path);
  void close();

  bool read_header(IndexFileHeader *out) const;
  bool write_header(const IndexFileHeader &h);

  bool pread_node(uint64_t node_id, NodeBlock *out) const;
  bool pwrite_node(uint64_t node_id, const NodeBlock &b);

  int fd() const { return fd_; }

private:
  int fd_ = -1;
};

} // namespace amio::index
