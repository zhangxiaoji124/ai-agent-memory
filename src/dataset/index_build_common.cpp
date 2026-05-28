#include "dataset/index_build_common.h"

#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include "index/external_vectors.h"

namespace amio::dataset {

index::NodeBlock make_node_block(const index::HnswIndex &idx, uint32_t dim, uint64_t node_id) {
  const uint64_t id_base = idx.id_base();
  const uint64_t local_id = node_id >= id_base ? node_id - id_base : node_id;
  index::NodeBlock b{};
  b.node_id = node_id;
  b.layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.node_max_layer(local_id))));
  b.vec_dim = dim > 65535u ? 65535u : static_cast<uint16_t>(dim);
  if (dim <= index::kMaxInlineVectorDim) {
    const auto &vec = idx.vector_at(local_id);
    for (uint32_t i = 0; i < dim && i < index::kMaxInlineVectorDim; i++) {
      b.vector[i] = vec[i];
    }
  }
  for (int l = 0; l < 8; l++) {
    const auto &nbl = idx.neighbors_at(l, local_id);
    const size_t cnt = std::min<size_t>(32, nbl.size());
    b.neighbor_counts[static_cast<size_t>(l)] = static_cast<uint8_t>(cnt);
    for (size_t j = 0; j < cnt; j++) {
      const uint64_t nid = id_base + nbl[j];
      b.neighbors[static_cast<size_t>(l)][j] = static_cast<uint32_t>(nid);
      // v3：邻居 offset 不再落盘，检索/预取时按 node_offset(nid) 推算。
    }
  }
  return b;
}

static bool append_external_vector_row(int fd, const index::IndexFileHeader &h, uint64_t node_id,
                                       const std::vector<float> &vec) {
  if (fd < 0 || !index::header_uses_external_vectors(h)) {
    return false;
  }
  const uint64_t stride = h.vector_stride_bytes;
  const uint64_t off = h.vector_section_offset + node_id * stride;
  const size_t nbytes =
      h.vector_encoding == static_cast<uint8_t>(index::VectorEncoding::UInt8)
          ? static_cast<size_t>(h.dim)
          : static_cast<size_t>(h.dim) * sizeof(float);

#if defined(_WIN32)
  if (_lseeki64(fd, static_cast<__int64>(off), SEEK_SET) < 0) {
    return false;
  }
  if (h.vector_encoding == static_cast<uint8_t>(index::VectorEncoding::UInt8)) {
    std::vector<uint8_t> row(h.dim);
    for (uint32_t i = 0; i < h.dim; i++) {
      row[i] = static_cast<uint8_t>(vec[i]);
    }
    return _write(fd, reinterpret_cast<const char *>(row.data()),
                  static_cast<unsigned int>(nbytes)) == static_cast<int>(nbytes);
  }
  return _write(fd, reinterpret_cast<const char *>(vec.data()),
                static_cast<unsigned int>(nbytes)) == static_cast<int>(nbytes);
#else
  if (h.vector_encoding == static_cast<uint8_t>(index::VectorEncoding::UInt8)) {
    std::vector<uint8_t> row(h.dim);
    for (uint32_t i = 0; i < h.dim; i++) {
      row[i] = static_cast<uint8_t>(vec[i]);
    }
    return ::pwrite(fd, row.data(), nbytes, static_cast<off_t>(off)) ==
           static_cast<ssize_t>(nbytes);
  }
  return ::pwrite(fd, vec.data(), nbytes, static_cast<off_t>(off)) ==
         static_cast<ssize_t>(nbytes);
#endif
}

bool write_single_node(index::IndexFile &f, index::IndexFileHeader *header,
                       const index::HnswIndex &idx, uint32_t dim, uint64_t node_id,
                       const std::vector<float> &vec) {
  if (!header || f.fd() < 0) {
    return false;
  }
  const uint64_t local_id = node_id >= idx.id_base() ? node_id - idx.id_base() : node_id;
  if (local_id >= idx.size()) {
    return false;
  }
  const index::NodeBlock b = make_node_block(idx, dim, node_id);
  if (!f.pwrite_node(node_id, b)) {
    return false;
  }
  if (dim > index::kMaxInlineVectorDim) {
    if (!append_external_vector_row(f.fd(), *header, node_id, vec)) {
      return false;
    }
  }
  return true;
}

bool write_graph_nodes(index::IndexFile &f, const index::HnswIndex &idx, uint32_t dim,
                       uint64_t total_nodes) {
  for (uint64_t id = 0; id < total_nodes; id++) {
    const index::NodeBlock b = make_node_block(idx, dim, id);
    if (!f.pwrite_node(id, b)) {
      return false;
    }
  }
  return true;
}

bool finalize_index_vectors(
    index::IndexFile &f, index::IndexFileHeader *h, uint32_t dim,
    index::VectorEncoding enc,
    const std::function<bool(uint64_t, std::vector<float> *)> &fill_float) {
  if (!h || !fill_float) {
    return false;
  }
  if (dim <= index::kMaxInlineVectorDim) {
    h->version = index::kIndexVersion1;
    h->vector_section_offset = 0;
    h->vector_stride_bytes = 0;
    return f.write_header(*h);
  }
  return index::ExternalVectorStore::write_vector_section(f.fd(), *h, h->total_nodes, enc,
                                                          fill_float);
}

} // namespace amio::dataset
