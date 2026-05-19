#include "dataset/index_build_common.h"

#include "index/external_vectors.h"

namespace amio::dataset {

bool write_graph_nodes(index::IndexFile &f, const index::HnswIndex &idx, uint32_t dim,
                       uint64_t total_nodes) {
  const bool external = dim > index::kMaxInlineVectorDim;
  for (uint64_t id = 0; id < total_nodes; id++) {
    index::NodeBlock b{};
    b.node_id = id;
    b.layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.node_max_layer(id))));
    b.vec_dim = dim > 65535u ? 65535u : static_cast<uint16_t>(dim);
    if (!external) {
      const auto &vec = idx.vector_at(id);
      for (uint32_t i = 0; i < dim && i < index::kMaxInlineVectorDim; i++) {
        b.vector[i] = vec[i];
      }
    }
    for (int l = 0; l < 8; l++) {
      const auto &nbl = idx.neighbors_at(l, id);
      const size_t cnt = std::min<size_t>(32, nbl.size());
      b.neighbor_counts[static_cast<size_t>(l)] = static_cast<uint8_t>(cnt);
      for (size_t j = 0; j < cnt; j++) {
        const uint64_t nid = nbl[j];
        b.neighbors[static_cast<size_t>(l)][j] = static_cast<uint32_t>(nid);
        b.neighbor_offsets[static_cast<size_t>(l)][j] = index::node_offset(nid);
      }
    }
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
