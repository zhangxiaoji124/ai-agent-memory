#include "dataset/index_builder.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "dataset/index_build_common.h"
#include "dataset/vector_dataset.h"
#include "index/external_vectors.h"
#include "index/hnsw.h"
#include "index/storage_layout.h"

namespace amio::dataset {

bool build_disk_index_streaming(const std::string &vector_path, const std::string &output,
                              uint32_t m, uint32_t ef_construction, uint64_t max_vectors,
                              uint64_t batch_vectors) {
  VectorDataset ds;
  if (!ds.open(vector_path)) {
    std::fprintf(stderr, "build_disk_index_streaming: open failed %s\n", vector_path.c_str());
    return false;
  }
  const uint64_t total = std::min(max_vectors > 0 ? max_vectors : ds.size(), ds.size());
  if (total == 0) {
    return false;
  }
  const uint32_t dim = static_cast<uint32_t>(ds.dim());
  if (dim == 0) {
    return false;
  }

  index::HnswIndex idx(dim, m, ef_construction, 42);
  const uint64_t batch = batch_vectors > 0 ? batch_vectors : 500000;

  for (uint64_t begin = 0; begin < total; begin += batch) {
    const uint64_t cnt = std::min(batch, total - begin);
    if (!ds.iterate_batch(begin, cnt,
                          [&idx](uint64_t id, const std::vector<float> &v) {
                            idx.insert(id, v);
                            return true;
                          })) {
      return false;
    }
    std::fprintf(stderr, "  indexed %llu / %llu\r", static_cast<unsigned long long>(begin + cnt),
                 static_cast<unsigned long long>(total));
  }
  std::fprintf(stderr, "\n");

  index::IndexFile f;
  if (!f.open_create_trunc(output)) {
    return false;
  }

  index::IndexFileHeader h{};
  h.version = dim > index::kMaxInlineVectorDim ? index::kIndexVersion2 : index::kIndexVersion1;
  h.total_nodes = total;
  h.dim = dim;
  h.entry_point = idx.entry_point();
  h.m = m;
  h.ef_construction = ef_construction;
  h.max_layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.max_layer())));
  if (!f.write_header(h)) {
    return false;
  }

  if (!write_graph_nodes(f, idx, dim, total)) {
    return false;
  }

  const index::VectorEncoding enc =
      ds.encoding() == VectorEncoding::UInt8 ? index::VectorEncoding::UInt8
                                             : index::VectorEncoding::Float32;

  if (!finalize_index_vectors(
          f, &h, dim, enc,
          [&ds](uint64_t id, std::vector<float> *out) { return ds.get_float(id, out); })) {
    return false;
  }

  std::fprintf(stderr, "index v%u dim=%u external=%d path=%s\n", h.version, dim,
               index::header_uses_external_vectors(h) ? 1 : 0, output.c_str());
  return true;
}

} // namespace amio::dataset
