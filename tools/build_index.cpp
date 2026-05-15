#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "dataset/fvecs.h"
#include "index/hnsw.h"
#include "index/storage_layout.h"

static const char *arg_value(int argc, char **argv, const char *key) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::strcmp(argv[i], key) == 0)
      return argv[i + 1];
  }
  return nullptr;
}

int main(int argc, char **argv) {
  const char *input = arg_value(argc, argv, "--input");
  const char *output = arg_value(argc, argv, "--output");
  const char *m_arg = arg_value(argc, argv, "--m");
  const char *ef_arg = arg_value(argc, argv, "--ef-construction");
  if (!input || !output) {
    std::fprintf(stderr,
                 "用法: build_index --input <base.fvecs> --output <out.index> "
                 "[--m 16] [--ef-construction 200]\n");
    return 1;
  }

  std::vector<std::vector<float>> vecs;
  if (!amio::dataset::load_fvecs(input, &vecs) || vecs.empty()) {
    std::fprintf(stderr, "读取 fvecs 失败: %s\n", input);
    return 2;
  }

  const uint32_t dim = static_cast<uint32_t>(vecs[0].size());
  const uint32_t M = m_arg ? static_cast<uint32_t>(std::atoi(m_arg)) : 16u;
  const uint32_t ef = ef_arg ? static_cast<uint32_t>(std::atoi(ef_arg)) : 200u;

  std::fprintf(stderr, "build_index: nodes=%zu dim=%u m=%u ef=%u (HNSW)\n", vecs.size(), dim,
               M, ef);

  amio::index::HnswIndex idx(dim, M, ef, 42);
  for (uint64_t id = 0; id < static_cast<uint64_t>(vecs.size()); id++) {
    idx.insert(id, vecs[static_cast<size_t>(id)]);
  }

  amio::index::IndexFile f;
  if (!f.open_create_trunc(output)) {
    std::fprintf(stderr, "打开输出失败: %s\n", output);
    return 3;
  }

  amio::index::IndexFileHeader h{};
  h.total_nodes = static_cast<uint64_t>(vecs.size());
  h.dim = dim;
  h.entry_point = idx.entry_point();
  h.m = M;
  h.ef_construction = ef;
  h.max_layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.max_layer())));
  f.write_header(h);

  for (uint64_t id = 0; id < static_cast<uint64_t>(vecs.size()); id++) {
    amio::index::NodeBlock b{};
    b.node_id = id;
    b.layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.node_max_layer(id))));
    b.vec_dim = static_cast<uint16_t>(dim);
    for (uint32_t i = 0; i < dim && i < 128; i++)
      b.vector[i] = vecs[static_cast<size_t>(id)][i];

    for (int l = 0; l < 8; l++) {
      const auto &nbl = idx.neighbors_at(l, id);
      const size_t cnt = std::min<size_t>(32, nbl.size());
      b.neighbor_counts[static_cast<size_t>(l)] = static_cast<uint8_t>(cnt);
      for (size_t j = 0; j < cnt; j++) {
        const uint64_t nid = nbl[j];
        b.neighbors[static_cast<size_t>(l)][j] = static_cast<uint32_t>(nid);
        b.neighbor_offsets[static_cast<size_t>(l)][j] = amio::index::node_offset(nid);
      }
    }
    if (!f.pwrite_node(id, b)) {
      std::fprintf(stderr, "写入节点失败: %llu\n",
                   static_cast<unsigned long long>(id));
      return 4;
    }
  }

  f.close();
  std::fprintf(stderr, "已生成索引: %s (nodes=%zu, dim=%u, max_layer=%u)\n", output,
               vecs.size(), dim, static_cast<unsigned>(h.max_layer));
  return 0;
}
