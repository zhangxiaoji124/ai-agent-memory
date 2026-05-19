#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "dataset/fvecs.h"
#include "dataset/index_build_common.h"
#include "dataset/index_builder.h"
#include "dataset/vector_dataset.h"
#include "index/external_vectors.h"
#include "index/hnsw.h"
#include "index/storage_layout.h"
#include "runtime/memory_partition.h"

static const char *arg_value(int argc, char **argv, const char *key) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::strcmp(argv[i], key) == 0)
      return argv[i + 1];
  }
  return nullptr;
}

static size_t arg_size_t(int argc, char **argv, const char *key, size_t def) {
  const char *v = arg_value(argc, argv, key);
  if (!v)
    return def;
  return static_cast<size_t>(std::strtoull(v, nullptr, 10));
}

int main(int argc, char **argv) {
  const char *input = arg_value(argc, argv, "--input");
  const char *output = arg_value(argc, argv, "--output");
  const char *m_arg = arg_value(argc, argv, "--m");
  const char *ef_arg = arg_value(argc, argv, "--ef-construction");
  const size_t max_vectors = arg_size_t(argc, argv, "--max-vectors", 0);
  const size_t ram_budget_gb = arg_size_t(argc, argv, "--ram-budget-gb", 0);
  if (!input || !output) {
    std::fprintf(stderr,
                 "用法: build_index --input <base.fvecs|bvecs> --output <out.index> "
                 "[--m 16] [--ef-construction 200] [--max-vectors N] "
                 "[--ram-budget-gb 100]\n"
                 "  支持任意维度（如 GIST 960）；dim>128 自动使用 v2 外置向量区。\n");
    return 1;
  }

  uint64_t ram_bytes = ram_budget_gb > 0 ? ram_budget_gb * 1024ull * 1024 * 1024 : 0;
  ram_bytes = amio::runtime::resolve_ram_budget_bytes(ram_bytes);
  const auto part = amio::runtime::select_memory_partition({input}, ram_bytes);
  std::fprintf(stderr, "build_index: partition=%s kind=%s dim=%u\n",
               amio::runtime::profile_name(part.profile),
               amio::runtime::vector_kind_name(part.dataset.kind), part.dataset.dim);

  const uint32_t M = m_arg ? static_cast<uint32_t>(std::atoi(m_arg)) : 16u;
  const uint32_t ef = ef_arg ? static_cast<uint32_t>(std::atoi(ef_arg)) : 200u;
  const uint64_t maxv =
      max_vectors > 0 ? max_vectors
                      : (part.dataset.num_vectors_est > 0 ? part.dataset.num_vectors_est : 0);

  const bool use_stream =
      part.dataset.kind == amio::runtime::VectorFileKind::Bvecs ||
      part.dataset.data_bytes > ram_bytes / 2 ||
      part.dataset.dim > amio::index::kMaxInlineVectorDim;

  if (use_stream) {
    std::fprintf(stderr, "build_index: streaming dim=%u max_vectors=%llu batch=%llu\n",
                 part.dataset.dim, static_cast<unsigned long long>(maxv),
                 static_cast<unsigned long long>(part.stream_index_batch_vectors));
    if (!amio::dataset::build_disk_index_streaming(input, output, M, ef, maxv,
                                                   part.stream_index_batch_vectors)) {
      std::fprintf(stderr, "流式建索引失败\n");
      return 2;
    }
    std::fprintf(stderr, "已生成索引(流式): %s\n", output);
    return 0;
  }

  std::vector<std::vector<float>> vecs;
  if (!amio::dataset::load_fvecs(input, &vecs) || vecs.empty()) {
    std::fprintf(stderr, "读取向量失败: %s\n", input);
    return 2;
  }
  if (maxv > 0 && vecs.size() > maxv) {
    vecs.resize(static_cast<size_t>(maxv));
  }

  const uint32_t dim = static_cast<uint32_t>(vecs[0].size());
  std::fprintf(stderr, "build_index: nodes=%zu dim=%u m=%u ef=%u\n", vecs.size(), dim, M,
               ef);

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
  h.version = dim > amio::index::kMaxInlineVectorDim ? amio::index::kIndexVersion2
                                                     : amio::index::kIndexVersion1;
  h.total_nodes = static_cast<uint64_t>(vecs.size());
  h.dim = dim;
  h.entry_point = idx.entry_point();
  h.m = M;
  h.ef_construction = ef;
  h.max_layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.max_layer())));
  if (!f.write_header(h)) {
    return 4;
  }
  if (!amio::dataset::write_graph_nodes(f, idx, dim, h.total_nodes)) {
    return 4;
  }
  const size_t n = vecs.size();
  if (!amio::dataset::finalize_index_vectors(
          f, &h, dim, amio::index::VectorEncoding::Float32,
          [&vecs, n](uint64_t id, std::vector<float> *out) {
            if (id >= n) {
              return false;
            }
            *out = vecs[static_cast<size_t>(id)];
            return true;
          })) {
    return 4;
  }

  f.close();
  std::fprintf(stderr, "已生成索引: %s (nodes=%zu, dim=%u, v%u external=%d)\n", output,
               vecs.size(), dim, h.version,
               amio::index::header_uses_external_vectors(h) ? 1 : 0);
  return 0;
}
