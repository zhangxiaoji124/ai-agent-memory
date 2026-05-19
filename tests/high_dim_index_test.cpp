#include "dataset/index_build_common.h"
#include "index/external_vectors.h"
#include "index/hnsw.h"
#include "index/storage_layout.h"
#include "test.h"
#include "vector_store.h"

#include <cmath>
#include <cstdio>
#include <vector>

TEST(high_dim_960_index_roundtrip) {
  constexpr uint32_t dim = 960;
  constexpr int n = 24;
  std::vector<std::vector<float>> vecs(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    vecs[static_cast<size_t>(i)].resize(dim);
    for (uint32_t d = 0; d < dim; d++) {
      vecs[static_cast<size_t>(i)][d] = static_cast<float>(i) * 0.01f + d * 0.001f;
    }
  }

  amio::index::HnswIndex idx(dim, 8, 40, 99);
  for (uint64_t id = 0; id < static_cast<uint64_t>(n); id++) {
    idx.insert(id, vecs[static_cast<size_t>(id)]);
  }

  const std::string path = "data/test_highdim960.index";
  std::remove(path.c_str());

  amio::index::IndexFile f;
  EXPECT_TRUE(f.open_create_trunc(path));

  amio::index::IndexFileHeader h{};
  h.version = amio::index::kIndexVersion2;
  h.total_nodes = static_cast<uint64_t>(n);
  h.dim = dim;
  h.entry_point = idx.entry_point();
  h.m = 8;
  h.ef_construction = 40;
  h.max_layer = static_cast<uint8_t>(idx.max_layer());
  EXPECT_TRUE(f.write_header(h));
  EXPECT_TRUE(amio::dataset::write_graph_nodes(f, idx, dim, h.total_nodes));

  const size_t nn = vecs.size();
  EXPECT_TRUE(amio::dataset::finalize_index_vectors(
      f, &h, dim, amio::index::VectorEncoding::Float32,
      [&vecs, nn](uint64_t id, std::vector<float> *out) {
        if (id >= nn) {
          return false;
        }
        *out = vecs[static_cast<size_t>(id)];
        return true;
      }));
  f.close();

  amio::Config cfg;
  cfg.index_path = path;
  cfg.cache_size_mb = 64;
  cfg.static_cache_mb = 16;
  cfg.enable_wal = false;
  cfg.ef_search = 48;
  amio::VectorStore store(cfg);
  EXPECT_TRUE(store.open());
  EXPECT_TRUE(store.uses_external_vectors());

  auto res = store.search_disk(vecs[0], 3);
  EXPECT_TRUE(!res.empty());
  std::remove(path.c_str());
}
