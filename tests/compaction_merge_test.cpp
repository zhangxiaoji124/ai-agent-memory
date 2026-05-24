#include "index/hnsw.h"
#include "index/storage_layout.h"
#include "test.h"
#include "vector_store.h"

#include <cmath>
#include <cstdio>
#include <vector>

TEST(compaction_merge_writes_index) {
  constexpr uint32_t dim = 32;
  constexpr int n0 = 16;
  std::vector<std::vector<float>> base(static_cast<size_t>(n0));
  for (int i = 0; i < n0; i++) {
    base[static_cast<size_t>(i)].resize(dim);
    for (uint32_t d = 0; d < dim; d++) {
      base[static_cast<size_t>(i)][d] = static_cast<float>(i) * 0.01f + d * 0.001f;
    }
  }

  amio::index::HnswIndex idx(dim, 8, 40, 7);
  for (uint64_t id = 0; id < static_cast<uint64_t>(n0); id++) {
    idx.insert(id, base[static_cast<size_t>(id)]);
  }

  const std::string path = "data/test_compaction.index";
  std::remove(path.c_str());

  amio::index::IndexFile f;
  EXPECT_TRUE(f.open_create_trunc(path));
  amio::index::IndexFileHeader h{};
  h.dim = dim;
  h.m = 8;
  h.ef_construction = 40;
  h.total_nodes = static_cast<uint64_t>(n0);
  h.entry_point = idx.entry_point();
  h.max_layer = static_cast<uint8_t>(idx.max_layer());
  EXPECT_TRUE(f.write_header(h));
  EXPECT_TRUE(amio::dataset::write_graph_nodes(f, idx, dim, h.total_nodes));
  f.close();

  amio::Config cfg;
  cfg.index_path = path;
  cfg.enable_wal = false;
  cfg.enable_nvtable = true;
  cfg.enable_noblsm = true;
  cfg.memtable_limit_mb = 1;
  cfg.cache_size_mb = 32;
  cfg.static_cache_mb = 8;
  amio::VectorStore store(cfg);
  EXPECT_TRUE(store.open());

  const int extra = 8;
  for (int i = 0; i < extra; i++) {
    std::vector<float> v(dim);
    for (uint32_t d = 0; d < dim; d++) {
      v[d] = static_cast<float>(n0 + i) * 0.02f + d * 0.001f;
    }
    EXPECT_TRUE(store.insert(static_cast<uint64_t>(n0 + i), v));
  }
  store.flush_writes();

  amio::index::IndexFile rf;
  EXPECT_TRUE(rf.open_readonly(path));
  amio::index::IndexFileHeader rh{};
  EXPECT_TRUE(rf.read_header(&rh));
  EXPECT_TRUE(rh.total_nodes >= static_cast<uint64_t>(n0 + extra));

  auto res = store.search_disk(base[0], 3);
  EXPECT_TRUE(!res.empty());
  std::remove(path.c_str());
}
