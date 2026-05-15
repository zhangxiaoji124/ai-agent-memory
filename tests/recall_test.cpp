#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "dataset/fvecs.h"
#include "test.h"
#include "vector_store.h"

static float l2_sq(const std::vector<float> &a, const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  float s = 0.0f;
  for (size_t i = 0; i < n; i++) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

TEST(recall_synthetic_sanity) {
  const size_t dim = 32;
  const size_t n = 512;
  std::vector<std::vector<float>> base;
  base.reserve(n);
  for (size_t i = 0; i < n; i++) {
    std::vector<float> v(dim, 0.0f);
    v[0] = static_cast<float>(i) * 0.01f;
    v[1] = static_cast<float>(i % 7);
    base.push_back(v);
  }

  amio::Config cfg;
  cfg.enable_wal = false;
  amio::VectorStore store(cfg);

  // 用 MemTable 填充并作为近似检索（骨架）。
  for (size_t i = 0; i < n; i++) {
    (void)store.insert(static_cast<uint64_t>(i), base[i]);
  }

  std::vector<std::vector<uint32_t>> results;
  std::vector<std::vector<uint32_t>> gt;
  const size_t nq = 50;
  for (size_t qi = 0; qi < nq; qi++) {
    auto r = store.search(base[qi], 10);
    std::vector<uint32_t> ids;
    for (auto &x : r)
      ids.push_back(static_cast<uint32_t>(x.id));
    results.push_back(std::move(ids));

    std::vector<std::pair<uint32_t, float>> scored;
    scored.reserve(base.size());
    for (size_t i = 0; i < base.size(); i++) {
      scored.push_back({static_cast<uint32_t>(i), l2_sq(base[qi], base[i])});
    }
    std::sort(scored.begin(), scored.end(),
              [](auto &a, auto &b) { return a.second < b.second; });
    std::vector<uint32_t> gid;
    for (size_t j = 0; j < 10; j++)
      gid.push_back(scored[j].first);
    gt.push_back(std::move(gid));
  }

  const double recall = amio::dataset::compute_recall_at_k(results, gt, 10);
  EXPECT_TRUE(recall >= 0.95); // MemTable 暴力应接近 1.0
}

TEST(recall_sift_subset_optional) {
  std::vector<std::vector<float>> base_all;
  std::vector<std::vector<float>> queries;
  if (!amio::dataset::load_fvecs("data/sift/sift_base.fvecs", &base_all) ||
      !amio::dataset::load_fvecs("data/sift/sift_query.fvecs", &queries)) {
    // 未准备数据集则跳过（不失败）。
    return;
  }
  const size_t n = std::min<size_t>(8000, base_all.size());
  const size_t nq = std::min<size_t>(100, queries.size());

  std::vector<std::vector<float>> base(base_all.begin(), base_all.begin() + n);
  const size_t dim = base[0].size();

  amio::Config cfg;
  cfg.enable_wal = false;
  amio::VectorStore store(cfg);
  for (size_t i = 0; i < base.size(); i++) {
    (void)store.insert(static_cast<uint64_t>(i), base[i]);
  }

  std::vector<std::vector<uint32_t>> results;
  std::vector<std::vector<uint32_t>> gt;
  results.reserve(nq);
  gt.reserve(nq);

  for (size_t qi = 0; qi < nq; qi++) {
    auto r = store.search(queries[qi], 10);
    std::vector<uint32_t> ids;
    for (auto &x : r)
      ids.push_back(static_cast<uint32_t>(x.id));
    results.push_back(std::move(ids));

    std::vector<std::pair<uint32_t, float>> scored;
    scored.reserve(base.size());
    for (size_t i = 0; i < base.size(); i++) {
      // 只取 dim 的有效部分
      scored.push_back({static_cast<uint32_t>(i), l2_sq(queries[qi], base[i])});
    }
    std::sort(scored.begin(), scored.end(),
              [](auto &a, auto &b) { return a.second < b.second; });
    std::vector<uint32_t> gid;
    for (size_t j = 0; j < 10; j++)
      gid.push_back(scored[j].first);
    gt.push_back(std::move(gid));
  }

  const double recall = amio::dataset::compute_recall_at_k(results, gt, 10);
  EXPECT_TRUE(recall >= 0.95);
  (void)dim;
}
