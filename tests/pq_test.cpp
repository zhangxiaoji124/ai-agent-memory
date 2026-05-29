#include "index/pq.h"
#include "test.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

TEST(pq_train_encode_adc_recall) {
  const uint32_t dim = 64, m = 8, k = 10;
  const int n = 4000, nq = 100, NC = 40;
  std::mt19937_64 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.10f);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);

  std::vector<std::vector<float>> centers(static_cast<size_t>(NC), std::vector<float>(dim));
  for (auto &c : centers)
    for (auto &x : c)
      x = u(rng);
  auto gen = [&]() {
    std::vector<float> v(dim);
    const int ci = static_cast<int>(rng() % static_cast<uint64_t>(NC));
    for (uint32_t d = 0; d < dim; d++)
      v[d] = centers[static_cast<size_t>(ci)][d] + noise(rng);
    return v;
  };
  std::vector<std::vector<float>> base(static_cast<size_t>(n)), q(static_cast<size_t>(nq));
  for (auto &v : base)
    v = gen();
  for (auto &v : q)
    v = gen();

  amio::index::ProductQuantizer pq;
  EXPECT_TRUE(pq.train(base, dim, m, 25, static_cast<uint32_t>(n), 7));
  EXPECT_TRUE(pq.ok());

  std::vector<uint8_t> codes(static_cast<size_t>(n) * m);
  for (int i = 0; i < n; i++)
    pq.encode(base[static_cast<size_t>(i)].data(), codes.data() + static_cast<size_t>(i) * m);

  auto l2 = [&](const float *a, const float *b) {
    float s = 0.0f;
    for (uint32_t d = 0; d < dim; d++) {
      const float x = a[d] - b[d];
      s += x * x;
    }
    return s;
  };

  const int RN = 100; // 粗筛宽度（PQ 实际用法：ADC 取 top-RN 候选，再全精度重排）
  double rec_direct = 0.0, rec_coarse = 0.0;
  for (int qi = 0; qi < nq; qi++) {
    std::vector<std::pair<float, int>> ex(static_cast<size_t>(n));
    for (int i = 0; i < n; i++)
      ex[static_cast<size_t>(i)] = {l2(q[static_cast<size_t>(qi)].data(),
                                       base[static_cast<size_t>(i)].data()), i};
    std::partial_sort(ex.begin(), ex.begin() + k, ex.end());

    std::vector<float> lut;
    pq.compute_lut(q[static_cast<size_t>(qi)].data(), &lut);
    std::vector<std::pair<float, int>> ad(static_cast<size_t>(n));
    for (int i = 0; i < n; i++)
      ad[static_cast<size_t>(i)] = {pq.adc_distance(codes.data() + static_cast<size_t>(i) * m, lut), i};
    std::partial_sort(ad.begin(), ad.begin() + RN, ad.end());

    int hit_d = 0, hit_c = 0;
    for (uint32_t g = 0; g < k; g++) {
      for (uint32_t j = 0; j < k; j++)
        if (ad[j].second == ex[g].second) { hit_d++; break; }
      for (int j = 0; j < RN; j++)
        if (ad[static_cast<size_t>(j)].second == ex[g].second) { hit_c++; break; }
    }
    rec_direct += static_cast<double>(hit_d) / k;
    rec_coarse += static_cast<double>(hit_c) / k;
  }
  rec_direct /= nq;
  rec_coarse /= nq;
  std::fprintf(stderr,
               "[pq] adc recall@%u(direct)=%.3f  coarse top-%d recall=%.3f  (random ~%.4f)\n",
               k, rec_direct, RN, rec_coarse, static_cast<double>(k) / n);
  // PQ 实际用法是粗筛+重排：真 top-k 应高概率落入 ADC top-RN（远优于随机）。
  EXPECT_TRUE(rec_coarse > 0.80);
  EXPECT_TRUE(rec_direct > 0.20);

  // 码本 save/load 一致性
  const std::string path = "data/test_pq.bin";
  EXPECT_TRUE(pq.save(path));
  amio::index::ProductQuantizer pq2;
  EXPECT_TRUE(pq2.load(path));
  EXPECT_TRUE(pq2.dim() == dim && pq2.m() == m && pq2.sub_dim() == dim / m);
  std::vector<uint8_t> c1(m), c2(m);
  pq.encode(base[0].data(), c1.data());
  pq2.encode(base[0].data(), c2.data());
  EXPECT_TRUE(std::equal(c1.begin(), c1.end(), c2.begin()));
  std::remove(path.c_str());
}
