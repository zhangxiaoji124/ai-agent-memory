#include <cstdio>
#include <random>
#include <vector>

#include "util/time.h"
#include "vector_store.h"

int main() {
  amio::Config cfg;
  cfg.enable_wal = false;
  cfg.memtable_limit_mb = 8;
  amio::VectorStore store(cfg);

  const size_t dim = 64;
  const size_t n = 100000;
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  const uint64_t t0 = amio::util::now_ns();
  for (size_t i = 0; i < n; i++) {
    std::vector<float> v(dim);
    for (size_t d = 0; d < dim; d++)
      v[d] = dist(rng);
    store.insert(static_cast<uint64_t>(i), v);
  }
  const uint64_t t1 = amio::util::now_ns();
  const double ms = (t1 - t0) / 1e6;
  std::printf("bench_write: n=%zu total_ms=%.3f vec_per_s=%.1f\n", n, ms,
              (n / (ms / 1000.0)));
  return 0;
}
