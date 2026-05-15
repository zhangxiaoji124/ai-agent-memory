#include <thread>
#include <vector>

#include "test.h"
#include "vector_store.h"

TEST(concurrent_insert_search_smoke) {
  amio::Config cfg;
  cfg.enable_wal = false;
  amio::VectorStore store(cfg);

  const size_t dim = 16;
  const size_t n = 2000;

  std::thread writer([&] {
    for (size_t i = 0; i < n; i++) {
      std::vector<float> v(dim, 0.0f);
      v[0] = static_cast<float>(i);
      (void)store.insert(static_cast<uint64_t>(i), v);
    }
  });

  std::vector<std::thread> readers;
  for (int t = 0; t < 4; t++) {
    readers.emplace_back([&] {
      for (int i = 0; i < 200; i++) {
        std::vector<float> q(dim, 0.0f);
        q[0] = static_cast<float>(i);
        auto r = store.search(q, 5);
        EXPECT_TRUE(!r.empty());
      }
    });
  }

  writer.join();
  for (auto &th : readers)
    th.join();
}
