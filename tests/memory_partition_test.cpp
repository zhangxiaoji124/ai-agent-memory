#include "runtime/memory_partition.h"
#include "test.h"

#include <cstdio>
#include <fstream>
#include <vector>

TEST(memory_partition_fvecs_small) {
  const std::string path = "data/tiny_part_test.fvecs";
  {
    std::ofstream ofs(path, std::ios::binary);
    const uint32_t dim = 8;
    for (int i = 0; i < 100; i++) {
      ofs.write(reinterpret_cast<const char *>(&dim), 4);
      std::vector<float> v(dim, static_cast<float>(i));
      ofs.write(reinterpret_cast<const char *>(v.data()), dim * 4);
    }
  }
  const uint64_t ram = 8ull * 1024 * 1024 * 1024;
  auto d = amio::runtime::select_memory_partition({path}, ram);
  EXPECT_TRUE(d.profile == amio::runtime::MemoryPartitionProfile::HostResident);
  EXPECT_TRUE(d.pools.dynamic_cache_bytes > d.pools.memtable_bytes);
  std::remove(path.c_str());
}

TEST(memory_partition_bvec_large) {
  const uint64_t ram = 100ull * 1024 * 1024 * 1024;
  const uint64_t data = 500ull * 1024 * 1024 * 1024;
  amio::runtime::DatasetDescriptor desc{};
  desc.kind = amio::runtime::VectorFileKind::Bvecs;
  desc.data_bytes = data;
  desc.dim = 128;
  desc.num_vectors_est = data / (4 + 128);
  desc.rho = static_cast<double>(data) / static_cast<double>(ram);

  auto d = amio::runtime::select_memory_partition({}, ram,
                                                  amio::runtime::MemoryPartitionProfile::BvecUltra500G100G);
  EXPECT_TRUE(d.profile == amio::runtime::MemoryPartitionProfile::BvecUltra500G100G);
  EXPECT_TRUE(std::string(d.distance_kernel) == "u8_l2");
  (void)desc;
}
