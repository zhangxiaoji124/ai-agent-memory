#include <cstdio>
#include <string>
#include <vector>

#include "dataset/fvecs.h"
#include "dataset/vector_dataset.h"
#include "runtime/memory_partition.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "用法: dataset_loader <file.fvecs|file.bvecs|file.ivecs>\n");
    return 1;
  }
  const std::string path = argv[1];
  if (path.size() >= 6 && path.substr(path.size() - 6) == ".fvecs") {
    std::vector<std::vector<float>> v;
    if (!amio::dataset::load_fvecs(path, &v)) {
      std::fprintf(stderr, "读取失败: %s\n", path.c_str());
      return 2;
    }
    std::printf("fvecs: %zu 条, dim=%zu\n", v.size(),
                v.empty() ? 0 : v[0].size());
    return 0;
  }
  if (path.size() >= 6 && path.substr(path.size() - 6) == ".bvecs") {
    amio::dataset::VectorDataset ds;
    if (!ds.open(path)) {
      std::fprintf(stderr, "读取失败: %s\n", path.c_str());
      return 2;
    }
    std::printf("bvecs(mmap): %llu 条, dim=%u, record_bytes=%llu\n",
                static_cast<unsigned long long>(ds.size()), ds.dim(),
                static_cast<unsigned long long>(ds.record_bytes()));
    const uint64_t ram = amio::runtime::resolve_ram_budget_bytes(0);
    auto part = amio::runtime::select_memory_partition({path}, ram);
    std::printf("  partition=%s rho=%.2f\n", amio::runtime::profile_name(part.profile),
                part.dataset.rho);
    return 0;
  }
  if (path.size() >= 6 && path.substr(path.size() - 6) == ".ivecs") {
    std::vector<std::vector<uint32_t>> v;
    if (!amio::dataset::load_ivecs(path, &v)) {
      std::fprintf(stderr, "读取失败: %s\n", path.c_str());
      return 2;
    }
    std::printf("ivecs: %zu 行, first_len=%zu\n", v.size(),
                v.empty() ? 0 : v[0].size());
    return 0;
  }

  std::fprintf(stderr, "未知扩展名: %s\n", path.c_str());
  return 3;
}
