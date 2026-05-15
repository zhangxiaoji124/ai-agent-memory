#include <cstdio>
#include <string>
#include <vector>

#include "dataset/fvecs.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "用法: dataset_loader <file.fvecs|file.ivecs>\n");
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
