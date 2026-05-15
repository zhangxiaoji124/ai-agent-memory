#include "dataset/fvecs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <unordered_set>

namespace amio::dataset {

template <class T>
static bool read_exact(std::ifstream &ifs, T *out) {
  return static_cast<bool>(ifs.read(reinterpret_cast<char *>(out), sizeof(T)));
}

bool load_fvecs(const std::string &path, std::vector<std::vector<float>> *out) {
  if (!out)
    return false;
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    return false;
  out->clear();
  while (true) {
    uint32_t dim = 0;
    if (!read_exact(ifs, &dim))
      break;
    std::vector<float> v(dim);
    if (!ifs.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(dim * 4)))
      return false;
    out->push_back(std::move(v));
  }
  return true;
}

bool load_ivecs(const std::string &path, std::vector<std::vector<uint32_t>> *out) {
  if (!out)
    return false;
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    return false;
  out->clear();
  while (true) {
    uint32_t dim = 0;
    if (!read_exact(ifs, &dim))
      break;
    std::vector<int32_t> tmp(dim);
    if (!ifs.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(dim * 4)))
      return false;
    std::vector<uint32_t> v(dim);
    for (size_t i = 0; i < dim; i++)
      v[i] = static_cast<uint32_t>(tmp[i]);
    out->push_back(std::move(v));
  }
  return true;
}

double compute_recall_at_k(const std::vector<std::vector<uint32_t>> &results,
                           const std::vector<std::vector<uint32_t>> &groundtruth,
                           size_t k) {
  if (results.size() != groundtruth.size() || results.empty() || k == 0)
    return 0.0;
  size_t total = 0;
  for (size_t i = 0; i < results.size(); i++) {
    const auto &res = results[i];
    const auto &gt = groundtruth[i];
    const size_t kk = std::min({k, res.size(), gt.size()});
    std::unordered_set<uint32_t> s;
    s.reserve(kk * 2);
    for (size_t j = 0; j < kk; j++)
      s.insert(gt[j]);
    for (size_t j = 0; j < kk; j++)
      total += (s.find(res[j]) != s.end()) ? 1 : 0;
  }
  return static_cast<double>(total) / static_cast<double>(results.size() * k);
}

} // namespace amio::dataset