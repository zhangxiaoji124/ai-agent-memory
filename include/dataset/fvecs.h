#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace amio::dataset {

bool load_fvecs(const std::string &path, std::vector<std::vector<float>> *out);
bool load_ivecs(const std::string &path, std::vector<std::vector<uint32_t>> *out);

double compute_recall_at_k(const std::vector<std::vector<uint32_t>> &results,
                           const std::vector<std::vector<uint32_t>> &groundtruth,
                           size_t k);

} // namespace amio::dataset
