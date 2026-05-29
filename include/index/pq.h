#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace amio::index {

/// 乘积量化（Product Quantization）。
/// 把 dim 维向量切成 m 个子空间（每段 sub_dim=dim/m 维），每段用 256 个聚类中心编码，
/// 于是每条向量压成 m 字节码。检索时对查询预计算 LUT，用 ADC（非对称距离）查表近似
/// L2 平方距离 = Σ_s LUT[s][code[s]]，再对 top 候选做全精度重排找回召回。
class ProductQuantizer {
public:
  bool ok() const { return trained_; }
  uint32_t dim() const { return dim_; }
  uint32_t m() const { return m_; }
  uint32_t ksub() const { return ksub_; }
  uint32_t sub_dim() const { return sub_dim_; }
  uint64_t centroids_bytes() const {
    return static_cast<uint64_t>(centroids_.size()) * sizeof(float);
  }

  /// 训练：对每个子空间跑 k-means（ksub=256 中心，Lloyd 迭代）。
  /// 要求 dim % m == 0。data 超过 train_subset 时随机采样以限训练耗时。
  bool train(const std::vector<std::vector<float>> &data, uint32_t dim, uint32_t m,
             uint32_t iters = 20, uint32_t train_subset = 50000, uint64_t seed = 42);

  /// 编码单条向量 → m 字节码（out 至少 m 字节）。
  void encode(const float *vec, uint8_t *code) const;

  /// 为查询预计算 LUT（大小 m*ksub）；随后 adc_distance 复用它。
  void compute_lut(const float *query, std::vector<float> *lut) const;

  /// 近似 L2 平方距离 = Σ_s lut[s*ksub + code[s]]。
  float adc_distance(const uint8_t *code, const std::vector<float> &lut) const;

  /// 码本（含 centroids）序列化到磁盘 / 从磁盘加载。
  bool save(const std::string &path) const;
  bool load(const std::string &path);

private:
  // centroids_[(s*ksub_ + c)*sub_dim_ + d]
  const float *centroid(uint32_t s, uint32_t c) const {
    return centroids_.data() +
           (static_cast<size_t>(s) * ksub_ + c) * sub_dim_;
  }

  bool trained_ = false;
  uint32_t dim_ = 0;
  uint32_t m_ = 0;
  uint32_t sub_dim_ = 0;
  uint32_t ksub_ = 256;
  std::vector<float> centroids_;
};

} // namespace amio::index
