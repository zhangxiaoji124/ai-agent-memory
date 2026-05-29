#include "index/pq.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

#include "util/simd_distance.h"

namespace amio::index {

namespace {
constexpr uint32_t kMagicPQ = 0x00715150; // "PQ\0\0"-ish
}

bool ProductQuantizer::train(const std::vector<std::vector<float>> &data, uint32_t dim,
                             uint32_t m, uint32_t iters, uint32_t train_subset,
                             uint64_t seed) {
  trained_ = false;
  if (data.empty() || dim == 0 || m == 0 || dim % m != 0) {
    return false;
  }
  dim_ = dim;
  m_ = m;
  sub_dim_ = dim / m;
  ksub_ = 256;

  std::mt19937_64 rng(seed);

  // 采样训练子集（限耗时）。
  const size_t n = data.size();
  std::vector<uint32_t> idx;
  if (n > train_subset) {
    idx.reserve(train_subset);
    std::uniform_int_distribution<uint32_t> pick(0, static_cast<uint32_t>(n - 1));
    for (uint32_t i = 0; i < train_subset; i++) {
      idx.push_back(pick(rng));
    }
  } else {
    idx.resize(n);
    for (size_t i = 0; i < n; i++) {
      idx[i] = static_cast<uint32_t>(i);
    }
  }
  const size_t nt = idx.size();
  if (nt < ksub_) {
    return false; // 训练样本太少
  }

  centroids_.assign(static_cast<size_t>(m_) * ksub_ * sub_dim_, 0.0f);

  std::vector<uint32_t> assign(nt, 0);
  std::vector<double> sum(static_cast<size_t>(ksub_) * sub_dim_, 0.0);
  std::vector<uint32_t> cnt(ksub_, 0);

  for (uint32_t s = 0; s < m_; s++) {
    const uint32_t off = s * sub_dim_;
    float *cents = centroids_.data() + static_cast<size_t>(s) * ksub_ * sub_dim_;

    // 初始化：随机取 ksub 个样本子向量作为初始中心。
    {
      std::vector<uint32_t> init = idx;
      std::shuffle(init.begin(), init.end(), rng);
      for (uint32_t c = 0; c < ksub_; c++) {
        const float *src = data[init[c]].data() + off;
        std::memcpy(cents + static_cast<size_t>(c) * sub_dim_, src, sub_dim_ * sizeof(float));
      }
    }

    for (uint32_t it = 0; it < iters; it++) {
      // 分配
      for (size_t i = 0; i < nt; i++) {
        const float *v = data[idx[i]].data() + off;
        float best = std::numeric_limits<float>::max();
        uint32_t bc = 0;
        for (uint32_t c = 0; c < ksub_; c++) {
          const float d = amio::util::l2_sq_f32(v, cents + static_cast<size_t>(c) * sub_dim_, sub_dim_);
          if (d < best) {
            best = d;
            bc = c;
          }
        }
        assign[i] = bc;
      }
      // 更新
      std::fill(sum.begin(), sum.end(), 0.0);
      std::fill(cnt.begin(), cnt.end(), 0u);
      for (size_t i = 0; i < nt; i++) {
        const uint32_t c = assign[i];
        const float *v = data[idx[i]].data() + off;
        double *acc = sum.data() + static_cast<size_t>(c) * sub_dim_;
        for (uint32_t d = 0; d < sub_dim_; d++) {
          acc[d] += v[d];
        }
        cnt[c]++;
      }
      std::uniform_int_distribution<size_t> rpick(0, nt - 1);
      for (uint32_t c = 0; c < ksub_; c++) {
        float *cc = cents + static_cast<size_t>(c) * sub_dim_;
        if (cnt[c] == 0) {
          // 空簇：重置到随机样本子向量。
          const float *v = data[idx[rpick(rng)]].data() + off;
          std::memcpy(cc, v, sub_dim_ * sizeof(float));
        } else {
          const double inv = 1.0 / static_cast<double>(cnt[c]);
          const double *acc = sum.data() + static_cast<size_t>(c) * sub_dim_;
          for (uint32_t d = 0; d < sub_dim_; d++) {
            cc[d] = static_cast<float>(acc[d] * inv);
          }
        }
      }
    }
  }

  trained_ = true;
  return true;
}

void ProductQuantizer::encode(const float *vec, uint8_t *code) const {
  if (!trained_) {
    return;
  }
  for (uint32_t s = 0; s < m_; s++) {
    const float *v = vec + s * sub_dim_;
    float best = std::numeric_limits<float>::max();
    uint32_t bc = 0;
    for (uint32_t c = 0; c < ksub_; c++) {
      const float d = amio::util::l2_sq_f32(v, centroid(s, c), sub_dim_);
      if (d < best) {
        best = d;
        bc = c;
      }
    }
    code[s] = static_cast<uint8_t>(bc);
  }
}

void ProductQuantizer::compute_lut(const float *query, std::vector<float> *lut) const {
  if (!trained_ || !lut) {
    return;
  }
  lut->resize(static_cast<size_t>(m_) * ksub_);
  for (uint32_t s = 0; s < m_; s++) {
    const float *q = query + s * sub_dim_;
    float *row = lut->data() + static_cast<size_t>(s) * ksub_;
    for (uint32_t c = 0; c < ksub_; c++) {
      row[c] = amio::util::l2_sq_f32(q, centroid(s, c), sub_dim_);
    }
  }
}

float ProductQuantizer::adc_distance(const uint8_t *code, const std::vector<float> &lut) const {
  float s = 0.0f;
  for (uint32_t i = 0; i < m_; i++) {
    s += lut[static_cast<size_t>(i) * ksub_ + code[i]];
  }
  return s;
}

bool ProductQuantizer::save(const std::string &path) const {
  if (!trained_) {
    return false;
  }
  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) {
    return false;
  }
  uint32_t hdr[5] = {kMagicPQ, dim_, m_, ksub_, sub_dim_};
  bool ok = std::fwrite(hdr, sizeof(uint32_t), 5, f) == 5;
  if (ok) {
    ok = std::fwrite(centroids_.data(), sizeof(float), centroids_.size(), f) ==
         centroids_.size();
  }
  std::fclose(f);
  return ok;
}

bool ProductQuantizer::load(const std::string &path) {
  trained_ = false;
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return false;
  }
  uint32_t hdr[5] = {0, 0, 0, 0, 0};
  if (std::fread(hdr, sizeof(uint32_t), 5, f) != 5 || hdr[0] != kMagicPQ) {
    std::fclose(f);
    return false;
  }
  dim_ = hdr[1];
  m_ = hdr[2];
  ksub_ = hdr[3];
  sub_dim_ = hdr[4];
  if (m_ == 0 || ksub_ == 0 || sub_dim_ == 0 || dim_ != m_ * sub_dim_) {
    std::fclose(f);
    return false;
  }
  centroids_.assign(static_cast<size_t>(m_) * ksub_ * sub_dim_, 0.0f);
  const bool ok = std::fread(centroids_.data(), sizeof(float), centroids_.size(), f) ==
                  centroids_.size();
  std::fclose(f);
  trained_ = ok;
  return ok;
}

} // namespace amio::index
