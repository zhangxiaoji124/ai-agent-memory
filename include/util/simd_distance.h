#pragma once

#include <cstddef>
#include <cstdlib>

// 统一的 L2 平方距离：x86(GCC/Clang) 用 AVX2+FMA 运行时派发，其余平台标量兜底。
// 设计：避免改动 CMake 的全局编译选项——AVX2 实现用 __attribute__((target)) 局部启用，
// 通过 __builtin_cpu_supports 在运行时选择，因此一份二进制可同时跑在有/无 AVX2 的机器上。

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define AMIO_SIMD_X86 1
#endif

#if defined(AMIO_SIMD_X86) && (defined(__GNUC__) || defined(__clang__))
#define AMIO_SIMD_AVX2 1
#include <immintrin.h>
#endif

namespace amio::util {

inline float l2_sq_f32_scalar(const float *a, const float *b, size_t n) {
  float s = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

#if defined(AMIO_SIMD_AVX2)
__attribute__((target("avx2,fma"))) inline float
l2_sq_f32_avx2(const float *a, const float *b, size_t n) {
  __m256 acc = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    const __m256 d = _mm256_sub_ps(va, vb);
    acc = _mm256_fmadd_ps(d, d, acc);
  }
  // 水平求和 8 → 1
  __m128 lo = _mm256_castps256_ps128(acc);
  const __m128 hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  float s = _mm_cvtss_f32(lo);
  for (; i < n; ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

inline bool cpu_has_avx2() {
  // 静态局部变量：CPU 特征探测只做一次。GCC/Clang 在运行期构造函数里已初始化特征表。
  static const bool v =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return v;
}
#endif

/// 环境变量 AMIO_FORCE_SCALAR=1 时强制走标量路径（用于 SIMD 前后 A/B 对比，无需重编）。
inline bool force_scalar() {
  static const bool v = []() {
    const char *e = std::getenv("AMIO_FORCE_SCALAR");
    return e && e[0] == '1';
  }();
  return v;
}

/// L2 平方距离 sum((a[i]-b[i])^2)，i in [0,n)。a/b 必须各有至少 n 个 float。
inline float l2_sq_f32(const float *a, const float *b, size_t n) {
#if defined(AMIO_SIMD_AVX2)
  if (!force_scalar() && cpu_has_avx2())
    return l2_sq_f32_avx2(a, b, n);
#endif
  return l2_sq_f32_scalar(a, b, n);
}

} // namespace amio::util
