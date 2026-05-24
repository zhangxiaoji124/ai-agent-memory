#pragma once

#include <cstdint>

namespace amio::cache {

/// ISVM 简化：整数特征线性分类器，用于 PAIC 驱逐优先级（Demand-MIN 近似）。
/// 分数越高越应被驱逐；预取未消费、组小、年龄大 → 高驱逐分。
class IsvmScorer {
public:
  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool enabled() const { return enabled_; }

  int32_t eviction_score(bool is_prefetch_loaded, uint64_t tick_age,
                         uint32_t kv_group_bytes) const;

private:
  bool enabled_ = true;
};

} // namespace amio::cache
