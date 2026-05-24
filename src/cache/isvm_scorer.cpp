#include "cache/isvm_scorer.h"

#include <algorithm>

namespace amio::cache {

int32_t IsvmScorer::eviction_score(bool is_prefetch_loaded, uint64_t tick_age,
                                   uint32_t kv_group_bytes) const {
  if (!enabled_) {
    return static_cast<int32_t>(tick_age);
  }
  int32_t s = static_cast<int32_t>(tick_age / 16);
  if (is_prefetch_loaded) {
    s += 64;
  }
  if (kv_group_bytes <= 64) {
    s -= 16;
  } else if (kv_group_bytes >= 512) {
    s += 24;
  }
  return s;
}

} // namespace amio::cache
