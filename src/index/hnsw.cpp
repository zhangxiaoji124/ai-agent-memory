#include "index/hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

#include "util/macros.h"
#include "util/simd_distance.h"

namespace amio::index {

float HnswIndex::l2_sq(const std::vector<float> &a, const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  return amio::util::l2_sq_f32(a.data(), b.data(), n);
}

HnswIndex::HnswIndex(size_t dim, size_t m, size_t ef_construction, uint64_t seed,
                     float prune_alpha)
    : dim_(dim), m_(m), ef_construction_(ef_construction),
      prune_alpha_(prune_alpha > 1.0f ? prune_alpha : 1.2f), rng_(seed) {
  layer_neighbors_.resize(kMaxLayers);
}

int HnswIndex::random_level() {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double ml = 1.0 / std::log(2.0);
  int level = static_cast<int>(-std::log(u(rng_) + 1e-10) * ml);
  if (level < 0)
    level = 0;
  if (level >= kMaxLayers)
    level = kMaxLayers - 1;
  return level;
}

size_t HnswIndex::max_neighbors_for_layer(int layer) const {
  if (layer <= 0)
    return std::max<size_t>(1, m_ * 2);
  return std::max<size_t>(1, m_);
}

void HnswIndex::ensure_layer_sizes(int layer, uint64_t node_id) {
  AMIO_ASSERT(layer >= 0 && layer < kMaxLayers);
  auto &g = layer_neighbors_[static_cast<size_t>(layer)];
  while (g.size() <= static_cast<size_t>(node_id)) {
    g.push_back({});
  }
}

std::vector<std::pair<uint64_t, float>>
HnswIndex::search_layer(uint64_t ep, const std::vector<float> &q, size_t ef,
                        int layer) const {
  struct Item {
    float d;
    uint64_t id;
  };
  auto cmp_min = [](const Item &a, const Item &b) { return a.d > b.d; };
  auto cmp_max = [](const Item &a, const Item &b) { return a.d < b.d; };

  std::priority_queue<Item, std::vector<Item>, decltype(cmp_min)> candidates(cmp_min);
  std::priority_queue<Item, std::vector<Item>, decltype(cmp_max)> topk(cmp_max);
  std::unordered_set<uint64_t> visited;
  visited.reserve(ef * 4);

  const float d0 = l2_sq(q, vectors_[static_cast<size_t>(ep)]);
  candidates.push(Item{d0, ep});
  topk.push(Item{d0, ep});
  visited.insert(ep);

  while (!candidates.empty()) {
    const Item cur = candidates.top();
    candidates.pop();
    const float worst = topk.top().d;
    if (topk.size() >= ef && cur.d > worst)
      break;

    const auto &nb_list = layer_neighbors_[static_cast<size_t>(layer)];
    if (static_cast<size_t>(cur.id) >= nb_list.size())
      continue;
    const auto &neigh = nb_list[static_cast<size_t>(cur.id)];
    for (uint64_t nb : neigh) {
      if (node_level_[static_cast<size_t>(nb)] < layer)
        continue;
      if (visited.insert(nb).second) {
        const float d = l2_sq(q, vectors_[static_cast<size_t>(nb)]);
        if (topk.size() < ef || d < topk.top().d) {
          candidates.push(Item{d, nb});
          topk.push(Item{d, nb});
          if (topk.size() > ef)
            topk.pop();
        }
      }
    }
  }

  std::vector<std::pair<uint64_t, float>> out;
  out.reserve(topk.size());
  while (!topk.empty()) {
    out.push_back({topk.top().id, topk.top().d});
    topk.pop();
  }
  return out;
}

uint64_t HnswIndex::pick_closest(const std::vector<std::pair<uint64_t, float>> &w) {
  if (w.empty())
    return 0;
  auto it = std::min_element(w.begin(), w.end(),
                             [](const auto &a, const auto &b) { return a.second < b.second; });
  return it->first;
}

std::vector<uint64_t> HnswIndex::robust_prune(
    uint64_t center_id,
    std::vector<std::pair<uint64_t, float>> candidates, size_t cap) const {
  if (cap == 0 || candidates.empty() || center_id >= vectors_.size()) {
    return {};
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b) {
              if (a.second != b.second) {
                return a.second < b.second;
              }
              return a.first < b.first;
            });

  std::vector<std::pair<uint64_t, float>> deduped;
  deduped.reserve(candidates.size());
  for (const auto &c : candidates) {
    if (c.first == center_id) {
      continue;
    }
    if (!deduped.empty() && deduped.back().first == c.first) {
      if (c.second < deduped.back().second) {
        deduped.back().second = c.second;
      }
      continue;
    }
    deduped.push_back(c);
  }

  std::vector<uint64_t> result;
  result.reserve(std::min(cap, deduped.size()));

  for (const auto &cand : deduped) {
    if (result.size() >= cap) {
      break;
    }
    const float dist_cp = cand.second;
    bool occluded = false;
    for (uint64_t sel : result) {
      const float dist_cs = l2_sq(vectors_[static_cast<size_t>(cand.first)],
                                  vectors_[static_cast<size_t>(sel)]);
      // DiskANN/Vamana 遮挡：已选邻居 sel 比 center 更靠近 cand 时剪掉 cand。
      if (prune_alpha_ * dist_cs <= dist_cp) {
        occluded = true;
        break;
      }
    }
    if (!occluded) {
      result.push_back(cand.first);
    }
  }
  return result;
}

std::vector<uint64_t>
HnswIndex::select_neighbors(uint64_t center_id,
                            const std::vector<std::pair<uint64_t, float>> &w,
                            int layer) const {
  return robust_prune(center_id, w, max_neighbors_for_layer(layer));
}

void HnswIndex::add_edge(int layer, uint64_t a, uint64_t b) {
  if (a == b) {
    return;
  }
  if (node_level_[static_cast<size_t>(a)] < layer ||
      node_level_[static_cast<size_t>(b)] < layer) {
    return;
  }

  auto try_add = [&](uint64_t x, uint64_t y) {
    if (node_level_[static_cast<size_t>(x)] < layer) {
      return;
    }
    auto &lst = layer_neighbors_[static_cast<size_t>(layer)][static_cast<size_t>(x)];
    if (std::find(lst.begin(), lst.end(), y) != lst.end()) {
      return;
    }

    const size_t cap = max_neighbors_for_layer(layer);
    // 未满：直接追加，避免每条边都重跑 O(cap^2) 的 robust_prune（hnswlib/DiskANN 同款）。
    if (lst.size() < cap) {
      lst.push_back(y);
      return;
    }
    // 已满：现有邻居 + 新点一起重剪枝，保留多样化的 cap 个。
    const auto &vx = vectors_[static_cast<size_t>(x)];
    std::vector<std::pair<uint64_t, float>> cand;
    cand.reserve(lst.size() + 1);
    for (uint64_t nb : lst) {
      cand.push_back({nb, l2_sq(vx, vectors_[static_cast<size_t>(nb)])});
    }
    cand.push_back({y, l2_sq(vx, vectors_[static_cast<size_t>(y)])});
    lst = robust_prune(x, std::move(cand), cap);
  };

  try_add(a, b);
  try_add(b, a);
}

int HnswIndex::node_max_layer(uint64_t id) const {
  if (id >= node_level_.size())
    return -1;
  return node_level_[static_cast<size_t>(id)];
}

const std::vector<uint64_t> &HnswIndex::neighbors_at(int layer, uint64_t id) const {
  static const std::vector<uint64_t> kEmpty{};
  if (layer < 0 || layer >= kMaxLayers)
    return kEmpty;
  const auto &g = layer_neighbors_[static_cast<size_t>(layer)];
  if (static_cast<size_t>(id) >= g.size())
    return kEmpty;
  return g[static_cast<size_t>(id)];
}

void HnswIndex::insert(uint64_t id, std::vector<float> vec) {
  if (id != id_base_ + vectors_.size() || vec.size() != dim_)
    return;

  const int cur_level = random_level();
  node_level_.push_back(cur_level);
  vectors_.push_back(std::move(vec));
  const auto &q = vectors_.back();

  for (int l = 0; l < kMaxLayers; l++)
    ensure_layer_sizes(l, id);

  if (id == 0) {
    entry_point_ = 0;
    max_layer_ = cur_level;
    return;
  }

  const int ml_old = max_layer_;
  uint64_t ep = entry_point_;

  for (int lc = ml_old; lc > cur_level; lc--) {
    auto w = search_layer(ep, q, 1, lc);
    if (!w.empty())
      ep = pick_closest(w);
  }

  uint64_t ep_conn = ep;
  for (int lc = cur_level; lc >= 0; lc--) {
    auto w = search_layer(ep_conn, q, ef_construction_, lc);
    const auto neigh = select_neighbors(id, w, lc);
    for (uint64_t nb : neigh)
      add_edge(lc, id, nb);
    ep_conn = w.empty() ? ep_conn : pick_closest(w);
  }

  if (cur_level > ml_old) {
    max_layer_ = cur_level;
    entry_point_ = id;
  }
}

std::vector<std::pair<uint64_t, float>>
HnswIndex::search(const std::vector<float> &query, size_t k, size_t ef) const {
  if (vectors_.empty() || query.size() != dim_)
    return {};
  uint64_t ep = entry_point_;
  const int ml = max_layer_;
  for (int lc = ml; lc > 0; lc--) {
    auto w = search_layer(ep, query, 1, lc);
    if (!w.empty())
      ep = pick_closest(w);
  }
  auto w0 = search_layer(ep, query, std::max(ef, k), 0);
  std::sort(w0.begin(), w0.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
  if (w0.size() > k)
    w0.resize(k);
  return w0;
}

} // namespace amio::index
