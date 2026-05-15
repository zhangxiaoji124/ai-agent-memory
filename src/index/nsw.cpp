#include "index/nsw.h"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace amio::index {

float NswIndex::l2_sq(const std::vector<float> &a, const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  float s = 0.0f;
  for (size_t i = 0; i < n; i++) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

NswIndex::NswIndex(size_t dim, size_t m, size_t ef_construction, uint64_t seed)
    : dim_(dim), m_(m), ef_construction_(ef_construction), rng_(seed) {}

std::vector<std::pair<uint64_t, float>>
NswIndex::search_layer(uint64_t ep, const std::vector<float> &q, size_t ef) const {
  struct Item {
    float d;
    uint64_t id;
  };
  auto cmp_min = [](const Item &a, const Item &b) { return a.d > b.d; };
  auto cmp_max = [](const Item &a, const Item &b) { return a.d < b.d; };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp_min)> candidates(cmp_min);
  std::priority_queue<Item, std::vector<Item>, decltype(cmp_max)> topk(cmp_max);

  std::unordered_set<uint64_t> visited;
  visited.reserve(ef * 2);

  const float d0 = l2_sq(q, vectors_[static_cast<size_t>(ep)]);
  candidates.push(Item{d0, ep});
  topk.push(Item{d0, ep});
  visited.insert(ep);

  while (!candidates.empty()) {
    const Item cur = candidates.top();
    candidates.pop();
    const float worst = topk.top().d;
    if (topk.size() >= ef && cur.d > worst) {
      break;
    }
    const auto &neigh = graph_[static_cast<size_t>(cur.id)];
    for (uint64_t nb : neigh) {
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

void NswIndex::add_edge(uint64_t a, uint64_t b) {
  if (a == b)
    return;
  auto push = [&](uint64_t x, uint64_t y) {
    auto &lst = graph_[static_cast<size_t>(x)];
    if (std::find(lst.begin(), lst.end(), y) == lst.end())
      lst.push_back(y);
    const size_t cap = std::max<size_t>(1, m_ * 2);
    if (lst.size() > cap)
      lst.resize(cap);
  };
  push(a, b);
  push(b, a);
}

void NswIndex::insert(uint64_t id, std::vector<float> vec) {
  // 连续 ID 约束，方便映射到磁盘块。
  if (id != vectors_.size())
    return;
  if (vec.size() != dim_)
    return;

  vectors_.push_back(std::move(vec));
  graph_.push_back({});

  if (!has_entry_) {
    entry_point_ = id;
    has_entry_ = true;
    return;
  }

  const auto &q = vectors_.back();
  const size_t ef = std::max<size_t>(ef_construction_, m_ * 2);
  auto found = search_layer(entry_point_, q, ef);
  if (found.empty()) {
    add_edge(id, entry_point_);
    return;
  }
  std::sort(found.begin(), found.end(),
            [](auto &a, auto &b) { return a.second < b.second; });
  const size_t cap = std::max<size_t>(1, m_ * 2);
  if (found.size() > cap)
    found.resize(cap);
  for (auto &p : found) {
    add_edge(id, p.first);
  }

  // 轻量启发式：小概率更新入口点（提升连通性）。
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  const double p = std::min(0.05, 1.0 / static_cast<double>(vectors_.size()));
  if (dis(rng_) < p) {
    entry_point_ = id;
  }
}

std::vector<std::pair<uint64_t, float>>
NswIndex::search(const std::vector<float> &query, size_t k, size_t ef) const {
  if (!has_entry_ || query.size() != dim_)
    return {};
  auto found = search_layer(entry_point_, query, std::max(k, ef));
  std::sort(found.begin(), found.end(),
            [](auto &a, auto &b) { return a.second < b.second; });
  if (found.size() > k)
    found.resize(k);
  return found;
}

} // namespace amio::index
