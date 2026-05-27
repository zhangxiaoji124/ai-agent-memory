#include "vector_store.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <queue>
#include <unordered_set>

#include "runtime/static_subgraph.h"
#include "util/macros.h"
#include "util/simd_distance.h"
#include "util/topk.h"

namespace amio {

static std::vector<SearchResult> merge_top_k(std::vector<SearchResult> a,
                                            std::vector<SearchResult> b,
                                            size_t k) {
  a.insert(a.end(), b.begin(), b.end());
  std::sort(a.begin(), a.end(),
            [](const auto &x, const auto &y) { return x.distance < y.distance; });
  if (a.size() > k)
    a.resize(k);
  return a;
}

namespace {

uint64_t pick_closest(const std::vector<std::pair<uint64_t, float>> &w) {
  if (w.empty())
    return 0;
  auto it = std::min_element(w.begin(), w.end(), [](const auto &a, const auto &b) {
    return a.second < b.second;
  });
  return it->first;
}

/// 在磁盘上复现单层 best-first 搜索（与内存 HNSW 同构）。
/// policy 非空时启用 θ 阶段切换：
///   - 收敛期（visited < θ·ef）：使用标准预取深度
///   - 探索期（visited ≥ θ·ef）：phase2=true，使用 layer0 扇出加速预取
std::vector<std::pair<uint64_t, float>>
disk_search_layer(index::IndexFile &file, cache::GraphAwareCache &cache,
                  prefetch::IoUringBackend *uring, prefetch::TopologyPrefetcher &prefetch,
                  const index::ExternalVectorStore *ext_vecs, uint64_t ep,
                  const std::vector<float> &q, size_t ef, int layer, size_t dim,
                  uint64_t total_nodes,
                  float (*dist_block)(const std::vector<float> &, const index::NodeBlock &,
                                      size_t),
                  const policy::AgentIoPolicy *pol, IoMetrics *io_metrics) {
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

  if (total_nodes == 0)
    return {};
  if (ep >= total_nodes)
    ep = 0;

  // θ 阶段切换参数（GoVector 动静混合核心）
  const float theta = (pol && pol->partition_theta > 0.0f) ? pol->partition_theta : 0.0f;
  const size_t theta_threshold = (theta > 0.0f && ef > 0)
                                     ? static_cast<size_t>(theta * static_cast<float>(ef))
                                     : 0;
  bool phase2 = false;
  bool phase2_logged = false;
  size_t visited_count = 0;

  index::NodeBlock epb{};
  if (!cache.get_or_load(&file, uring, ep, &epb)) {
    if (ep != 0 && cache.get_or_load(&file, uring, 0, &epb))
      ep = 0;
    else
      return {};
  }
  prefetch.on_visit_node(epb, static_cast<uint8_t>(layer), phase2);
  visited.insert(ep);
  ++visited_count;

  const float d0 =
      (ext_vecs && ext_vecs->ok()) ? ext_vecs->l2_sq(q, ep) : dist_block(q, epb, dim);
  candidates.push(Item{d0, ep});
  topk.push(Item{d0, ep});

  while (!candidates.empty()) {
    // θ 阶段检测：visited_count 超过阈值时切换到探索期
    if (!phase2 && theta_threshold > 0 && visited_count >= theta_threshold) {
      phase2 = true;
      if (!phase2_logged && io_metrics) {
        io_metrics->theta_phase_switches.fetch_add(1, std::memory_order_relaxed);
        phase2_logged = true;
      }
    }

    const Item cur = candidates.top();
    candidates.pop();
    const float worst = topk.top().d;
    if (topk.size() >= ef && cur.d > worst)
      break;

    index::NodeBlock curb{};
    if (!cache.get_or_load(&file, uring, cur.id, &curb))
      continue;
    prefetch.on_visit_node(curb, static_cast<uint8_t>(layer), phase2);
    const uint8_t cnt = curb.neighbor_counts[layer];
    const size_t n = cnt > 32 ? 32 : cnt;
    for (size_t j = 0; j < n; j++) {
      const uint64_t nb = static_cast<uint64_t>(curb.neighbors[layer][j]);
      if (nb >= total_nodes)
        continue;
      if (visited.insert(nb).second) {
        ++visited_count;
        index::NodeBlock nbk{};
        if (!cache.get_or_load(&file, uring, nb, &nbk))
          continue;
        const float d = (ext_vecs && ext_vecs->ok()) ? ext_vecs->l2_sq(q, nb)
                                                     : dist_block(q, nbk, dim);
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

} // namespace

VectorStore::VectorStore(const Config &cfg)
    : cfg_(cfg),
      index_(128, 16, 200, 42),
      io_policy_(policy::AgentIoPolicy::from_config(cfg)),
      cache_(cfg.cache_size_mb * 1024ull * 1024ull,
             (cfg.static_cache_mb > 0
                  ? cfg.static_cache_mb
                  : std::max<size_t>(
                        1, cfg.partition.pools.static_cache_bytes / (1024 * 1024))) *
                 1024ull * 1024ull,
             &io_policy_, &io_metrics_),
      prefetch_(cfg.prefetch_depth, &io_policy_, &io_metrics_),
      memtable_(cfg.memtable_limit_mb * 1024ull * 1024ull),
      nvtable_(cfg.memtable_limit_mb * 1024ull * 1024ull * 2) {
  prefetch_.set_cache(&cache_);
  cache_.set_isvm_kv_cache_enabled(cfg.enable_isvm_kv_cache);
}

VectorStore::~VectorStore() = default;

bool VectorStore::open() {
  if (cfg_.index_path.empty())
    return true;

  io_policy_ = policy::AgentIoPolicy::from_config(cfg_);
  if (cfg_.agent_policy_mode == AgentPolicyMode::Learned &&
      !cfg_.agent_policy_path.empty()) {
    std::string err;
    if (!policy::AgentIoPolicy::merge_json_file(cfg_.agent_policy_path, &io_policy_,
                                                &err)) {
      std::fprintf(stderr, "agent policy: %s (使用 Config 默认策略)\n", err.c_str());
    }
  }

  if (cfg_.enable_compaction) {
    if (!file_.open_readwrite(cfg_.index_path)) {
      if (!file_.open_readonly(cfg_.index_path)) {
        return false;
      }
      index_writable_ = false;
    } else {
      index_writable_ = true;
    }
  } else if (!file_.open_readonly(cfg_.index_path)) {
    return false;
  }
  if (!file_.read_header(&header_)) {
    return false;
  }
  next_node_id_ = header_.total_nodes;

  ext_vecs_.reset();
  if (index::header_uses_external_vectors(header_)) {
    ext_vecs_ = std::make_unique<index::ExternalVectorStore>();
    if (!ext_vecs_->attach(file_.fd(), header_, cfg_.index_path)) {
      std::fprintf(stderr, "external vector section attach failed (dim=%u)\n", header_.dim);
      ext_vecs_.reset();
    } else {
      std::fprintf(stderr, "external vectors: dim=%u stride=%u enc=%u\n", header_.dim,
                   header_.vector_stride_bytes,
                   static_cast<unsigned>(header_.vector_encoding));
    }
  }

  if (cfg_.enable_wal) {
    wal_ = std::make_unique<write::Wal>(cfg_.index_path + ".wal");
  }

  {
    std::lock_guard lk(index_mu_);
    index_ = index::HnswIndex(static_cast<size_t>(header_.dim), header_.m ? header_.m : 16,
                              header_.ef_construction ? header_.ef_construction : 200, 42);
    index_.set_id_base(header_.total_nodes);
  }

  if (cfg_.enable_compaction) {
    compaction_ = std::make_unique<write::CompactionWorker>(
        [this](const write::CompactionWorker::Batch &batch) {
          write::IndexMergeStats st{};
          {
            std::lock_guard lk(index_mu_);
            if (index_writable_ && file_.fd() >= 0) {
              st = write::merge_batch_into_index(file_, header_, index_, batch,
                                                 cfg_.enable_compaction ? &remap_ : nullptr);
            } else {
              for (const auto &kv : batch) {
                if (kv.first == index_.id_base() + index_.size() &&
                    kv.second.size() == header_.dim) {
                  index_.insert(kv.first, kv.second);
                }
              }
              st.ok = true;
              st.nodes_merged = batch.size();
            }
          }
          if (st.ok) {
            on_compaction_applied(st);
          }
        });
  }

#if defined(AMIO_HAS_URING) && AMIO_HAS_URING
  if (!cfg_.force_disable_uring) {
    uring_ = std::make_unique<prefetch::IoUringBackend>(file_.fd(), 256);
    if (uring_->ok()) {
      prefetch_.set_io_backend(uring_.get());
    } else {
      uring_.reset();
      prefetch_.set_io_backend(nullptr);
    }
  } else {
    prefetch_.set_io_backend(nullptr);
  }
#else
  prefetch_.set_io_backend(nullptr);
#endif

  if (cfg_.enable_static_subgraph_pin && header_.total_nodes > 0) {
    const uint64_t budget =
        cfg_.partition.pools.static_cache_bytes > 0
            ? cfg_.partition.pools.static_cache_bytes
            : cfg_.static_cache_mb * 1024ull * 1024ull;
    const uint64_t pinned = runtime::pin_static_navigation_subgraph(
        file_, cache_, uring_.get(), header_.entry_point, header_.max_layer,
        header_.total_nodes, budget);
    std::fprintf(stderr, "static subgraph: pinned_bytes=%llu pins=%llu budget=%llu\n",
                 static_cast<unsigned long long>(pinned),
                 static_cast<unsigned long long>(cache_.static_pins_count()),
                 static_cast<unsigned long long>(budget));
  }
  return true;
}

void VectorStore::reset_search_observability() {
  io_metrics_.reset();
  cache_.reset_access_counters();
}

float VectorStore::l2_sq(const std::vector<float> &a,
                         const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  return amio::util::l2_sq_f32(a.data(), b.data(), n);
}

float VectorStore::l2_sq_block(const std::vector<float> &q,
                               const amio::index::NodeBlock &b, size_t dim) {
  const size_t n = std::min(dim, static_cast<size_t>(128));
  return amio::util::l2_sq_f32(q.data(), b.vector, n);
}

std::vector<SearchResult> VectorStore::search(const std::vector<float> &query,
                                               size_t k) {
  std::vector<SearchResult> main_res;
  {
    std::lock_guard lk(index_mu_);
    auto found = index_.search(query, k, cfg_.ef_search);
    main_res.reserve(found.size());
    for (auto &p : found)
      main_res.push_back({p.first, p.second});
  }

  if (!cfg_.enable_memtable_search)
    return main_res;

  auto mem = memtable_.brute_force_search(query, k);
  std::vector<SearchResult> mem_res;
  mem_res.reserve(mem.size());
  for (auto &p : mem)
    mem_res.push_back({p.first, p.second});

  return merge_top_k(std::move(main_res), std::move(mem_res), k);
}

bool VectorStore::insert(uint64_t id, const std::vector<float> &vec) {
  uint64_t nid = id;
  {
    std::lock_guard lk(index_mu_);
    if (file_.fd() >= 0) {
      if (nid < next_node_id_) {
        nid = next_node_id_;
      } else if (nid > next_node_id_) {
        nid = next_node_id_;
      }
      next_node_id_++;
    }
  }

  if (wal_) {
    (void)wal_->append(nid, vec);
  }
  const bool need_flush = memtable_.insert(nid, vec);
  if (need_flush) {
    auto batch = memtable_.drain();
    if (cfg_.enable_nvtable) {
      const bool nv_full = nvtable_.append_slice(std::move(batch));
      if (nv_full || nvtable_.slice_count() >= 4) {
        submit_compaction_batch(nvtable_.drain_all());
      }
    } else {
      submit_compaction_batch(std::move(batch));
    }
  } else if (!cfg_.enable_compaction || !index_writable_) {
    std::lock_guard lk(index_mu_);
    if (nid == index_.id_base() + index_.size() &&
        vec.size() == (header_.dim ? header_.dim : vec.size())) {
      index_.insert(nid, vec);
    }
  }
  return true;
}

void VectorStore::submit_compaction_batch(write::CompactionWorker::Batch batch) {
  if (batch.empty()) {
    return;
  }
  if (compaction_) {
    compaction_->submit(std::move(batch));
  }
}

void VectorStore::on_compaction_applied(const write::IndexMergeStats &st) {
  if (!st.ok) {
    return;
  }
  if (cfg_.enable_noblsm && wal_) {
    const std::string wal_path = cfg_.index_path + ".wal";
    noblsm_.check_commit(wal_path);
    noblsm_.mark_data_written(wal_path);
    noblsm_.mark_data_written(cfg_.index_path);
    for (const auto &p : noblsm_.poll_committed()) {
      if (p == wal_path) {
        (void)wal_->truncate();
      }
    }
  }
  std::fprintf(stderr,
               "compaction: merged=%llu blocks_written=%llu remapped=%llu total_nodes=%llu\n",
               static_cast<unsigned long long>(st.nodes_merged),
               static_cast<unsigned long long>(st.blocks_written),
               static_cast<unsigned long long>(st.blocks_remapped),
               static_cast<unsigned long long>(header_.total_nodes));
}

void VectorStore::flush_writes() {
  if (cfg_.enable_nvtable && !nvtable_.empty()) {
    submit_compaction_batch(nvtable_.drain_all());
  }
  if (compaction_) {
    compaction_->flush();
  }
}

std::vector<SearchResult> VectorStore::search_disk(const std::vector<float> &query,
                                                  size_t k) {
  std::vector<SearchResult> out;
  if (file_.fd() < 0)
    return out;
  const size_t dim = static_cast<size_t>(header_.dim);
  const uint64_t total_nodes = header_.total_nodes;
  if (total_nodes == 0)
    return out;
  if (query.size() != dim)
    return out;

  uint64_t ep = header_.entry_point;
  if (ep >= total_nodes)
    ep = 0;
  const int ml = std::min(7, static_cast<int>(header_.max_layer));
  // 高层贪心导航：不启用 θ（不在 layer0，每层只取 1 个最近点）
  for (int lc = ml; lc > 0; lc--) {
    auto w = disk_search_layer(file_, cache_, uring_.get(), prefetch_, ext_vecs_.get(), ep,
                               query, 1, lc, dim, total_nodes, &VectorStore::l2_sq_block,
                               nullptr, nullptr);
    if (!w.empty())
      ep = pick_closest(w);
  }

  // layer0 beam search：启用 θ 阶段切换（GoVector 动静混合核心路径）
  auto w0 =
      disk_search_layer(file_, cache_, uring_.get(), prefetch_, ext_vecs_.get(), ep, query,
                        cfg_.ef_search, 0, dim, total_nodes, &VectorStore::l2_sq_block,
                        &io_policy_, &io_metrics_);
  std::sort(w0.begin(), w0.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
  if (w0.size() > k)
    w0.resize(k);

  out.reserve(w0.size());
  for (auto &p : w0)
    out.push_back({p.first, p.second});

  if (!cfg_.enable_memtable_search)
    return out;

  auto mem = memtable_.brute_force_search(query, k);
  std::vector<SearchResult> mem_res;
  mem_res.reserve(mem.size());
  for (auto &p : mem)
    mem_res.push_back({p.first, p.second});
  return merge_top_k(std::move(out), std::move(mem_res), k);
}

} // namespace amio
