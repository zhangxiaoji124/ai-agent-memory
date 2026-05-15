#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "dataset/fvecs.h"
#include "index/hnsw.h"
#include "index/storage_layout.h"
#include "util/time.h"
#include "vector_store.h"

static const char *arg_value(int argc, char **argv, const char *key) {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::strcmp(argv[i], key) == 0) {
      return argv[i + 1];
    }
  }
  return nullptr;
}

static size_t arg_size_t(int argc, char **argv, const char *key, size_t def) {
  const char *v = arg_value(argc, argv, key);
  if (!v) {
    return def;
  }
  return static_cast<size_t>(std::strtoull(v, nullptr, 10));
}

static bool arg_bool(int argc, char **argv, const char *key, bool def) {
  const char *v = arg_value(argc, argv, key);
  if (!v) {
    return def;
  }
  return std::atoi(v) != 0;
}

static std::string arg_string(int argc, char **argv, const char *key,
                              const std::string &def) {
  const char *v = arg_value(argc, argv, key);
  if (!v) {
    return def;
  }
  return std::string(v);
}

static double recall_one_query(const std::vector<uint32_t> &res,
                             const std::vector<uint32_t> &gt, size_t k) {
  if (k == 0)
    return 0.0;
  const size_t kk = std::min({k, res.size(), gt.size()});
  if (kk == 0)
    return 0.0;
  std::unordered_set<uint32_t> s;
  s.reserve(kk * 2);
  for (size_t j = 0; j < kk; j++)
    s.insert(gt[j]);
  size_t hit = 0;
  for (size_t j = 0; j < kk; j++) {
    if (s.find(res[j]) != s.end())
      hit++;
  }
  return static_cast<double>(hit) / static_cast<double>(kk);
}

static bool build_index_file(const std::vector<std::vector<float>> &vecs,
                             const std::string &output, uint32_t m,
                             uint32_t ef_construction) {
  if (vecs.empty()) {
    return false;
  }
  const uint32_t dim = static_cast<uint32_t>(vecs[0].size());

  amio::index::HnswIndex idx(dim, m, ef_construction, 42);
  for (uint64_t id = 0; id < static_cast<uint64_t>(vecs.size()); id++) {
    idx.insert(id, vecs[static_cast<size_t>(id)]);
  }

  amio::index::IndexFile f;
  if (!f.open_create_trunc(output)) {
    return false;
  }

  amio::index::IndexFileHeader h{};
  h.total_nodes = static_cast<uint64_t>(vecs.size());
  h.dim = dim;
  h.entry_point = idx.entry_point();
  h.m = m;
  h.ef_construction = ef_construction;
  h.max_layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.max_layer())));
  if (!f.write_header(h)) {
    return false;
  }

  for (uint64_t id = 0; id < static_cast<uint64_t>(vecs.size()); id++) {
    amio::index::NodeBlock b{};
    b.node_id = id;
    b.layer = static_cast<uint8_t>(std::min(255, std::max(0, idx.node_max_layer(id))));
    b.vec_dim = dim > 65535u ? 65535u : static_cast<uint16_t>(dim);
    for (uint32_t i = 0; i < dim && i < 128; i++) {
      b.vector[i] = vecs[static_cast<size_t>(id)][i];
    }
    for (int l = 0; l < 8; l++) {
      const auto &nbl = idx.neighbors_at(l, id);
      const size_t cnt = std::min<size_t>(32, nbl.size());
      b.neighbor_counts[static_cast<size_t>(l)] = static_cast<uint8_t>(cnt);
      for (size_t j = 0; j < cnt; j++) {
        const uint64_t nid = nbl[j];
        b.neighbors[static_cast<size_t>(l)][j] = static_cast<uint32_t>(nid);
        b.neighbor_offsets[static_cast<size_t>(l)][j] = amio::index::node_offset(nid);
      }
    }
    if (!f.pwrite_node(id, b)) {
      return false;
    }
  }

  f.close();
  return true;
}

static float l2_sq(const std::vector<float> &a, const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  float s = 0.0f;
  for (size_t i = 0; i < n; i++) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

static std::vector<std::vector<uint32_t>>
compute_gt_from_base(const std::vector<std::vector<float>> &base,
                     const std::vector<std::vector<float>> &queries, size_t k) {
  struct Item {
    float dist;
    uint32_t id;
  };
  auto cmp = [](const Item &a, const Item &b) { return a.dist < b.dist; };
  std::vector<std::vector<uint32_t>> gt;
  gt.reserve(queries.size());
  for (const auto &q : queries) {
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> best(cmp);
    for (uint32_t id = 0; id < static_cast<uint32_t>(base.size()); id++) {
      const float d = l2_sq(q, base[id]);
      if (best.size() < k) {
        best.push(Item{d, id});
      } else if (d < best.top().dist) {
        best.pop();
        best.push(Item{d, id});
      }
    }
    std::vector<Item> sorted;
    sorted.reserve(best.size());
    while (!best.empty()) {
      sorted.push_back(best.top());
      best.pop();
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const Item &a, const Item &b) { return a.dist < b.dist; });
    std::vector<uint32_t> one;
    one.reserve(sorted.size());
    for (const auto &x : sorted) {
      one.push_back(x.id);
    }
    gt.push_back(std::move(one));
  }
  return gt;
}

int main(int argc, char **argv) {
  const std::string base_path =
      arg_value(argc, argv, "--base") ? arg_value(argc, argv, "--base")
                                      : "data/sift/sift_base.fvecs";
  const std::string query_path =
      arg_value(argc, argv, "--query") ? arg_value(argc, argv, "--query")
                                       : "data/sift/sift_query.fvecs";
  const std::string gt_path =
      arg_value(argc, argv, "--gt") ? arg_value(argc, argv, "--gt")
                                    : "data/sift/sift_groundtruth.ivecs";
  const std::string index_path =
      arg_value(argc, argv, "--index") ? arg_value(argc, argv, "--index")
                                       : "data/sift_base.eval.index";
  const std::string log_path =
      arg_value(argc, argv, "--log") ? arg_value(argc, argv, "--log")
                                     : "logs/eval_disk.log";

  const size_t base_limit = arg_size_t(argc, argv, "--base-limit", 100000);
  const size_t query_limit = arg_size_t(argc, argv, "--query-limit", 200);
  const size_t k = arg_size_t(argc, argv, "--k", 10);
  const size_t ef_search = arg_size_t(argc, argv, "--ef-search", 128);
  const uint32_t m = static_cast<uint32_t>(arg_size_t(argc, argv, "--m", 16));
  const uint32_t ef_construction =
      static_cast<uint32_t>(arg_size_t(argc, argv, "--ef-construction", 200));
  const bool rebuild = arg_bool(argc, argv, "--rebuild", true);
  const bool recompute_gt = arg_bool(argc, argv, "--recompute-gt", true);
  const std::string mode = arg_string(argc, argv, "--mode", "full");
  const bool force_disable_uring = arg_bool(argc, argv, "--force-disable-uring", false);
  const std::string agent_policy =
      arg_string(argc, argv, "--agent-policy", "");
  const std::string policy_mode_str = arg_string(argc, argv, "--policy-mode", "");
  const std::string metrics_log_arg =
      arg_string(argc, argv, "--metrics-log", "logs/eval_disk_per_query.csv");
  const std::string metrics_log = (metrics_log_arg == "none" || metrics_log_arg == "off")
                                      ? std::string()
                                      : metrics_log_arg;

  std::vector<std::vector<float>> base_all;
  std::vector<std::vector<float>> queries_all;
  std::vector<std::vector<uint32_t>> gt_all;
  if (!amio::dataset::load_fvecs(base_path, &base_all) || base_all.empty()) {
    std::fprintf(stderr, "读取 base 失败: %s\n", base_path.c_str());
    return 2;
  }
  if (!amio::dataset::load_fvecs(query_path, &queries_all) || queries_all.empty()) {
    std::fprintf(stderr, "读取 query 失败: %s\n", query_path.c_str());
    return 3;
  }
  if (!amio::dataset::load_ivecs(gt_path, &gt_all) || gt_all.empty()) {
    std::fprintf(stderr, "读取 groundtruth 失败: %s\n", gt_path.c_str());
    return 4;
  }

  const size_t nbase = std::min(base_limit, base_all.size());
  const size_t nq = std::min({query_limit, queries_all.size(), gt_all.size()});
  std::vector<std::vector<float>> base(base_all.begin(), base_all.begin() + nbase);

  if (rebuild) {
    std::fprintf(stderr,
                 "rebuild index: base=%zu dim=%zu m=%u ef_construction=%u -> %s\n",
                 nbase, base[0].size(), m, ef_construction, index_path.c_str());
    if (!build_index_file(base, index_path, m, ef_construction)) {
      std::fprintf(stderr, "构建索引失败: %s\n", index_path.c_str());
      return 5;
    }
  }

  std::vector<std::vector<uint32_t>> gt_cut;
  if (recompute_gt) {
    std::vector<std::vector<float>> queries;
    queries.reserve(nq);
    for (size_t i = 0; i < nq; i++) {
      queries.push_back(queries_all[i]);
    }
    gt_cut = compute_gt_from_base(base, queries, k);
  } else {
    gt_cut.reserve(nq);
    for (size_t i = 0; i < nq; i++) {
      std::vector<uint32_t> one;
      const size_t kk = std::min(k, gt_all[i].size());
      one.reserve(kk);
      for (size_t j = 0; j < kk; j++) {
        if (gt_all[i][j] < static_cast<uint32_t>(nbase)) {
          one.push_back(gt_all[i][j]);
        }
      }
      gt_cut.push_back(std::move(one));
    }
  }

  amio::AgentPolicyMode policy_mode = amio::AgentPolicyMode::Builtin;
  std::string policy_path_resolved = agent_policy;

  if (policy_mode_str.empty()) {
    if (!policy_path_resolved.empty()) {
      policy_mode = amio::AgentPolicyMode::Learned;
    }
  } else if (policy_mode_str == "learned" || policy_mode_str == "Learned") {
    policy_mode = amio::AgentPolicyMode::Learned;
    if (policy_path_resolved.empty()) {
      policy_path_resolved = "data/agent_io_policy.json";
    }
  } else if (policy_mode_str == "builtin" || policy_mode_str == "Builtin") {
    policy_mode = amio::AgentPolicyMode::Builtin;
    policy_path_resolved.clear();
  } else {
    std::fprintf(stderr, "未知 --policy-mode=%s，按 builtin 处理\n",
                 policy_mode_str.c_str());
    policy_path_resolved.clear();
  }

  amio::Config cfg;
  cfg.index_path = index_path;
  cfg.enable_wal = false;
  cfg.ef_search = ef_search;
  cfg.memtable_limit_mb = 1;
  cfg.cache_size_mb = 512;
  cfg.prefetch_depth = 1;
  cfg.enable_compaction = true;
  cfg.enable_memtable_search = true;
  cfg.force_disable_uring = force_disable_uring;
  cfg.agent_policy_mode = policy_mode;
  cfg.agent_policy_path = policy_path_resolved;

  if (mode == "baseline1") {
    // 关闭预取/缓存：贴合文档中的 Baseline-1 定义。
    cfg.cache_size_mb = 0;
    cfg.prefetch_depth = 0;
  } else if (mode == "baseline2") {
    // 仅保留磁盘 HNSW 主路径，不启用写优化相关逻辑。
    cfg.enable_wal = false;
    cfg.enable_compaction = false;
    cfg.enable_memtable_search = false;
    cfg.cache_size_mb = 0;
    cfg.prefetch_depth = 0;
  }

  amio::VectorStore store(cfg);
  if (!store.open()) {
    std::fprintf(stderr, "打开索引失败: %s\n", index_path.c_str());
    return 6;
  }

  std::fprintf(stderr,
               "eval_disk: policy_mode=%s agent_policy_path=%s "
               "metrics_log=%s\n",
               policy_mode == amio::AgentPolicyMode::Learned ? "learned" : "builtin",
               policy_path_resolved.empty() ? "(none)" : policy_path_resolved.c_str(),
               metrics_log.empty() ? "(disabled)" : metrics_log.c_str());

  std::vector<std::vector<uint32_t>> results;
  results.reserve(nq);
  std::vector<double> latencies_ms;
  latencies_ms.reserve(nq);

  std::ofstream mq;
  if (!metrics_log.empty()) {
    const std::filesystem::path mqp(metrics_log);
    if (!mqp.parent_path().empty()) {
      std::error_code ec;
      std::filesystem::create_directories(mqp.parent_path(), ec);
    }
    mq.open(metrics_log, std::ios::out | std::ios::trunc);
    if (!mq) {
      std::fprintf(stderr, "无法写入逐查询指标: %s\n", metrics_log.c_str());
      return 8;
    }
    mq << "query_idx,latency_ms,recall_at_k,disk_sync_block_reads,"
          "disk_sync_read_bytes,disk_via_uring,disk_via_pread,"
          "prefetch_blocks_submitted,cache_hits,cache_misses\n";
  }

  uint64_t sum_disk_blocks = 0;
  uint64_t sum_disk_bytes = 0;
  uint64_t sum_prefetch = 0;

  uint64_t t0_ns = amio::util::now_ns();
  for (size_t i = 0; i < nq; i++) {
    store.reset_search_observability();
    const uint64_t q0 = amio::util::now_ns();
    auto found = store.search_disk(queries_all[i], k);
    const uint64_t q1 = amio::util::now_ns();
    const double lat_ms = static_cast<double>(q1 - q0) / 1e6;
    latencies_ms.push_back(lat_ms);
    std::vector<uint32_t> ids;
    ids.reserve(found.size());
    for (const auto &x : found) {
      ids.push_back(static_cast<uint32_t>(x.id));
    }
    const amio::IoMetricsSnapshot snap = store.io_metrics_snapshot();
    sum_disk_blocks += snap.disk_sync_block_reads;
    sum_disk_bytes += snap.disk_sync_read_bytes;
    sum_prefetch += snap.prefetch_blocks_submitted;
    const uint64_t ch = store.cache_hits_total();
    const uint64_t cm = store.cache_misses_total();
    const double r1 =
        i < gt_cut.size() ? recall_one_query(ids, gt_cut[i], k) : 0.0;
    results.push_back(std::move(ids));
    if (mq) {
      mq << i << ',' << std::fixed << std::setprecision(6) << lat_ms << ',' << r1
         << ',' << snap.disk_sync_block_reads << ',' << snap.disk_sync_read_bytes << ','
         << snap.disk_sync_via_uring << ',' << snap.disk_sync_via_pread << ','
         << snap.prefetch_blocks_submitted << ',' << ch << ',' << cm << '\n';
    }
  }
  uint64_t t1_ns = amio::util::now_ns();

  const double recall = amio::dataset::compute_recall_at_k(results, gt_cut, k);
  const double total_ms = static_cast<double>(t1_ns - t0_ns) / 1e6;
  const double avg_ms = nq == 0 ? 0.0 : (total_ms / static_cast<double>(nq));
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  if (!latencies_ms.empty()) {
    std::sort(latencies_ms.begin(), latencies_ms.end());
    const size_t i50 = (latencies_ms.size() - 1) * 50 / 100;
    const size_t i95 = (latencies_ms.size() - 1) * 95 / 100;
    const size_t i99 = (latencies_ms.size() - 1) * 99 / 100;
    p50_ms = latencies_ms[i50];
    p95_ms = latencies_ms[i95];
    p99_ms = latencies_ms[i99];
  }
  const double qps = total_ms > 0.0 ? (static_cast<double>(nq) * 1000.0 / total_ms) : 0.0;

  const std::filesystem::path logp(log_path);
  if (!logp.parent_path().empty()) {
    std::error_code ec;
    std::filesystem::create_directories(logp.parent_path(), ec);
  }
  std::ofstream ofs(log_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    std::fprintf(stderr, "写日志失败: %s\n", log_path.c_str());
    return 7;
  }

  ofs << "eval_disk\n";
  ofs << "base_path=" << base_path << "\n";
  ofs << "query_path=" << query_path << "\n";
  ofs << "gt_path=" << gt_path << "\n";
  ofs << "index_path=" << index_path << "\n";
  ofs << "base_size=" << nbase << "\n";
  ofs << "query_size=" << nq << "\n";
  ofs << "mode=" << mode << "\n";
  ofs << "policy_mode="
      << (policy_mode == amio::AgentPolicyMode::Learned ? "learned" : "builtin")
      << "\n";
  ofs << "agent_policy_cli=" << agent_policy << "\n";
  ofs << "agent_policy_file_applied=" << policy_path_resolved << "\n";
  ofs << "per_query_metrics_csv=" << metrics_log << "\n";
  const auto &epol = store.effective_io_policy();
  ofs << "effective_agent_profile=" << epol.agent_profile << "\n";
  ofs << "effective_prefetch_depth_upper=" << epol.prefetch_depth_upper << "\n";
  ofs << "effective_prefetch_depth_layer0=" << epol.prefetch_depth_layer0 << "\n";
  ofs << "effective_max_neighbor_fanout_layer0=" << epol.max_neighbor_fanout_layer0
      << "\n";
  ofs << "effective_use_layer_aware_prefetch=" << (epol.use_layer_aware_prefetch ? 1 : 0)
      << "\n";
  ofs << "effective_sort_prefetch_by_disk_offset="
      << (epol.sort_prefetch_by_disk_offset ? 1 : 0) << "\n";
  ofs << "effective_hot_insert_min_prior_misses=" << epol.hot_insert_min_prior_misses
      << "\n";
  ofs << "io_uring_active=" << (store.io_uring_active() ? 1 : 0) << "\n";
  ofs << "k=" << k << "\n";
  ofs << "ef_search=" << ef_search << "\n";
  ofs << "m=" << m << "\n";
  ofs << "ef_construction=" << ef_construction << "\n";
  ofs << "recompute_gt=" << (recompute_gt ? 1 : 0) << "\n";
  ofs << "recall_at_k=" << recall << "\n";
  ofs << "total_ms=" << total_ms << "\n";
  ofs << "avg_latency_ms=" << avg_ms << "\n";
  ofs << "p50_latency_ms=" << p50_ms << "\n";
  ofs << "p95_latency_ms=" << p95_ms << "\n";
  ofs << "p99_latency_ms=" << p99_ms << "\n";
  ofs << "qps=" << qps << "\n";
  ofs << "sum_disk_sync_block_reads=" << sum_disk_blocks << "\n";
  ofs << "sum_disk_sync_read_bytes=" << sum_disk_bytes << "\n";
  ofs << "sum_prefetch_blocks_submitted=" << sum_prefetch << "\n";
  ofs << "avg_disk_sync_blocks_per_query="
      << (nq ? static_cast<double>(sum_disk_blocks) / static_cast<double>(nq) : 0.0)
      << "\n";
  ofs << "avg_prefetch_blocks_submitted_per_query="
      << (nq ? static_cast<double>(sum_prefetch) / static_cast<double>(nq) : 0.0)
      << "\n";
  if (!results.empty()) {
    ofs << "first_query_result_ids=";
    for (size_t i = 0; i < results[0].size(); i++) {
      if (i)
        ofs << ",";
      ofs << results[0][i];
    }
    ofs << "\n";
  }
  if (!gt_cut.empty()) {
    ofs << "first_query_gt_ids=";
    const size_t kk = std::min(k, gt_cut[0].size());
    for (size_t i = 0; i < kk; i++) {
      if (i)
        ofs << ",";
      ofs << gt_cut[0][i];
    }
    ofs << "\n";
  }
  ofs.close();

  std::printf(
      "done: mode=%s policy=%s recall@%zu=%.6f qps=%.3f total_ms=%.3f avg_ms=%.6f "
      "p50_ms=%.6f p95_ms=%.6f p99_ms=%.6f io_uring=%d log=%s metrics_csv=%s\n",
      mode.c_str(),
      policy_mode == amio::AgentPolicyMode::Learned ? "learned" : "builtin", k, recall, qps,
      total_ms, avg_ms, p50_ms, p95_ms, p99_ms, store.io_uring_active() ? 1 : 0,
      log_path.c_str(), metrics_log.empty() ? "(none)" : metrics_log.c_str());
  return 0;
}
