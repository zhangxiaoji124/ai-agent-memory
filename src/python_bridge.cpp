#define AMIO_PY_BUILDING
#include "python_bridge.h"

#include <cstring>
#include <memory>

#include "runtime/memory_partition.h"
#include "vector_store.h"

// VectorStore is not copyable; wrap in a heap-allocated struct.
struct StoreHandle {
  amio::Config cfg;
  std::unique_ptr<amio::VectorStore> store;
};

extern "C" {

void *amio_store_open(const char *index_path, int ef_search, int cache_mb,
                      int static_cache_mb) {
  if (!index_path)
    return nullptr;
  auto *h = new (std::nothrow) StoreHandle{};
  if (!h)
    return nullptr;

  h->cfg.index_path = index_path;
  h->cfg.ef_search = (ef_search > 0) ? static_cast<size_t>(ef_search) : 128;
  h->cfg.cache_size_mb = (cache_mb > 0) ? static_cast<size_t>(cache_mb) : 512;
  h->cfg.static_cache_mb =
      (static_cache_mb > 0) ? static_cast<size_t>(static_cache_mb) : 128;
  h->cfg.enable_wal = false;        // read-only usage from Python by default
  h->cfg.enable_compaction = false;

  h->store = std::make_unique<amio::VectorStore>(h->cfg);
  if (!h->store->open()) {
    delete h;
    return nullptr;
  }
  return h;
}

void amio_store_close(void *handle) {
  delete static_cast<StoreHandle *>(handle);
}

int amio_store_search_disk(void *handle, const float *query, int dim, int k,
                           uint64_t *out_ids, float *out_dists) {
  if (!handle || !query || k <= 0 || !out_ids || !out_dists)
    return -1;
  auto *h = static_cast<StoreHandle *>(handle);
  std::vector<float> q(query, query + dim);
  const auto results = h->store->search_disk(q, static_cast<size_t>(k));
  const int n = static_cast<int>(results.size());
  for (int i = 0; i < n; i++) {
    out_ids[i] = results[static_cast<size_t>(i)].id;
    out_dists[i] = results[static_cast<size_t>(i)].distance;
  }
  return n;
}

int amio_store_search_mem(void *handle, const float *query, int dim, int k,
                          uint64_t *out_ids, float *out_dists) {
  if (!handle || !query || k <= 0 || !out_ids || !out_dists)
    return -1;
  auto *h = static_cast<StoreHandle *>(handle);
  std::vector<float> q(query, query + dim);
  const auto results = h->store->search(q, static_cast<size_t>(k));
  const int n = static_cast<int>(results.size());
  for (int i = 0; i < n; i++) {
    out_ids[i] = results[static_cast<size_t>(i)].id;
    out_dists[i] = results[static_cast<size_t>(i)].distance;
  }
  return n;
}

int amio_store_insert(void *handle, uint64_t id, const float *vec, int dim) {
  if (!handle || !vec || dim <= 0)
    return 0;
  auto *h = static_cast<StoreHandle *>(handle);
  // Re-enable compaction/WAL if inserts are used
  std::vector<float> v(vec, vec + dim);
  return h->store->insert(id, v) ? 1 : 0;
}

void amio_store_metrics(void *handle, AmioMetrics *out) {
  if (!handle || !out)
    return;
  std::memset(out, 0, sizeof(*out));
  auto *h = static_cast<StoreHandle *>(handle);
  const amio::IoMetricsSnapshot snap = h->store->io_metrics_snapshot();
  out->cache_hits = h->store->cache_hits_total();
  out->cache_misses = h->store->cache_misses_total();
  out->disk_reads = snap.disk_sync_block_reads;
  out->prefetch_submitted = snap.prefetch_blocks_submitted;
  out->theta_phase_switches = snap.theta_phase_switches;
  out->useful_prefetches = snap.useful_prefetch_demand_hits;
  out->wasted_prefetches = snap.wasted_prefetch_evictions;
  out->static_pins = h->store->static_pins_count();
  out->uring_active = h->store->io_uring_active() ? 1 : 0;
}

void amio_store_reset_metrics(void *handle) {
  if (!handle)
    return;
  static_cast<StoreHandle *>(handle)->store->reset_search_observability();
}

const char *amio_store_partition_profile(void *handle) {
  if (!handle)
    return "UNKNOWN";
  auto *h = static_cast<StoreHandle *>(handle);
  return amio::runtime::profile_name(h->store->partition().profile);
}

int amio_detect_vector_kind(const char *path) {
  if (!path)
    return 0;
  return static_cast<int>(amio::runtime::detect_vector_file_kind(path));
}

int amio_probe_vector_file(const char *path, uint32_t *dim_out, uint64_t *num_vectors_out) {
  if (!path || !dim_out || !num_vectors_out)
    return 0;
  const amio::runtime::VectorFileKind kind =
      amio::runtime::detect_vector_file_kind(path);
  return amio::runtime::probe_vector_file(path, kind, dim_out, num_vectors_out) ? 1 : 0;
}

} // extern "C"
