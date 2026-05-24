#pragma once

// C API for the Python ctypes bridge.
// All functions use plain C types so ctypes can call them without pybind11.

#include <stdint.h>

#ifdef _WIN32
#  ifdef AMIO_PY_BUILDING
#    define AMIO_PY_API __declspec(dllexport)
#  else
#    define AMIO_PY_API __declspec(dllimport)
#  endif
#else
#  define AMIO_PY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ── Lifecycle ──────────────────────────────────────────────────────────────

/// Open (or create) a VectorStore from `index_path`.
/// Returns an opaque handle; NULL on failure.
AMIO_PY_API void *amio_store_open(const char *index_path, int ef_search,
                                  int cache_mb, int static_cache_mb);

/// Close handle and free all resources.
AMIO_PY_API void amio_store_close(void *handle);

// ── Search / Insert ────────────────────────────────────────────────────────

/// Disk-based approximate k-NN search.
/// query: float array of length dim.
/// out_ids / out_dists: caller-allocated arrays of length k.
/// Returns the number of results written (≤ k); -1 on error.
AMIO_PY_API int amio_store_search_disk(void *handle, const float *query, int dim,
                                       int k, uint64_t *out_ids, float *out_dists);

/// In-memory HNSW search (no disk I/O).
AMIO_PY_API int amio_store_search_mem(void *handle, const float *query, int dim,
                                      int k, uint64_t *out_ids, float *out_dists);

/// Insert a vector. Returns 1 on success, 0 on failure.
AMIO_PY_API int amio_store_insert(void *handle, uint64_t id, const float *vec, int dim);

// ── Observability ──────────────────────────────────────────────────────────

typedef struct {
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t disk_reads;
  uint64_t prefetch_submitted;
  uint64_t theta_phase_switches;
  uint64_t useful_prefetches;
  uint64_t wasted_prefetches;
  uint64_t static_pins;
  int      uring_active;
} AmioMetrics;

AMIO_PY_API void amio_store_metrics(void *handle, AmioMetrics *out);
AMIO_PY_API void amio_store_reset_metrics(void *handle);

/// Returns the partition profile name string (e.g. "DISK_FIRST").
AMIO_PY_API const char *amio_store_partition_profile(void *handle);

// ── Vector file utilities ──────────────────────────────────────────────────

/// Detect vector file kind: 0=unknown, 1=fvecs, 2=bvecs, 3=ivecs.
AMIO_PY_API int amio_detect_vector_kind(const char *path);

/// Probe vector file: write dim and num_vectors via output pointers.
/// Returns 1 on success, 0 on failure.
AMIO_PY_API int amio_probe_vector_file(const char *path, uint32_t *dim_out,
                                       uint64_t *num_vectors_out);

#ifdef __cplusplus
}
#endif
