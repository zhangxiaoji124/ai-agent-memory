#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace amio {
struct Config;
}

namespace amio::runtime {

enum class VectorFileKind {
  Unknown = 0,
  Fvecs,
  Bvecs,
  Ivecs,
};

enum class MemoryPartitionProfile {
  Unspecified = 0, // 由区分器自动判定
  HostResident,
  HybridFloat,
  DiskFirst,
  BvecDiskTiered,
  BvecUltra500G100G,
};

struct MemoryPoolBudget {
  uint64_t memtable_bytes = 0;
  uint64_t static_cache_bytes = 0;
  uint64_t dynamic_cache_bytes = 0;
};

struct DatasetDescriptor {
  VectorFileKind kind = VectorFileKind::Unknown;
  uint64_t data_bytes = 0;
  uint32_t dim = 0;
  uint64_t num_vectors_est = 0;
  double rho = 0.0;
};

struct PartitionDecision {
  MemoryPartitionProfile profile = MemoryPartitionProfile::DiskFirst;
  MemoryPoolBudget pools{};
  DatasetDescriptor dataset{};
  bool allow_full_memory_graph = false;
  uint64_t stream_index_batch_vectors = 0;
  float partition_theta = 0.35f;
  const char *distance_kernel = "l2_f32";
  size_t ef_search_hint = 128;
  size_t prefetch_depth_hint = 1;
};

uint64_t default_stream_batch(MemoryPartitionProfile p);

VectorFileKind detect_kind_by_extension(const std::string &path);
VectorFileKind detect_kind_by_content(const std::string &path);
VectorFileKind detect_vector_file_kind(const std::string &path);

bool probe_vector_file(const std::string &path, VectorFileKind kind, uint32_t *dim_out,
                       uint64_t *num_vectors_est_out);

uint64_t file_size_bytes(const std::string &path);

/// 解析可用内存预算：显式参数 > 环境变量 AMIO_RAM_BUDGET_GB > 物理内存 80%。
uint64_t resolve_ram_budget_bytes(uint64_t explicit_budget_bytes);

PartitionDecision select_memory_partition(const std::vector<std::string> &paths,
                                          uint64_t ram_budget_bytes,
                                          MemoryPartitionProfile override_profile =
                                              MemoryPartitionProfile::Unspecified);

void apply_partition_to_config(const PartitionDecision &dec, Config *cfg);

const char *profile_name(MemoryPartitionProfile p);
const char *vector_kind_name(VectorFileKind k);

MemoryPartitionProfile parse_profile_name(const char *name);

} // namespace amio::runtime
