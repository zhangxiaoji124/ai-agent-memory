#include "runtime/memory_partition.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "vector_store.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace amio::runtime {

namespace {

constexpr uint64_t k100GB = 100ull * 1024 * 1024 * 1024;
constexpr uint64_t k400GB = 400ull * 1024 * 1024 * 1024;

uint64_t physical_memory_bytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX st{};
  st.dwLength = sizeof(st);
  if (GlobalMemoryStatusEx(&st)) {
    return static_cast<uint64_t>(st.ullTotalPhys);
  }
  return 16ull * 1024 * 1024 * 1024;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long psz = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && psz > 0) {
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(psz);
  }
  return 16ull * 1024 * 1024 * 1024;
#endif
}

void split_pool(MemoryPartitionProfile p, uint64_t ram, MemoryPoolBudget *out) {
  double mt = 0.10, st = 0.20, dy = 0.70;
  if (p == MemoryPartitionProfile::HostResident) {
    mt = 0.10;
    st = 0.15;
    dy = 0.75;
  }
  out->memtable_bytes = static_cast<uint64_t>(ram * mt);
  out->static_cache_bytes = static_cast<uint64_t>(ram * st);
  out->dynamic_cache_bytes = ram - out->memtable_bytes - out->static_cache_bytes;
}

MemoryPartitionProfile classify(VectorFileKind kind, uint64_t data_bytes, uint64_t ram,
                                double rho) {
  if (kind == VectorFileKind::Bvecs) {
    if (data_bytes >= k400GB && ram >= 80ull * 1024 * 1024 * 1024 &&
        ram <= 128ull * 1024 * 1024 * 1024) {
      return MemoryPartitionProfile::BvecUltra500G100G;
    }
    if (data_bytes >= 100ull * 1024 * 1024 * 1024 || rho >= 3.0) {
      return MemoryPartitionProfile::BvecDiskTiered;
    }
    return MemoryPartitionProfile::DiskFirst;
  }
  if (kind == VectorFileKind::Fvecs) {
    if (data_bytes <= ram / 2) {
      return MemoryPartitionProfile::HostResident;
    }
    if (data_bytes <= ram * 2) {
      return MemoryPartitionProfile::HybridFloat;
    }
    return MemoryPartitionProfile::DiskFirst;
  }
  return MemoryPartitionProfile::DiskFirst;
}

void tune_profile_hints(MemoryPartitionProfile p, PartitionDecision *d) {
  switch (p) {
  case MemoryPartitionProfile::HostResident:
    d->allow_full_memory_graph = true;
    d->partition_theta = 0.0f;
    d->distance_kernel = "l2_f32";
    d->ef_search_hint = 128;
    d->prefetch_depth_hint = 1;
    d->stream_index_batch_vectors = 0;
    break;
  case MemoryPartitionProfile::HybridFloat:
    d->allow_full_memory_graph = true;
    d->partition_theta = 0.30f;
    d->distance_kernel = "l2_f32";
    d->ef_search_hint = 160;
    d->prefetch_depth_hint = 1;
    d->stream_index_batch_vectors = 1000000;
    break;
  case MemoryPartitionProfile::DiskFirst:
    d->allow_full_memory_graph = false;
    d->partition_theta = 0.35f;
    d->distance_kernel = "l2_f32";
    d->ef_search_hint = 192;
    d->prefetch_depth_hint = 2;
    d->stream_index_batch_vectors = 500000;
    break;
  case MemoryPartitionProfile::BvecDiskTiered:
    d->allow_full_memory_graph = false;
    d->partition_theta = 0.35f;
    d->distance_kernel = "u8_l2";
    d->ef_search_hint = 192;
    d->prefetch_depth_hint = 2;
    d->stream_index_batch_vectors = 500000;
    break;
  case MemoryPartitionProfile::BvecUltra500G100G:
    d->allow_full_memory_graph = false;
    d->partition_theta = 0.35f;
    d->distance_kernel = "u8_l2";
    d->ef_search_hint = 224;
    d->prefetch_depth_hint = 2;
    d->stream_index_batch_vectors = 500000;
    break;
  default:
    break;
  }
  if (d->stream_index_batch_vectors == 0 && p != MemoryPartitionProfile::HostResident) {
    d->stream_index_batch_vectors = default_stream_batch(p);
  }
}

} // namespace

uint64_t default_stream_batch(MemoryPartitionProfile p) {
  switch (p) {
  case MemoryPartitionProfile::HostResident:
    return 0;
  case MemoryPartitionProfile::HybridFloat:
    return 1000000;
  default:
    return 500000;
  }
}

VectorFileKind detect_kind_by_extension(const std::string &path) {
  if (path.size() >= 6) {
    const std::string ext = path.substr(path.size() - 6);
    if (ext == ".fvecs") {
      return VectorFileKind::Fvecs;
    }
    if (ext == ".bvecs") {
      return VectorFileKind::Bvecs;
    }
    if (ext == ".ivecs") {
      return VectorFileKind::Ivecs;
    }
  }
  return VectorFileKind::Unknown;
}

uint64_t file_size_bytes(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    return 0;
  }
  return static_cast<uint64_t>(ifs.tellg());
}

bool probe_vector_file(const std::string &path, VectorFileKind kind, uint32_t *dim_out,
                       uint64_t *num_vectors_est_out) {
  if (!dim_out || !num_vectors_est_out) {
    return false;
  }
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    return false;
  }
  uint32_t dim = 0;
  if (!ifs.read(reinterpret_cast<char *>(&dim), 4)) {
    return false;
  }
  if (dim == 0 || dim > 65535) {
    return false;
  }
  const uint64_t fsize = file_size_bytes(path);
  const uint64_t rec = 4ull + ((kind == VectorFileKind::Bvecs) ? dim : dim * 4ull);
  if (rec == 0) {
    return false;
  }
  *dim_out = dim;
  *num_vectors_est_out = fsize / rec;
  return true;
}

uint64_t resolve_ram_budget_bytes(uint64_t explicit_budget_bytes) {
  if (explicit_budget_bytes > 0) {
    return explicit_budget_bytes;
  }
  const char *env = std::getenv("AMIO_RAM_BUDGET_GB");
  if (env && env[0]) {
    const double gb = std::atof(env);
    if (gb > 0) {
      return static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
    }
  }
  const uint64_t phys = physical_memory_bytes();
  return phys * 80 / 100;
}

PartitionDecision select_memory_partition(const std::vector<std::string> &paths,
                                          uint64_t ram_budget_bytes,
                                          MemoryPartitionProfile override_profile) {
  PartitionDecision d;
  if (ram_budget_bytes == 0) {
    ram_budget_bytes = resolve_ram_budget_bytes(0);
  }

  uint64_t data_bytes = 0;
  VectorFileKind primary = VectorFileKind::Unknown;
  uint32_t dim = 0;
  uint64_t nvec = 0;

  for (const auto &p : paths) {
    if (p.empty()) {
      continue;
    }
    VectorFileKind k = detect_kind_by_extension(p);
    if (k == VectorFileKind::Ivecs) {
      continue;
    }
    if (k == VectorFileKind::Unknown) {
      k = VectorFileKind::Fvecs;
    }
    if (primary == VectorFileKind::Unknown) {
      primary = k;
    }
    data_bytes += file_size_bytes(p);
    uint32_t dprobe = 0;
    uint64_t nprobe = 0;
    if (probe_vector_file(p, k, &dprobe, &nprobe)) {
      if (dim == 0) {
        dim = dprobe;
      }
      nvec += nprobe;
    }
  }

  if (primary == VectorFileKind::Unknown) {
    primary = VectorFileKind::Fvecs;
  }

  const double rho =
      ram_budget_bytes > 0 ? static_cast<double>(data_bytes) / static_cast<double>(ram_budget_bytes)
                           : 0.0;

  d.dataset.kind = primary;
  d.dataset.data_bytes = data_bytes;
  d.dataset.dim = dim;
  d.dataset.num_vectors_est = nvec;
  d.dataset.rho = rho;

  MemoryPartitionProfile prof = override_profile;
  if (prof == MemoryPartitionProfile::Unspecified) {
    const char *envp = std::getenv("AMIO_MEMORY_PROFILE");
    if (envp && envp[0]) {
      prof = parse_profile_name(envp);
    }
    if (prof == MemoryPartitionProfile::Unspecified) {
      prof = classify(primary, data_bytes, ram_budget_bytes, rho);
    }
  }

  d.profile = prof;
  split_pool(prof, ram_budget_bytes, &d.pools);
  tune_profile_hints(prof, &d);
  return d;
}

void apply_partition_to_config(const PartitionDecision &dec, Config *cfg) {
  if (!cfg) {
    return;
  }
  cfg->partition = dec;
  cfg->ram_budget_bytes = dec.pools.memtable_bytes + dec.pools.static_cache_bytes +
                          dec.pools.dynamic_cache_bytes;
  if (cfg->ram_budget_bytes == 0) {
    cfg->ram_budget_bytes = resolve_ram_budget_bytes(0);
  }
  cfg->memtable_limit_mb = std::max<size_t>(1, dec.pools.memtable_bytes / (1024 * 1024));
  cfg->static_cache_mb = std::max<size_t>(1, dec.pools.static_cache_bytes / (1024 * 1024));
  cfg->cache_size_mb = std::max<size_t>(1, dec.pools.dynamic_cache_bytes / (1024 * 1024));
  cfg->enable_static_subgraph_pin =
      dec.profile == MemoryPartitionProfile::BvecDiskTiered ||
      dec.profile == MemoryPartitionProfile::BvecUltra500G100G ||
      dec.profile == MemoryPartitionProfile::DiskFirst ||
      dec.profile == MemoryPartitionProfile::HybridFloat;
}

const char *profile_name(MemoryPartitionProfile p) {
  switch (p) {
  case MemoryPartitionProfile::HostResident:
    return "HOST_RESIDENT";
  case MemoryPartitionProfile::HybridFloat:
    return "HYBRID_FLOAT";
  case MemoryPartitionProfile::DiskFirst:
    return "DISK_FIRST";
  case MemoryPartitionProfile::BvecDiskTiered:
    return "BVEC_DISK_TIERED";
  case MemoryPartitionProfile::BvecUltra500G100G:
    return "BVEC_ULTRA_500G_100G";
  default:
    return "UNSPECIFIED";
  }
}

const char *vector_kind_name(VectorFileKind k) {
  switch (k) {
  case VectorFileKind::Fvecs:
    return "fvecs";
  case VectorFileKind::Bvecs:
    return "bvecs";
  case VectorFileKind::Ivecs:
    return "ivecs";
  default:
    return "unknown";
  }
}

MemoryPartitionProfile parse_profile_name(const char *name) {
  if (!name || !name[0]) {
    return MemoryPartitionProfile::Unspecified;
  }
  if (std::strcmp(name, "HOST_RESIDENT") == 0 || std::strcmp(name, "M0") == 0) {
    return MemoryPartitionProfile::HostResident;
  }
  if (std::strcmp(name, "HYBRID_FLOAT") == 0 || std::strcmp(name, "M1") == 0) {
    return MemoryPartitionProfile::HybridFloat;
  }
  if (std::strcmp(name, "DISK_FIRST") == 0 || std::strcmp(name, "M2") == 0) {
    return MemoryPartitionProfile::DiskFirst;
  }
  if (std::strcmp(name, "BVEC_DISK_TIERED") == 0 || std::strcmp(name, "M3") == 0) {
    return MemoryPartitionProfile::BvecDiskTiered;
  }
  if (std::strcmp(name, "BVEC_ULTRA_500G_100G") == 0 || std::strcmp(name, "M4") == 0) {
    return MemoryPartitionProfile::BvecUltra500G100G;
  }
  return MemoryPartitionProfile::Unspecified;
}

} // namespace amio::runtime
