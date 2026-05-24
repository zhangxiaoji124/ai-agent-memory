#include "write/index_merger.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include "dataset/index_build_common.h"
#include "index/external_vectors.h"

namespace amio::write {

namespace {

static bool raw_copy_range(int src_fd, int dst_fd, uint64_t off, uint64_t len) {
  std::vector<uint8_t> buf(static_cast<size_t>(std::min<uint64_t>(len, 1 << 20)));
  uint64_t done = 0;
  while (done < len) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(len - done, buf.size()));
#if defined(_WIN32)
    if (_lseeki64(src_fd, static_cast<__int64>(off + done), SEEK_SET) < 0) {
      return false;
    }
    if (_read(src_fd, reinterpret_cast<char *>(buf.data()), static_cast<unsigned int>(chunk)) !=
        static_cast<int>(chunk)) {
      return false;
    }
    if (_lseeki64(dst_fd, static_cast<__int64>(off + done), SEEK_SET) < 0) {
      return false;
    }
    if (_write(dst_fd, reinterpret_cast<const char *>(buf.data()),
               static_cast<unsigned int>(chunk)) != static_cast<int>(chunk)) {
      return false;
    }
#else
    if (::pread(src_fd, buf.data(), chunk, static_cast<off_t>(off + done)) !=
        static_cast<ssize_t>(chunk)) {
      return false;
    }
    if (::pwrite(dst_fd, buf.data(), chunk, static_cast<off_t>(off + done)) !=
        static_cast<ssize_t>(chunk)) {
      return false;
    }
#endif
    done += chunk;
  }
  return true;
}

} // namespace

IndexMergeStats merge_batch_into_index(index::IndexFile &file, index::IndexFileHeader &header,
                                       index::HnswIndex &graph,
                                       const CompactionWorker::Batch &batch,
                                       RemapComTracker *remap) {
  IndexMergeStats st{};
  if (file.fd() < 0 || batch.empty() || header.dim == 0) {
    return st;
  }

  auto sorted = batch;
  std::sort(sorted.begin(), sorted.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  const uint64_t base = header.total_nodes;
  const uint32_t dim = header.dim;

  for (const auto &kv : sorted) {
    if (kv.first != graph.id_base() + graph.size() || kv.second.size() != dim) {
      continue;
    }
    graph.insert(kv.first, kv.second);
  }

  const uint64_t new_total = graph.size();
  if (new_total <= base) {
    st.ok = true;
    return st;
  }

  if (dim > index::kMaxInlineVectorDim && header.vector_section_offset == 0) {
    header.version = index::kIndexVersion2;
    header.vector_section_offset = index::default_vector_section_offset(new_total);
    header.vector_stride_bytes = static_cast<uint32_t>(dim * sizeof(float));
    header.vector_encoding = static_cast<uint8_t>(index::VectorEncoding::Float32);
  }

  for (uint64_t id = base; id < new_total; id++) {
    const auto &vec = graph.vector_at(id);
    if (!dataset::write_single_node(file, &header, graph, dim, id, vec)) {
      return st;
    }
    st.blocks_written++;
    if (remap) {
      remap->record_remap(id, index::node_offset(id));
    }
  }

  header.total_nodes = new_total;
  const uint64_t disk_base = graph.id_base();
  if (graph.max_layer() >= static_cast<int>(header.max_layer)) {
    header.max_layer =
        static_cast<uint8_t>(std::min(255, std::max(0, graph.max_layer())));
    header.entry_point = disk_base + graph.entry_point();
  }
  if (!file.write_header(header)) {
    return st;
  }

  st.nodes_merged = new_total - base;
  st.blocks_remapped = remap ? remap->remapped_blocks() : 0;
  st.ok = true;
  return st;
}

bool compact_index_with_remap(const std::string &src_path, const std::string &dst_path,
                              std::string *report) {
  index::IndexFile src;
  if (!src.open_readonly(src_path)) {
    return false;
  }
  index::IndexFileHeader hdr{};
  if (!src.read_header(&hdr) || hdr.total_nodes == 0) {
    return false;
  }

  index::IndexFile dst;
  if (!dst.open_create_trunc(dst_path)) {
    return false;
  }

  RemapComTracker remap;
  uint64_t written = 0;
  uint64_t skipped = 0;

  for (uint64_t id = 0; id < hdr.total_nodes; id++) {
    index::NodeBlock b{};
    if (!src.pread_node(id, &b)) {
      return false;
    }
    index::NodeBlock existing{};
    const bool had = dst.pread_node(id, &existing);
    if (had && remap.classify(existing, b) == BlockDisposition::Unchanged) {
      remap.record_remap(id, index::node_offset(id));
      skipped++;
      continue;
    }
    if (!dst.pwrite_node(id, b)) {
      return false;
    }
    written++;
  }

  if (!dst.write_header(hdr)) {
    return false;
  }

  if (index::header_uses_external_vectors(hdr)) {
    const uint64_t bytes = static_cast<uint64_t>(hdr.vector_stride_bytes) * hdr.total_nodes;
    if (!raw_copy_range(src.fd(), dst.fd(), hdr.vector_section_offset, bytes)) {
      return false;
    }
  }

  if (report) {
    std::ostringstream oss;
    oss << "nodes=" << hdr.total_nodes << " written=" << written << " udb_skipped=" << skipped
        << " remapped=" << remap.remapped_blocks();
    *report = oss.str();
  }
  std::fprintf(stderr, "compact_index_with_remap: %s -> %s (%s)\n", src_path.c_str(),
               dst_path.c_str(), report ? report->c_str() : "");
  return true;
}

} // namespace amio::write
