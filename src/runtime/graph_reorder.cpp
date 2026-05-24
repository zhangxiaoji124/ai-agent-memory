#include "runtime/graph_reorder.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "index/storage_layout.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace amio::runtime {

namespace {

// 用 pread 从任意偏移读取（跨平台）
static bool raw_pread(int fd, uint64_t off, void *buf, size_t len) {
  uint8_t *p = static_cast<uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
#if defined(_WIN32)
    if (_lseeki64(fd, static_cast<__int64>(off + done), SEEK_SET) < 0)
      return false;
    int r = _read(fd, p + done, static_cast<unsigned int>(len - done));
#else
    ssize_t r = ::pread(fd, p + done, len - done, static_cast<off_t>(off + done));
#endif
    if (r <= 0)
      return false;
    done += static_cast<size_t>(r);
  }
  return true;
}

static bool raw_pwrite(int fd, uint64_t off, const void *buf, size_t len) {
  const uint8_t *p = static_cast<const uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
#if defined(_WIN32)
    if (_lseeki64(fd, static_cast<__int64>(off + done), SEEK_SET) < 0)
      return false;
    int r = _write(fd, p + done, static_cast<unsigned int>(len - done));
#else
    ssize_t r = ::pwrite(fd, p + done, len - done, static_cast<off_t>(off + done));
#endif
    if (r <= 0)
      return false;
    done += static_cast<size_t>(r);
  }
  return true;
}

// 计算 layer0 邻居对的平均 |new_id_a - new_id_b|（局部性指标，越小越好）
static double compute_layer0_locality(const std::vector<index::NodeBlock> &blocks,
                                      const std::vector<uint64_t> &old_to_new,
                                      uint64_t N) {
  uint64_t total_dist = 0;
  uint64_t count = 0;
  for (uint64_t new_id = 0; new_id < N; new_id++) {
    const auto &b = blocks[static_cast<size_t>(new_id)];
    const uint8_t cnt = b.neighbor_counts[0];
    const size_t n = cnt > 32 ? 32 : cnt;
    for (size_t j = 0; j < n; j++) {
      const uint64_t nb_old = static_cast<uint64_t>(b.neighbors[0][j]);
      if (nb_old >= N)
        continue;
      const uint64_t nb_new = old_to_new[static_cast<size_t>(nb_old)];
      const uint64_t d = nb_new > new_id ? nb_new - new_id : new_id - nb_new;
      total_dist += d;
      count++;
    }
  }
  if (count == 0)
    return 0.0;
  return static_cast<double>(total_dist) / static_cast<double>(count);
}

} // namespace

bool reorder_index_bfs(const std::string &src_path, const std::string &dst_path,
                       std::string *locality_report) {
  // ─── 1. 打开并读取源索引 ───
  index::IndexFile src;
  if (!src.open_readonly(src_path)) {
    std::fprintf(stderr, "reorder_index_bfs: 无法打开源索引 %s\n", src_path.c_str());
    return false;
  }
  index::IndexFileHeader hdr{};
  if (!src.read_header(&hdr)) {
    std::fprintf(stderr, "reorder_index_bfs: 读取头部失败\n");
    return false;
  }
  const uint64_t N = hdr.total_nodes;
  if (N == 0) {
    std::fprintf(stderr, "reorder_index_bfs: 索引为空\n");
    return false;
  }

  std::fprintf(stderr, "reorder_index_bfs: nodes=%llu dim=%u v%u\n",
               static_cast<unsigned long long>(N), hdr.dim, hdr.version);

  // ─── 2. 读入所有 NodeBlock ───
  std::vector<index::NodeBlock> blocks(static_cast<size_t>(N));
  for (uint64_t id = 0; id < N; id++) {
    if (!src.pread_node(id, &blocks[static_cast<size_t>(id)])) {
      std::fprintf(stderr, "reorder_index_bfs: pread_node %llu 失败\n",
                   static_cast<unsigned long long>(id));
      return false;
    }
  }

  // ─── 3. BFS 遍历建立访问顺序 ───
  // 优先展开 layer0（空间最密集），再展开上层边，保证 layer0 邻居物理上相邻。
  std::vector<uint64_t> order;
  order.reserve(static_cast<size_t>(N));
  std::unordered_set<uint64_t> visited;
  visited.reserve(static_cast<size_t>(N));
  std::queue<uint64_t> q;

  const uint64_t ep = hdr.entry_point < N ? hdr.entry_point : 0;
  q.push(ep);
  visited.insert(ep);
  order.push_back(ep);

  while (!q.empty()) {
    const uint64_t cur = q.front();
    q.pop();
    const auto &b = blocks[static_cast<size_t>(cur)];
    // 先展开 layer0（最重要的局部性），再展开上层
    for (int lyr = 0; lyr <= static_cast<int>(hdr.max_layer) && lyr < 8; lyr++) {
      const uint8_t cnt = b.neighbor_counts[static_cast<size_t>(lyr)];
      const size_t n = cnt > 32 ? 32 : cnt;
      for (size_t j = 0; j < n; j++) {
        const uint64_t nb = static_cast<uint64_t>(b.neighbors[static_cast<size_t>(lyr)][j]);
        if (nb >= N)
          continue;
        if (visited.insert(nb).second) {
          order.push_back(nb);
          q.push(nb);
        }
      }
    }
  }

  // 将未被 BFS 访问的孤立节点追加
  for (uint64_t id = 0; id < N; id++) {
    if (visited.find(id) == visited.end()) {
      order.push_back(id);
    }
  }

  // ─── 4. 建立 old→new 映射 ───
  // old_to_new[old_id] = new_id
  std::vector<uint64_t> old_to_new(static_cast<size_t>(N));
  for (uint64_t new_id = 0; new_id < N; new_id++) {
    old_to_new[static_cast<size_t>(order[static_cast<size_t>(new_id)])] = new_id;
  }

  // 局部性指标（重排前）：使用原始 old_id 作为 new_id（恒等映射）进行计算
  double locality_before = 0.0;
  if (locality_report) {
    // 以原始 id 顺序（老映射 = 恒等）计算平均 layer0 邻居 id 距离
    std::vector<uint64_t> identity(static_cast<size_t>(N));
    for (uint64_t i = 0; i < N; i++)
      identity[static_cast<size_t>(i)] = i;
    locality_before = compute_layer0_locality(blocks, identity, N);
  }

  // ─── 5. 就地重写 blocks 的 id 与邻居映射 ───
  // 构建新 blocks 数组（按 new_id 排列）
  std::vector<index::NodeBlock> new_blocks(static_cast<size_t>(N));
  for (uint64_t new_id = 0; new_id < N; new_id++) {
    const uint64_t old_id = order[static_cast<size_t>(new_id)];
    index::NodeBlock nb = blocks[static_cast<size_t>(old_id)];
    nb.node_id = new_id;
    for (int lyr = 0; lyr < 8; lyr++) {
      const uint8_t cnt = nb.neighbor_counts[static_cast<size_t>(lyr)];
      const size_t nn = cnt > 32 ? 32 : cnt;
      for (size_t j = 0; j < nn; j++) {
        const uint64_t old_nb =
            static_cast<uint64_t>(nb.neighbors[static_cast<size_t>(lyr)][j]);
        if (old_nb < N) {
          const uint64_t new_nb = old_to_new[static_cast<size_t>(old_nb)];
          nb.neighbors[static_cast<size_t>(lyr)][j] = static_cast<uint32_t>(new_nb);
          nb.neighbor_offsets[static_cast<size_t>(lyr)][j] = index::node_offset(new_nb);
        }
      }
    }
    new_blocks[static_cast<size_t>(new_id)] = nb;
  }

  // 局部性指标（重排后）
  double locality_after = 0.0;
  if (locality_report) {
    locality_after = compute_layer0_locality(new_blocks, old_to_new, N);
    std::ostringstream oss;
    oss << "avg_layer0_neighbor_id_dist_before=" << locality_before << "\n";
    oss << "avg_layer0_neighbor_id_dist_after=" << locality_after << "\n";
    if (locality_before > 0.0)
      oss << "locality_improvement_ratio="
          << (locality_before - locality_after) / locality_before << "\n";
    *locality_report = oss.str();
  }

  // ─── 6. 处理 v2 外置向量区 ───
  const bool has_ext = index::header_uses_external_vectors(hdr);
  std::vector<uint8_t> vec_section;
  if (has_ext && hdr.vector_stride_bytes > 0) {
    const uint64_t vec_total = N * static_cast<uint64_t>(hdr.vector_stride_bytes);
    vec_section.resize(static_cast<size_t>(vec_total));
    if (!raw_pread(src.fd(), hdr.vector_section_offset, vec_section.data(),
                   static_cast<size_t>(vec_total))) {
      std::fprintf(stderr, "reorder_index_bfs: 读取向量区失败\n");
      return false;
    }
    // 重排：new_vec[new_id] = old_vec[old_id]
    std::vector<uint8_t> new_vec_section(static_cast<size_t>(vec_total));
    const size_t stride = static_cast<size_t>(hdr.vector_stride_bytes);
    for (uint64_t new_id = 0; new_id < N; new_id++) {
      const uint64_t old_id = order[static_cast<size_t>(new_id)];
      std::memcpy(new_vec_section.data() + new_id * stride,
                  vec_section.data() + old_id * stride, stride);
    }
    vec_section = std::move(new_vec_section);
  }

  // ─── 7. 写出目标文件 ───
  index::IndexFile dst;
  if (!dst.open_create_trunc(dst_path)) {
    std::fprintf(stderr, "reorder_index_bfs: 无法创建目标文件 %s\n", dst_path.c_str());
    return false;
  }

  // 更新头部：entry_point 变为 BFS 第一个节点 = new_id 0
  index::IndexFileHeader new_hdr = hdr;
  new_hdr.entry_point = old_to_new[static_cast<size_t>(ep)]; // = 0

  if (!dst.write_header(new_hdr)) {
    std::fprintf(stderr, "reorder_index_bfs: 写头部失败\n");
    return false;
  }

  for (uint64_t new_id = 0; new_id < N; new_id++) {
    if (!dst.pwrite_node(new_id, new_blocks[static_cast<size_t>(new_id)])) {
      std::fprintf(stderr, "reorder_index_bfs: pwrite_node %llu 失败\n",
                   static_cast<unsigned long long>(new_id));
      return false;
    }
  }

  // 写外置向量区（偏移与原始相同，内容已重排）
  if (has_ext && !vec_section.empty()) {
    if (!raw_pwrite(dst.fd(), new_hdr.vector_section_offset, vec_section.data(),
                    vec_section.size())) {
      std::fprintf(stderr, "reorder_index_bfs: 写向量区失败\n");
      return false;
    }
  }

  std::fprintf(stderr, "reorder_index_bfs: 完成 nodes=%llu → %s\n",
               static_cast<unsigned long long>(N), dst_path.c_str());
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Gorder-style greedy reordering
// ─────────────────────────────────────────────────────────────────────────────

bool reorder_index_gorder(const std::string &src_path, const std::string &dst_path,
                          uint32_t window_size, std::string *locality_report) {
  // ── 1. 读取源索引 ──
  index::IndexFile src;
  if (!src.open_readonly(src_path)) {
    std::fprintf(stderr, "reorder_index_gorder: 无法打开 %s\n", src_path.c_str());
    return false;
  }
  index::IndexFileHeader hdr{};
  if (!src.read_header(&hdr)) {
    std::fprintf(stderr, "reorder_index_gorder: 读头失败\n");
    return false;
  }
  const uint64_t N = hdr.total_nodes;
  if (N == 0) {
    std::fprintf(stderr, "reorder_index_gorder: 索引为空\n");
    return false;
  }
  if (window_size == 0)
    window_size = 64;

  std::fprintf(stderr, "reorder_index_gorder: nodes=%llu dim=%u window=%u\n",
               static_cast<unsigned long long>(N), hdr.dim, window_size);

  std::vector<index::NodeBlock> blocks(static_cast<size_t>(N));
  for (uint64_t id = 0; id < N; id++) {
    if (!src.pread_node(id, &blocks[static_cast<size_t>(id)])) {
      std::fprintf(stderr, "reorder_index_gorder: pread_node %llu 失败\n",
                   static_cast<unsigned long long>(id));
      return false;
    }
  }

  // ── 2. Gorder 贪心排序 ──
  // score[u]: 在当前滑动窗口内与 u 共享的 layer0 邻居数
  std::vector<int32_t> score(static_cast<size_t>(N), 0);
  std::vector<bool> placed(static_cast<size_t>(N), false);

  // 最大堆 (score, node_id)，惰性删除过时条目
  using ScoreEntry = std::pair<int32_t, uint64_t>;
  std::priority_queue<ScoreEntry> heap;

  // 滑动窗口记录最近 window_size 个已放置节点
  std::deque<uint64_t> window;

  auto layer0_neighbors = [&](uint64_t id) -> std::vector<uint64_t> {
    const auto &b = blocks[static_cast<size_t>(id)];
    const uint8_t cnt = b.neighbor_counts[0];
    const size_t n = cnt > 32 ? 32 : cnt;
    std::vector<uint64_t> nbrs;
    nbrs.reserve(n);
    for (size_t j = 0; j < n; j++) {
      const uint64_t nb = static_cast<uint64_t>(b.neighbors[0][j]);
      if (nb < N)
        nbrs.push_back(nb);
    }
    return nbrs;
  };

  std::vector<uint64_t> order;
  order.reserve(static_cast<size_t>(N));

  const uint64_t ep = hdr.entry_point < N ? hdr.entry_point : 0;
  heap.push({0, ep});

  while (order.size() < N) {
    // 找下一个未放置的最高得分节点
    uint64_t cur = N; // sentinel
    while (!heap.empty()) {
      auto [s, u] = heap.top();
      heap.pop();
      if (placed[static_cast<size_t>(u)])
        continue;
      if (s < score[static_cast<size_t>(u)]) {
        // 过时条目：重新入堆
        heap.push({score[static_cast<size_t>(u)], u});
        continue;
      }
      cur = u;
      break;
    }
    if (cur == N) {
      // 堆空但还有孤立节点：线性扫描找第一个未放置的
      for (uint64_t id = 0; id < N; id++) {
        if (!placed[static_cast<size_t>(id)]) {
          cur = id;
          break;
        }
      }
    }
    if (cur == N)
      break;

    placed[static_cast<size_t>(cur)] = true;
    order.push_back(cur);

    // 窗口头部离开：对其邻居 score--
    if (window.size() >= window_size) {
      const uint64_t leaving = window.front();
      window.pop_front();
      for (const uint64_t nb : layer0_neighbors(leaving)) {
        if (!placed[static_cast<size_t>(nb)]) {
          score[static_cast<size_t>(nb)]--;
        }
      }
    }

    // 放置 cur：对其邻居 score++，并把未放置的邻居加入堆
    window.push_back(cur);
    for (const uint64_t nb : layer0_neighbors(cur)) {
      if (!placed[static_cast<size_t>(nb)]) {
        score[static_cast<size_t>(nb)]++;
        heap.push({score[static_cast<size_t>(nb)], nb});
      }
    }
  }

  // 追加仍未访问的节点（应为0，但以防万一）
  for (uint64_t id = 0; id < N; id++) {
    if (!placed[static_cast<size_t>(id)])
      order.push_back(id);
  }

  // ── 3. 建立 old→new 映射，计算局部性指标 ──
  std::vector<uint64_t> old_to_new(static_cast<size_t>(N));
  for (uint64_t new_id = 0; new_id < N; new_id++)
    old_to_new[static_cast<size_t>(order[static_cast<size_t>(new_id)])] = new_id;

  double locality_before = 0.0, locality_after_val = 0.0;
  if (locality_report) {
    std::vector<uint64_t> identity(static_cast<size_t>(N));
    for (uint64_t i = 0; i < N; i++)
      identity[static_cast<size_t>(i)] = i;
    locality_before = compute_layer0_locality(blocks, identity, N);
  }

  // ── 4. 构建新 NodeBlock 数组 ──
  std::vector<index::NodeBlock> new_blocks(static_cast<size_t>(N));
  for (uint64_t new_id = 0; new_id < N; new_id++) {
    const uint64_t old_id = order[static_cast<size_t>(new_id)];
    index::NodeBlock nb = blocks[static_cast<size_t>(old_id)];
    nb.node_id = new_id;
    for (int lyr = 0; lyr < 8; lyr++) {
      const uint8_t cnt = nb.neighbor_counts[static_cast<size_t>(lyr)];
      const size_t nn = cnt > 32 ? 32 : cnt;
      for (size_t j = 0; j < nn; j++) {
        const uint64_t old_nb =
            static_cast<uint64_t>(nb.neighbors[static_cast<size_t>(lyr)][j]);
        if (old_nb < N) {
          const uint64_t new_nb = old_to_new[static_cast<size_t>(old_nb)];
          nb.neighbors[static_cast<size_t>(lyr)][j] = static_cast<uint32_t>(new_nb);
          nb.neighbor_offsets[static_cast<size_t>(lyr)][j] = index::node_offset(new_nb);
        }
      }
    }
    new_blocks[static_cast<size_t>(new_id)] = nb;
  }

  if (locality_report) {
    locality_after_val = compute_layer0_locality(new_blocks, old_to_new, N);
    std::ostringstream oss;
    oss << "algo=gorder window=" << window_size << "\n";
    oss << "avg_layer0_neighbor_id_dist_before=" << locality_before << "\n";
    oss << "avg_layer0_neighbor_id_dist_after=" << locality_after_val << "\n";
    if (locality_before > 0.0)
      oss << "locality_improvement_ratio="
          << (locality_before - locality_after_val) / locality_before << "\n";
    *locality_report = oss.str();
  }

  // ── 5. 处理 v2 外置向量区 ──
  const bool has_ext = index::header_uses_external_vectors(hdr);
  std::vector<uint8_t> vec_section;
  if (has_ext && hdr.vector_stride_bytes > 0) {
    const uint64_t vec_total = N * static_cast<uint64_t>(hdr.vector_stride_bytes);
    vec_section.resize(static_cast<size_t>(vec_total));
    if (!raw_pread(src.fd(), hdr.vector_section_offset, vec_section.data(),
                   static_cast<size_t>(vec_total))) {
      std::fprintf(stderr, "reorder_index_gorder: 读向量区失败\n");
      return false;
    }
    std::vector<uint8_t> new_vec(static_cast<size_t>(vec_total));
    const size_t stride = static_cast<size_t>(hdr.vector_stride_bytes);
    for (uint64_t new_id = 0; new_id < N; new_id++) {
      const uint64_t old_id = order[static_cast<size_t>(new_id)];
      std::memcpy(new_vec.data() + new_id * stride,
                  vec_section.data() + old_id * stride, stride);
    }
    vec_section = std::move(new_vec);
  }

  // ── 6. 写出目标文件 ──
  index::IndexFile dst;
  if (!dst.open_create_trunc(dst_path)) {
    std::fprintf(stderr, "reorder_index_gorder: 无法创建 %s\n", dst_path.c_str());
    return false;
  }
  index::IndexFileHeader new_hdr = hdr;
  new_hdr.entry_point = old_to_new[static_cast<size_t>(ep)];
  if (!dst.write_header(new_hdr)) {
    std::fprintf(stderr, "reorder_index_gorder: 写头失败\n");
    return false;
  }
  for (uint64_t new_id = 0; new_id < N; new_id++) {
    if (!dst.pwrite_node(new_id, new_blocks[static_cast<size_t>(new_id)])) {
      std::fprintf(stderr, "reorder_index_gorder: pwrite_node %llu 失败\n",
                   static_cast<unsigned long long>(new_id));
      return false;
    }
  }
  if (has_ext && !vec_section.empty()) {
    if (!raw_pwrite(dst.fd(), new_hdr.vector_section_offset, vec_section.data(),
                    vec_section.size())) {
      std::fprintf(stderr, "reorder_index_gorder: 写向量区失败\n");
      return false;
    }
  }
  std::fprintf(stderr, "reorder_index_gorder: 完成 nodes=%llu → %s\n",
               static_cast<unsigned long long>(N), dst_path.c_str());
  return true;
}

} // namespace amio::runtime
