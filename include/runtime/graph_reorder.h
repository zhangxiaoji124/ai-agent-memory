#pragma once

#include <string>

namespace amio::runtime {

/// BFS 图重排：从入口节点出发广度优先遍历，将拓扑邻近节点赋予相近的物理 id，
/// 使 layer0 邻居块在磁盘上尽量连续排列，减少检索时的随机 I/O 跨度。
///
/// 算法：
///  1. 从 IndexFileHeader.entry_point 开始 BFS，优先展开 layer0 邻居（再扩展上层）。
///  2. 按 BFS 访问顺序分配新 id：entry_point → new_id=0，其 layer0 邻居→ 1,2,3…
///  3. 无法通过 BFS 到达的孤立节点按原 id 顺序追加在尾部。
///  4. 用新 id 重写所有 NodeBlock（node_id、neighbors、neighbor_offsets）。
///  5. v2（外置向量区）也同步重排，保证 vec[new_id] = 原 vec[old_id]。
///
/// @param src_path  原始索引文件路径（只读）
/// @param dst_path  重排后输出路径（会被截断覆盖）
/// @param locality_report  若非 nullptr，写入局部性指标字符串（可选）
/// @return true 表示成功
bool reorder_index_bfs(const std::string &src_path, const std::string &dst_path,
                       std::string *locality_report = nullptr);

/// Gorder 风格图重排：贪心选择与最近 window_size 个已放置节点共享最多邻居的下一节点，
/// 比 BFS 产生更紧密的 layer0 局部性（但耗时约为 BFS 的 3-5 倍）。
///
/// 算法（参考 Wei et al. VLDB 2016 简化版）：
///  1. 维护 score[u]：u 有多少邻居已被放入最近窗口。
///  2. 每放置节点 v 后，v 所有 layer0 邻居 score++ ；
///     当窗口头部节点 w 离开窗口时，w 所有 layer0 邻居 score--。
///  3. 用最大堆（懒删除）每步取得分最高的未访问节点。
///
/// @param window_size  滑动窗口大小（推荐 32–128，默认 64）
bool reorder_index_gorder(const std::string &src_path, const std::string &dst_path,
                          uint32_t window_size = 64,
                          std::string *locality_report = nullptr);

} // namespace amio::runtime
