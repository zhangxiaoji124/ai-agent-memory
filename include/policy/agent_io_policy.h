#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace amio {
struct Config;

namespace policy {

/// 由离线 Python（sklearn 等）根据 Agent 访问轨迹学习后导出的 I/O 策略，
/// 运行时被 C++ 加载，用于调节拓扑预取与热点缓存准入（对齐 GoVector / PAIC /
/// Baleen / 一次性访问排除等思路的工程化子集）。
struct AgentIoPolicy {
  std::string agent_profile = "default";

  /// 高层贪心（layer>0）预取重复轮数。
  size_t prefetch_depth_upper = 1;
  /// Layer0 best-first 阶段预取重复轮数。
  size_t prefetch_depth_layer0 = 1;

  /// Layer0 单次预取的最大邻居扇出（上限 32）。
  size_t max_neighbor_fanout_layer0 = 32;

  /// 为 true 时仅预取当前搜索层的邻居，减少无效异步读。
  bool use_layer_aware_prefetch = false;

  /// 为 true 时按磁盘 offset 对预取 id 排序，偏向顺序读（SARC / 拓扑顺序）。
  bool sort_prefetch_by_disk_offset = false;

  /// 第几次「冷未命中加载」后才进入 hot 区；1 等价于传统「总是插入」，
  /// 2 近似「二次访问再缓存」，对应一次性访问排除的简化版。
  uint32_t hot_insert_min_prior_misses = 1;

  /// GoVector θ 阶段切换比例（0 表示关闭）。
  /// 当 layer0 搜索中 visited_count >= θ * ef_search 时进入探索期（phase2）。
  /// M3/M4 默认 0.35，M0 关闭（0.0）。
  float partition_theta = 0.0f;

  /// 与 `Config::prefetch_depth` 对齐的默认策略（不改变既有行为）。
  static AgentIoPolicy from_config(const Config &cfg);

  /// 自 `path` 读取 JSON，覆盖 `*out` 中在文件里出现的字段；失败返回 false。
  static bool merge_json_file(const std::string &path, AgentIoPolicy *out,
                             std::string *error_message);
};

} // namespace policy
} // namespace amio
