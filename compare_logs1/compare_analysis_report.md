# Linux 经典配置 vs 学习策略（cursor）对比分析报告

数据来源目录：`compare_logs1/`  
对比组：`compare_builtin_summary.log` / `compare_learned_summary.log` 及对应 `*_per_query.csv`  
附注：`eval_quick_*.log` 为另一套更小规模实验，见文末说明。

---

## 1. 实验是否可比

| 项目 | builtin（经典） | learned（学习策略） |
|------|-----------------|----------------------|
| 数据集 | SIFT，`base_size=100000`，`query_size=200` | 相同 |
| 索引 | `data/sift_compare.index`（同一文件，learned 阶段 `--rebuild 0`） | 相同 |
| 检索参数 | `k=10`，`ef_search=128`，`m=16`，`ef_construction=200`，`mode=full` | 相同 |
| Ground truth | `recompute_gt=1` | 相同 |
| io_uring | `io_uring_active=1` | 相同 |

**结论**：两次运行除 **Agent I/O 策略** 外其余条件一致，`recall_at_k` 均为 **0.756**，说明检索质量未因策略切换而下降，延迟与吞吐差异主要来自 **I/O 调度、预取与缓存准入** 行为。

---

## 2. 汇总指标含义说明

| 指标 | 含义 |
|------|------|
| **recall_at_k** | 相对 ground truth 的 recall@k（k=10），衡量检索正确性。 |
| **total_ms** | 200 条 query 串行评测总墙钟时间。 |
| **avg_latency_ms** | 单条 query 平均耗时（总时间 / query 数）。 |
| **p50 / p95 / p99_latency_ms** | 单条 query 延迟的分位数，反映典型与尾部延迟。 |
| **qps** | 吞吐估算：query 数 / (total_ms/1000)。 |
| **sum_disk_sync_block_reads** | 全轮次 **同步** 向磁盘读取的 **4KB 块** 累计次数（缓存未命中后经 `pread`/io_uring 同步读的路径）。 |
| **sum_disk_sync_read_bytes** | 上述同步读的累计字节数（≈ 块数 × 4096）。 |
| **sum_prefetch_blocks_submitted** | 通过拓扑预取提交的异步读 **块数** 累计（每条可能多次 `on_visit_node` 累加）。 |
| **avg_*_per_query** | 上述累加量除以 200 条 query。 |
| **effective_*** | 生效策略参数：预取深度、Layer0 扇出、是否分层/按盘序预取、热点准入「第几次冷未命中再进 hot」等。 |

### 逐查询 CSV 列（`compare_*_per_query.csv`）

| 列名 | 含义 |
|------|------|
| query_idx | 查询序号。 |
| latency_ms | 该条 query 的端到端延迟。 |
| recall_at_k | 该条 query 单独计算的 recall@10。 |
| disk_sync_block_reads | 该 query 内同步读块数。 |
| disk_sync_read_bytes | 该 query 同步读字节。 |
| disk_via_uring / disk_via_pread | 同步读走 io_uring 与 pread 的路径计数。 |
| prefetch_blocks_submitted | 该 query 内提交的预取块数。 |
| cache_hits / cache_misses | 图缓存访问计数（该 query 区间内）。 |

---

## 3. 核心结果对比（100k base × 200 queries）

| 指标 | builtin（经典） | learned（cursor） | 变化 |
|------|----------------|-------------------|------|
| recall_at_k | 0.756 | 0.756 | **持平**（质量无损） |
| total_ms | 3925.11 | 1548.88 | **−60.6%** 总时间 |
| avg_latency_ms | 19.626 | 7.744 | **−60.5%** |
| p50_latency_ms | 11.342 | 7.192 | **−36.6%** |
| p95_latency_ms | 54.295 | 11.859 | **−78.2%** |
| p99_latency_ms | 81.351 | 12.911 | **−84.1%** |
| qps | 50.95 | 129.13 | **+153%**（约为 **2.53×**） |
| sum_prefetch_blocks_submitted | 1 737 712 | 1 327 584 | **−23.6%** |
| sum_disk_sync_block_reads | 87 158 | 159 749 | **+83.3%** |
| sum_disk_sync_read_bytes | ~357 MB | ~654 MB | **+83.3%** |

### 逐查询延迟分布（CSV 统计）

| 项 | builtin | learned |
|----|---------|---------|
| 平均 latency_ms | 19.61 | 7.73 |
| 标准差 | 17.73 | 2.61 |

学习策略下延迟 **均值更低、波动明显更小**（标准差约为 builtin 的 **1/7**），与 p95/p99 的大幅下降一致。

---

## 4. 学习策略相对「经典 Linux 默认」的优势解读

1. **尾延迟显著收缩（p95/p99）**  
   在 recall 不变前提下，p99 从约 **81 ms** 降至约 **13 ms**，对交互式 Agent 检索、批量压测中的 **长尾** 最不友好部分改善最大。

2. **吞吐约 2.5 倍**  
   qps 从约 **51** 到约 **129**，与总时间下降 **~60%** 一致，说明单位时间内完成更多 disk search。

3. **预取提交总量下降**  
   预取块提交减少约 **23.6%**，同时仍保持 recall；学习策略（Layer0 扇出 24、分层感知、按磁盘 offset 排序）有助于 **减少无效预取**，降低 ring/队列压力。

4. **生效策略差异（摘自日志）**  
   - **classic**：默认 `prefetch_depth` 单层、fanout 32、无分层预取与盘序排序、hot 准入为「每次都进」等价（prior_misses=1）。  
   - **learned**：`agent_profile=cursor`，`prefetch_depth_layer0=2` 但 **`max_neighbor_fanout_layer0=24`**，`use_layer_aware_prefetch=1`，`sort_prefetch_by_disk_offset=1`，`hot_insert_min_prior_misses=2`（二次冷未命中再进 hot）。  
   整体上更偏向：**顺序带宽友好 + 控制扇出 + 缓存过滤一次性访问**，与更低的尾延迟和更高 qps 一致。

### 为何「同步读块数」反而更高却仍更快？

汇总里 learned 的 **sum_disk_sync_block_reads** 与 **读字节数** 高于 builtin，但 **墙钟时间与分位延迟更低**。这在工程上常见，可能包括但不限于：

- 预取按 **offset** 排序后，底层顺序读、预读与 SSD 并行度更有效，单次同步读 **摊销延迟**更低；  
- 分层预取使 **关键路径上的阻塞** 与 **无效异步工作量** 减少，尽管计数器仍统计了更多落到同步路径上的块；  
- 缓存准入策略改变 **命中/未命中时序**，与预取叠加后整体 **并行度与调度** 更优。

因此对比结论应 **以 recall + 延迟分位 + qps 为主**；同步读块数需结合预取量与实现细节解读，不宜单独当作「越慢」。

---

## 5. 提升幅度小结（相对 builtin）

| 维度 | 提升幅度（约） |
|------|----------------|
| 平均延迟 | **降低 ~60%** |
| 总评测时间 | **降低 ~61%** |
| 吞吐 qps | **提升至 ~2.5×** |
| p99 延迟 | **降低 ~84%** |
| 预取提交块数 | **减少 ~24%** |
| recall@10 | **无变化（0.756）** |

---

## 6. 附：`eval_quick_summary.log` 说明

`compare_logs1/eval_quick_summary.log` 对应 **另一组** 实验参数：

- `base_size=5000`，`query_size=20`，`ef_search=64`，索引为 `data/sift_base.eval.index`；  
- 与上表 **100k×200、ef_search=128** 的 compare **不可直接横向比数值**，仅可作「小规模 learned 单次跑通」参考（该次 recall_at_k≈0.955）。

若要统一口径，请在相同 `base_limit`/`query_limit`/`ef_search`/`index_path` 下重新跑两遍并归档日志。

---

## 7. 复现实验命令（参考）

```bash
./scripts/linux_run.sh compare
# 或手动两次 eval_disk，与 linux_run.sh compare 等价逻辑
```

将生成的 `logs/compare_*` 复制到 `compare_logs1/` 即可与本报告结构对齐。

---

*报告由 `compare_logs1` 内现有汇总与 CSV 自动生成分析，不涉及修改原始日志。*
