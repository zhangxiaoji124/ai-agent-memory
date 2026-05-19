# Agent 长时记忆存储引擎 — 设计 vs 实现对照

> 对应设计原文：`Agent 长时记忆存储引擎设计(1).md`（赛题目标架构论述）  
> 本文档说明**哪些内容已在 `agent-memory-io-cpp` 仓库落地、哪些仍为设计愿景**，并给出后续完善优先级。  
> 关联文档：`docs/赛题剩余工作与交付计划.md`、`papers-algorithm-summary.md`、`README.md`

---

## 1. 实现状态图例

| 状态 | 含义 |
|------|------|
| **已实现** | 有可运行代码，可通过 build / eval_disk / 单测验证 |
| **部分实现** | 骨架或简化版，与论文/设计文档差距明显 |
| **未实现** | 设计有述，仓库无对应模块 |
| **离线/代理** | Python 合成轨迹或 learn 集游走近似特征，非真实 Agent 遥测 |

---

## 2. 顶层三子系统对照

| 设计子系统 | 设计目标（摘要） | 实现状态 | 仓库落点 |
|------------|------------------|----------|----------|
| 用户态智能读缓存 | GoVector 动静混合、Group Cache、PAIC | **部分** | `cache/graph_aware_cache.*` |
| 拓扑感知异步预取 | 图重排、Episode、io_uring、ML 门控 | **部分** | `prefetch/topology_prefetcher.*`、`prefetch/io_uring_backend.*` |
| 写优化 LSM | MemTable、NVTable、RemapCom、NobLSM | **部分（写骨架）** | `write/memtable.*`、`wal.*`、`compaction.*` |
| Agent ML 闭环 | 学习访问模式 → 调缓存/预取 | **离线/代理** | `learning/train_agent_policy.py`、`policy/agent_io_policy.*` |

```text
┌──────────────────────────────────────────────────────────────┐
│ 设计：读缓存 │ 实现：GraphAwareCache（pinned + hot LRU）       │
│ 设计：预取   │ 实现：TopologyPrefetcher + io_uring（Linux）    │
│ 设计：写路径 │ 实现：MemTable + WAL + CompactionWorker 骨架    │
└──────────────────────────────────────────────────────────────┘
              ▲ 离线 JSON 合并 ── AgentIoPolicy ◄── train_agent_policy.py
```

---

## 3. 读优化（一）：用户态智能混合缓存

### 3.1 设计文档描述 vs 现状

| 设计要点 | 状态 | 说明 |
|----------|------|------|
| 内存 10% MemTable / 20% Static / 70% Dynamic 三区 | **未** | 读侧无按比例硬切；写侧仅有 MemTable 概念 |
| GoVector 静态缓存（入口+高层邻居） | **部分** | `GraphAwareCache::pin()` 可锚定节点；无自动提取高层子图 |
| GoVector 动态缓存（路径+相似邻居页） | **未** | 无环形区域观测、无相似页批量加载 |
| 阶段切换参数 θ（收敛期→探索期） | **未** | 检索统一 `get_or_load`，无在线 θ |
| Group Cache（KV Group + 范围不对称性） | **未** | 非 LSM 多层 KV；缓存单元为整页 `NodeBlock`（4KB） |
| LRU-S 大小感知淘汰 | **未** | hot 区为 tick + 全表扫描淘汰（O(n) 骨架） |
| PAIC（ISVM 双预测器、Demand-MIN、PCHR） | **未** | 无 demand/prefetch 分流、无 ISVM |
| 一次性访问排除 | **部分** | `hot_insert_min_prior_misses`：第 N 次冷未命中才入 hot |

### 3.2 已实现代码路径

- `include/cache/graph_aware_cache.h`、`src/cache/graph_aware_cache.cpp`
- 策略指针：`const policy::AgentIoPolicy *policy_`
- 指标：`hits_total()` / `misses_total()`，与 `eval_disk` CSV 列 `cache_hits` / `cache_misses` 联动

---

## 4. 读优化（二）：拓扑预取与图布局

| 设计要点 | 状态 | 说明 |
|----------|------|------|
| Gorder / Corder / Porder 图重排 | **未** | `NodeBlock` 按 `node_id` 线性偏移落盘（`storage_layout.h`） |
| Episode + Baleen ML 预取/准入 | **未** | 无 Episode 统计与模型 |
| io_uring 环形队列批量预取 | **已实现**（Linux） | `AMIO_ENABLE_URING=ON`；非 Linux 为 stub |
| ML-Range / ML-When + 负载 tanh 门控 | **未（在线）** | 由离线 JSON 静态参数近似 |
| 按磁盘 offset 顺序预取（SARC 类） | **部分** | `sort_prefetch_by_disk_offset=true` 时排序 `neighbor_offsets` |
| 分层预取深度 | **部分** | `prefetch_depth_upper` / `prefetch_depth_layer0` |
| 当前层邻居预取（减无效读） | **部分** | `use_layer_aware_prefetch` |
| layer0 扇出上限 | **部分** | `max_neighbor_fanout_layer0` |

### 4.1 已实现代码路径

- `include/prefetch/topology_prefetcher.h`、`src/prefetch/topology_prefetcher.cpp`
- `include/prefetch/io_uring_backend.h`、`src/prefetch/io_uring_backend.cpp`
- 集成：`src/vector_store.cpp` → `search_disk`

---

## 5. 写优化：LSM 与高级合并

| 设计要点 | 状态 | 说明 |
|----------|------|------|
| MemTable 吸收写入 | **部分** | `write/memtable.*` |
| WAL | **部分** | `write/wal.*`（**仍保留 WAL**，非 NVTable 无 WAL） |
| 多层 SST + 后台 Compaction | **部分** | `CompactionWorker` 队列 + 回调，无完整 LSM 层级 |
| NVTable / List Compaction | **未** | 无 NVM 链表、无指针合并 |
| RemapCom（UDB + SSD remap） | **未** | 无 UDB 状态机、无 remap ioctl |
| NobLSM（check_commit / is_committed） | **未** | 无免 fsync 内核协同 |

> **评测说明**：当前赛题主评测路径为磁盘**读**（`tools/eval_disk`）；写路径对检索尾延迟的影响尚未在统一脚本中量化。

---

## 6. Agent 机器学习与策略（赛题评审重点）

### 6.1 流水线（当前）

```text
数据来源 ──► 轨迹/游走 ──► TraceStats(4维) ──► 规则选簇 ──► JSON ──► C++ merge_json_file
   │              │              │                │
   ├ synthetic    ├ 200步/轨迹   ├ mean_fanout    └ cluster 0/1 → 两套预取/缓存参数
   └ learn-fvecs  └ 随机池+L2    └ seq_locality, high_layer_share
```

### 6.2 四维特征（`learning/train_agent_policy.py`）

| 特征 | 含义 |
|------|------|
| `mean_fanout` | 每步扩展邻居数均值 |
| `std_fanout` | 扇出波动 |
| `seq_locality` | 邻居 id 是否成簇（近似顺序读） |
| `high_layer_share` | 「高层导航」步占比（learn 模式用距离比启发式） |

### 6.3 能力对照

| 能力 | 状态 |
|------|------|
| Cursor / Qwen / Qwen3 合成轨迹 | **已实现**（`--dump-features`） |
| SIFT learn 集随机游走 | **已实现**（`--data-source learn-fvecs`；`linux_run.sh policy` 自动选用） |
| sklearn 决策树 | **探针 only**（打印 `feature_importances`，不导出到 C++） |
| 策略输出 | **已实现**（`_policy_for_cluster` 两套 JSON 参数） |
| 运行时 learned 模式 | **已实现**（`eval_disk --policy-mode learned`） |
| 真实 Agent 生产 trace | **未实现** |
| 在线 Bandit / ISVM | **未实现** |

### 6.4 learn 模式已知局限

- 候选池为**随机 id**，`seq_locality` / `high_layer_share` 常接近 0，易落入 **cluster 1**（宽扇出、关闭按层/按 offset 优化）。
- `agent_profile` 字段仅为命名；**簇由聚合特征规则决定**，与 profile 字符串无强制绑定。

### 6.5 两套运行时策略（cluster）

| 字段 | cluster 0（偏窄扇出、顺序化） | cluster 1 |
|------|-------------------------------|-----------|
| `prefetch_depth_upper` | 1 | 2 |
| `prefetch_depth_layer0` | 2 | 2 |
| `max_neighbor_fanout_layer0` | 24 | 32 |
| `use_layer_aware_prefetch` | true | false |
| `sort_prefetch_by_disk_offset` | true | false |
| `hot_insert_min_prior_misses` | 2 | 1 |

---

## 7. 透明集成与性能评估

| 设计要点 | 状态 |
|----------|------|
| LangChain / LlamaIndex VectorStore | **未** |
| Session ID / 时间戳边过滤 | **未** |
| 统一 QPS / P99 / Recall@10 脚本 | **部分**（`eval_disk`、`linux_run.sh compare`） |
| Recall@10 ≥ 85%（10–20% 内存） | **待 Linux 全量固化** |
| ZNS / Open-Channel 深度适配 | **未** |

---

## 8. bvec 与 500GB / 100GB 内存

| 项 | 状态 | 说明 |
|----|------|------|
| `.bvecs` mmap + 子集加载 | **已实现** | `dataset/vector_dataset.*` |
| 流式建索引 | **已实现** | `dataset/index_builder.*`、`build_index --ram-budget-gb` |
| 高维向量 dim>128（如 960） | **已实现** | 索引 v2：图块 + 尾部外置向量区 `external_vectors.*` |
| uint8 外置存储 | **已实现** | bvec 可 `VectorEncoding::UInt8` 写入向量区 |
| 500GB 全量构图 | **部分** | 需 Linux 大磁盘；逻辑为分批 mmap |
| 三区预算 10/20/70 | **已实现** | `runtime/memory_partition.*` → `Config` |

---

## 9. 设计文档新增章节索引（主文档）

主文档 `Agent 长时记忆存储引擎设计(1).md` 除实现对照外，尚包含下列**可实施规范**（代码未全落地）：

| 章节 | 内容 |
|------|------|
| 待实现功能清单 R1–R12 | 验收标准与模块边界 |
| 大规模 bvec / 500G·100G | 容量粗算、三区预算、读路径 |
| 内存划分区分器 M0–M4 | 决策伪代码、JSON 示例 |
| 各模式运行时参数缺省 | Config 联动表 |
| VectorDataset / NodeBlock v1–v3 | 统一 I/O 与布局演进 |
| 端到端数据流 | build → open → search_disk |
| 三区与模块映射、Static 提取、θ | R4/R5 算法规范 |
| bvec 距离、分片、磁盘规划 | 工程部署 |
| Agent JSON 扩展、CLI 契约 | R11/R12 |
| Phase 1–5 排期、异常降级、可观测性 | 项目管理 |

---

## 10. 内存划分模式区分器（MemoryPartitionDiscriminator）

| 模式 | 状态 | 落点 |
|------|------|------|
| M0–M4 自动/强制选择 | **已实现** | `src/runtime/memory_partition.cpp` |
| Static 子图 pin | **已实现** | `src/runtime/static_subgraph.cpp`，`VectorStore::open` |
| Dynamic / Static 分池缓存 | **已实现** | `GraphAwareCache` 双容量 |
| CLI | **已实现** | `eval_disk --ram-budget-gb`、`--memory-profile`；`AMIO_RAM_BUDGET_GB` |

---

## 11. 未完成项汇总（按设计章节）

| 设计章节 | 未完成核心内容 |
|----------|----------------|
| § 读优化（一） | θ 阶段切换、Group Cache、LRU-S、PAIC/ISVM、70% 动态相似缓存 |
| § 读优化（二） | 图重排、Episode/Baleen 在线模型、ML-When/Range 在线推理 |
| § 写优化 | NVTable、RemapCom、NobLSM、完整 LSM 多层与写放大评测 |
| § 透明集成 | 标准 Agent SDK、标量过滤 |
| § 性能评估 | 受限内存下 recall 基线证明、混合读写压测报告 |
| § bvec / 500G·100G | 流式 bvec、mmap、分批建索引 |
| § 区分器 | 文件类型+大小→M0–M4 与三区预算 |

---

## 12. 后续完善优先级

| 优先级 | 工作项 | 对齐设计 |
|--------|--------|----------|
| **P0** | **bvec** I/O + **MemoryPartitionDiscriminator** + `--ram-budget-gb` | 500GB/100GB 赛题场景 |
| **P0** | Linux + io_uring 对比数据；固化评测口径 | 性能评估 |
| **P0** | 真实 search trace 或图邻接游走替代随机 id 池 | Agent ML、seq_locality |
| **P1** | 阶段感知缓存（θ 或按 layer 切换策略） | GoVector |
| **P1** | prefetch useful/wasted 计数；预取限流 | PAIC、ML-When |
| **P1** | 建索引后图重排（BFS/RCM 简化版即可） | Graph Reordering |
| **P2** | RemapCom / NobLSM / NVTable | 写优化全文 |
| **P2** | LangChain 薄封装 | 透明集成 |

---

## 13. 复现命令

```bash
# 策略（有 sift_learn.fvecs 则 learn-fvecs）
python learning/train_agent_policy.py --data-source auto \
  --learn-fvecs data/sift/sift_learn.fvecs --out data/agent_io_policy.json

# 合成 Agent 特征分布（不写 JSON）
python learning/train_agent_policy.py --dump-features --dump-profiles cursor,qwen3

# Linux：builtin vs learned
./scripts/linux_run.sh compare
```

---

## 14. 建议合并回主设计文档

若 `Agent 长时记忆存储引擎设计(1).md` 可写，建议在 **「全域性能评估」与「结语」之间** 插入一节「工程实现状态与文档对照」，并链接本文档，避免答辩时将**设计目标**误述为**已全部实现**。
