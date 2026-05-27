# ai-agent-memory（agent-memory-io-cpp）

面向 Agent 记忆的向量检索系统 I/O 优化（C++）。支持磁盘 HNSW 检索、io_uring 预取、图感知缓存（Static/Dynamic 分池）、**内存划分模式自动区分**（fvecs/bvecs、体量/维度）、**高维向量**（dim>128 如 GIST 960），以及 **builtin / learned** 策略对照评测。

> 仓库地址：[zhangxiaoji124/ai-agent-memory](https://github.com/zhangxiaoji124/ai-agent-memory)

## 实现进度摘要（2026-05）

### 已完成功能

| 模块 | 对应设计 | 要点 |
|------|----------|------|
| 磁盘 HNSW 检索（`search_disk`） | GoVector 核心路径 | `disk_search_layer` 多层贪心导航 + layer0 beam search |
| **GoVector θ 阶段切换** | R5 / GoVector 动静混合 | `visited_count ≥ θ·ef_search` 触发 phase2，提高预取扇出；`theta_phase_switches` 可观测 |
| GraphAwareCache 双池（Static + Dynamic LRU） | R4 | pinned 静态导航子图 + hot LRU 动态缓存 |
| 静态子图 pin（启动时 BFS 导航层） | R4 | `pin_static_navigation_subgraph`，M4 下最激进 |
| io_uring 异步预取（Linux） | 读优化（二） | `IoUringBackend` + `TopologyPrefetcher`；非 Linux 自动 noop |
| **io_uring slot 循环复用修复** | — | 修复 user_data 编码错误；`drain_completions_nonblocking` 正确释放 slot |
| 内存划分区分器 M0–M4 | R2/R3 | 根据文件类型 + `data_bytes/ram_budget` 自动选择；三区切分 MemTable 10% / Static 20% / Dynamic 70% |
| bvec mmap + `VectorDataset` 统一抽象 | R1 | 支持 fvecs/bvecs 任意 dim；无需整文件载入 RAM |
| 高维向量 v2 外置区（dim > 128） | R6 | 960 维等高维向量存放索引尾部，NodeBlock 仅存图结构 |
| `partition_theta` 策略字段 | R5/R11 | JSON 解析 + 各 profile 默认值（M3/M4 = 0.35，M0 = 0.0） |
| 离线策略学习（`train_agent_policy.py`） | R11 | sklearn 聚类 → `agent_io_policy.json`；含 memory_profile、partition_theta |
| `eval_disk` 评测工具 | R12 | QPS / P50/P95/P99 / Recall@k；逐查询 CSV；`theta_phase_switches` 列 |
| DiskANN RobustPrune HNSW 构图 | — | `build_index` 流式分批构图，支持 bvec + dim > 128；遮挡不等式已修正（见下方本轮） |
| **SIMD 距离（AVX2/FMA 运行时派发）** | — | `util/simd_distance.h` 统一 L2 平方距离；GIST 960 维端到端 **2.31×**，召回不变 |
| WAL + MemTable + CompactionWorker | 写优化 | NVTable 链表缓冲、NobLSM 异步提交、RemapCom UDB、**增量写回 `.index`** |
| **KV Group + ISVM 缓存驱逐** | R8 简化 | layer0 邻居组字节估算 + 整数线性 ISVM 驱逐分 |
| **PAIC 预取有效性追踪** | R8 简化 | `useful_prefetch_demand_hits` / `wasted_prefetch_evictions`；`prefetch_utilization_rate` 写入汇总 |
| **BFS 图重排** | R9 简化 | `reorder_index` BFS 重排节点顺序；`compare-reorder` 一键量化 I/O 局部性改善 |
| Linux 一键测试脚本 | R12 | `scripts/linux_run.sh all/compare/eval-quick/reorder/compare-reorder` |

### 最近完善（本轮）

**GoVector θ 阶段切换（上轮）：**
- 在 `disk_search_layer` 实现 `visited_count ≥ θ·ef` 触发 `phase2=true`，layer0 搜索中自动切换预取扇出；上层贪心导航不启用 θ。
- io_uring slot 泄漏修复：`user_data` 改为直接存 `slot_idx`；`drain_completions_nonblocking` 每次 `on_visit_node` 前执行。
- `AgentIoPolicy::partition_theta`：JSON 解析 + 各内存 profile 默认值（M3/M4 = 0.35）。

**PAIC 预取有效性追踪（本轮）：**
- `GraphAwareCache` 新增 `record_prefetch_submitted(ids)` + `HotEntry.is_prefetch_loaded` 标志。
- `TopologyPrefetcher` 在每次 `submit_prefetch` 后通知缓存（`set_cache` 接口）。
- 命中已标记条目 → `io_metrics_.useful_prefetch_demand_hits++`（有效预取）。
- 淘汰仍标记条目 → `io_metrics_.wasted_prefetch_evictions++`（无效预取）。
- `eval_disk` CSV 新增 `useful_prefetch_demand_hits` / `wasted_prefetch_evictions`；汇总日志新增 `prefetch_utilization_rate`。

**BFS 图重排（本轮）：**
- `include/runtime/graph_reorder.h` + `src/runtime/graph_reorder.cpp`：从入口节点 BFS，将拓扑邻近节点赋予相近物理 id，layer0 邻居在磁盘上趋向连续。
- 支持 v1（内联向量）与 v2（外置向量区）；输出同格式索引可直接替换使用。
- `tools/reorder_index.cpp`：独立 CLI，打印重排前后 `avg_layer0_neighbor_id_dist` 局部性指标。
- `scripts/linux_run.sh reorder` + `compare-reorder`：一键重排 + 对比评测。

**2026-05-27 实测修复与优化（关键）：**
- **修复召回崩塌 bug**：`HnswIndex::robust_prune` 的 DiskANN 遮挡不等式写反（alpha 放错边），导致每节点被过度剪枝到约 1 个邻居。修正后召回大幅恢复（Linux 实机 `--recompute-gt 1` 口径）：
  - SIFT 100K @ ef=192：recall@10 **0.12 → 1.000**
  - GIST 10K（960 维）@ ef=192：recall@10 **0.007 → 0.999**
- **SIMD 距离（AVX2 + FMA，运行时派发）**：新增 `include/util/simd_distance.h` 统一 L2 平方距离，`HnswIndex` / `ExternalVectorStore` / `VectorStore` 三处接入；无 AVX2 时标量自动兜底，`AMIO_FORCE_SCALAR=1` 可强制标量做 A/B。端到端（建图+GT+检索）实测：
  - GIST 10K（960 维）：**2.31×**（288.6s → 124.8s）
  - SIFT 100K（128 维）：**1.43×**（855.5s → 597.7s）
- **建图加速**：`add_edge` 邻居表未满时直接追加，仅在超过容量上限时才重跑 RobustPrune（hnswlib/DiskANN 同款），避免每条边 O(cap²) 重剪枝。
- **`eval_disk` 受限内存评测**：新增 `--cache-size-mb` / `--static-cache-mb`，可将动态缓存压到远小于索引大小，制造真实缓存未命中以评测预取/缓存（对齐 10–20% 内存场景）。
- **修复 `eval_disk --recompute-gt 1`**：此前 `nq` 取 `gt_all.size()`（重算 GT 时为 0）导致评测 0 条 query。
- 详见 `docs/提升方案-基于实测结果.md` 与 `docs/Agent长时记忆-实现状态对照.md`（已同步修订）。

### 待后续实现（设计文档对应优先级）

| 优先级 | 内容 | 对应设计节 |
|--------|------|-----------|
| ~~P1~~ 已完成 | ~~PAIC~~ demand/prefetch 计数（useful_hits / wasted_evictions）已落地；KV Group 粒度缓存与 ISVM 替换策略仍为研究级 | § 读优化（一）R8 |
| ~~P1~~ 已完成 | ~~图重排（BFS 简化版）~~已落地（`reorder_index`）；Gorder/Porder 权重重排为研究级 | § 读优化（二）R9 |
| P2 | LangChain / LlamaIndex Python VectorStore 封装 | § 透明集成（`python/amio` 已提供 ctypes 基础） |
| ~~P2 部分~~ | ~~RemapCom / NobLSM / NVTable~~ | 简化版已落地；完整内核协同仍为后续 |
| — | 全量 500 GB bvec 端到端评测 + `--ram-budget-gb 100` 口径文档 | R7/R12 |

---

## 目录结构

```
├── CMakeLists.txt          # 构建配置
├── Dockerfile              # Ubuntu 容器内一键构建示例
├── scripts/linux_run.sh    # Linux 实机测试脚本（推荐入口）
├── include/ src/           # 核心库 amio
├── tools/eval_disk.cpp     # 磁盘检索评测 + 逐查询指标 CSV
├── learning/               # 离线策略学习（Python）
├── tests/ benches/         # 单测与微基准
└── docs/                   # 方案与交付说明
```

**不会提交到 Git 的本地大文件**（见 `.gitignore`）：`data/sift/`、`data/*.index`、`data/sift.tar/`、`build/`、`logs/` 等。索引与 SIFT 需在 Linux 上自行下载/生成。

---

## Linux 实机操作（详细）

以下在 **Ubuntu 22.04 / Debian / Fedora** 等常见发行版上验证流程；其他发行版可先手动安装依赖再执行 `build`。

### 0. 克隆与准备

```bash
git clone https://github.com/zhangxiaoji124/ai-agent-memory.git
cd ai-agent-memory
chmod +x scripts/linux_run.sh
```

### 1. 一键全流程（推荐首次使用）

自动：安装依赖 → 下载 SIFT → 生成学习策略 JSON → 编译 → 单测 → 基准 → 小规模磁盘评测。

```bash
./scripts/linux_run.sh all
```

### 2. 分步执行

| 步骤 | 命令 | 说明 |
|------|------|------|
| 安装依赖 | `./scripts/linux_run.sh deps` | 按发行版安装 cmake、g++、liburing-dev、wget、python3（需 sudo） |
| 下载数据 | `./scripts/linux_run.sh fetch-sift` | SIFT 解压到 `data/sift/` |
| 探测数据集 | `./scripts/linux_run.sh probe-dataset data/sift/sift_base.fvecs` | 类型/维度/体量 + 内存划分模式 M0–M4 |
| 学习策略 | `./scripts/linux_run.sh policy` | 生成 `data/agent_io_policy.json`（fvecs/bvecs learn） |
| 编译 | `./scripts/linux_run.sh build` | 产物在 `build/` |
| 建索引 | `./scripts/linux_run.sh build-index` | 默认 SIFT fvecs；支持 bvec、dim>128 流式 |
| 图重排 | `./scripts/linux_run.sh reorder` | BFS 重排索引节点顺序，改善 layer0 空间局部性 |
| 单测 | `./scripts/linux_run.sh test` | 含 memory_partition、960 维索引测试 |
| 基准 | `./scripts/linux_run.sh bench` | `bench_search` + `bench_write` |

### 3. 两种 I/O 策略对比（builtin vs learned）

**builtin**：仅使用 C++ 内置默认策略（`AgentIoPolicy::from_config`），不加载 JSON。  
**learned**：在默认策略上合并 `data/agent_io_policy.json`（由 `learning/train_agent_policy.py` 导出）。

一键对比（同一索引，先后跑两种模式，日志在 `logs/compare_*`）：

```bash
./scripts/linux_run.sh compare
```

也可手动指定：

```bash
# 内置策略
./build/eval_disk --policy-mode builtin \
  --log logs/builtin_summary.log \
  --metrics-log logs/builtin_per_query.csv

# 学习策略
./build/eval_disk --policy-mode learned \
  --agent-policy data/agent_io_policy.json \
  --log logs/learned_summary.log \
  --metrics-log logs/learned_per_query.csv
```

### 4. 评测日志与指标

`eval_disk` 会写入两类日志：

1. **汇总**（`--log`，默认 `logs/eval_disk.log`）：recall@k、QPS、P50/P95/P99 延迟、io_uring 是否启用、磁盘读块总数、生效策略参数等。  
2. **逐查询 CSV**（`--metrics-log`，默认 `logs/eval_disk_per_query.csv`）：

| 列名 | 含义 |
|------|------|
| `latency_ms` | 单次查询耗时 |
| `recall_at_k` | 该查询 recall@k |
| `disk_sync_block_reads` | 同步磁盘读块数（4KB/块） |
| `disk_sync_read_bytes` | 同步读字节数 |
| `disk_via_uring` / `disk_via_pread` | 读路径分布 |
| `prefetch_blocks_submitted` | 预取提交块数 |
| `cache_hits` / `cache_misses` | 图缓存命中/未命中 |
| `theta_phase_switches` | 该查询中 θ 阶段切换次数（0 = 未触发，1 = 已切换至探索期） |
| `useful_prefetch_demand_hits` | 预取协同加载的缓存条目被后续需求命中次数（有效预取） |
| `wasted_prefetch_evictions` | 预取加载的缓存条目在需求到来前被驱逐次数（无效预取） |

关闭逐查询 CSV：`--metrics-log none`

### 5. 向量格式、维度与内存划分

| 格式 | 说明 |
|------|------|
| `.fvecs` | float32 向量（SIFT 子集、GIST 960 等） |
| `.bvecs` | uint8 向量（SIFT1B 等大规模库） |
| `.ivecs` | groundtruth，仅评测用 |

- **dim ≤ 128**：索引 **v1**，向量内联在 4KB `NodeBlock` 中。  
- **dim > 128**（如 **960 维**）：索引 **v2**，图结构仍在 4KB 块，向量在索引文件**尾部外置区**（mmap 读取）。  
- **内存划分区分器** 根据文件类型与 `data_bytes / ram_budget` 选择模式（`HOST_RESIDENT` … `BVEC_ULTRA_500G_100G`），并切分 MemTable 10% / Static 20% / Dynamic 70%。

```bash
# 100GB 内存预算下评测（大 bvec 场景）
export AMIO_RAM_BUDGET_GB=100
export AMIO_MEMORY_PROFILE=BVEC_ULTRA_500G_100G   # 可选，强制模式
./scripts/linux_run.sh eval-quick

./build/build_index --input /path/base.bvecs --output data/big.index \
  --ram-budget-gb 100 --max-vectors 10000000
```

### 6. 常用环境变量

```bash
export CC=gcc CXX=g++              # 或 clang/clang++
export BUILD_DIR=build
export JOBS=$(nproc)
export AMIO_ENABLE_URING=ON        # 无 liburing 时设为 OFF
export CMP_BASE_LIMIT=100000       # compare 时 base 向量数上限
export CMP_QUERY_LIMIT=200         # compare 时 query 数上限
export AMIO_RAM_BUDGET_GB=100      # 内存划分预算（与 --ram-budget-gb 等价）
export AMIO_MEMORY_PROFILE=DISK_FIRST
```

### 7. 不用脚本时的等价命令

```bash
# Debian/Ubuntu 依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake wget liburing-dev python3

cmake -B build -DAMIO_ENABLE_URING=ON
cmake --build build -j$(nproc)
./build/run_tests

# 生成索引（fvecs；dim>128 或 bvec 自动流式 + v2 外置向量）
./build/build_index --input data/sift/sift_base.fvecs --output data/sift_base.index \
  --ram-budget-gb 100

# 磁盘评测（自动内存划分 + 静态子图 pin）
./build/eval_disk --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --index data/sift_base.index \
  --ram-budget-gb 100 \
  --policy-mode learned --agent-policy data/agent_io_policy.json
```

### 8. Docker（可选）

```bash
docker build -t ai-agent-memory .
docker run --rm ai-agent-memory
```

容器内会下载 SIFT、编译并运行 `run_tests`。

### 9. 离线学习策略（Python）

```bash
python3 learning/train_agent_policy.py --data-source auto --out data/agent_io_policy.json
python3 learning/train_agent_policy.py --data-source learn-bvecs \
  --learn-fvecs data/sift/sift_learn.bvecs --out data/agent_io_policy.json
python3 learning/train_agent_policy.py --dump-features
```

示例 JSON 见 `learning/policies/*.example.json`。设计说明见 `Agent 长时记忆存储引擎设计(1).md`、`docs/Agent长时记忆-实现状态对照.md`。

---

## 构建（通用）

```bash
cmake -B build
cmake --build build -j
```

## 说明

- 非 Linux 或未找到 `liburing` 时使用 **noop 预取后端**，保证可编译可跑。
- `io_uring` 完整异步流水线（O_DIRECT + 对齐缓冲池 + CQ 收割）为后续迭代点。
- 更多子命令：`./scripts/linux_run.sh help`
