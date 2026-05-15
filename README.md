# ai-agent-memory（agent-memory-io-cpp）

面向 Agent 记忆的向量检索系统 I/O 优化（C++）。支持磁盘 HNSW 检索、io_uring 预取、图感知缓存，以及 **内置策略 / 离线学习策略** 对照评测。

> 仓库地址：[zhangxiaoji124/ai-agent-memory](https://github.com/zhangxiaoji124/ai-agent-memory)

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
| 安装依赖 | `./scripts/linux_run.sh deps` | 按发行版安装 cmake、g++、liburing-dev、wget（需 sudo） |
| 下载数据 | `./scripts/linux_run.sh fetch-sift` | SIFT 解压到 `data/sift/` |
| 学习策略 | `./scripts/linux_run.sh policy` | 生成 `data/agent_io_policy.json`（需 python3） |
| 编译 | `./scripts/linux_run.sh build` | 产物在 `build/` |
| 单测 | `./scripts/linux_run.sh test` | `./build/run_tests` |
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

关闭逐查询 CSV：`--metrics-log none`

### 5. 常用环境变量

```bash
export CC=gcc CXX=g++              # 或 clang/clang++
export BUILD_DIR=build
export JOBS=$(nproc)
export AMIO_ENABLE_URING=ON        # 无 liburing 时设为 OFF
export CMP_BASE_LIMIT=100000       # compare 时 base 向量数上限
export CMP_QUERY_LIMIT=200         # compare 时 query 数上限
```

### 6. 不用脚本时的等价命令

```bash
# Debian/Ubuntu 依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake wget liburing-dev python3

cmake -B build -DAMIO_ENABLE_URING=ON
cmake --build build -j$(nproc)
./build/run_tests

# 生成索引（需先有 data/sift/sift_base.fvecs）
./build/build_index --input data/sift/sift_base.fvecs --output data/sift_base.index

# 磁盘评测
./build/eval_disk --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --index data/sift_base.index
```

### 7. Docker（可选）

```bash
docker build -t ai-agent-memory .
docker run --rm ai-agent-memory
```

容器内会下载 SIFT、编译并运行 `run_tests`。

### 8. 离线学习策略（Python）

```bash
python3 learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json
python3 learning/train_agent_policy.py --profile qwen-ai --out data/agent_io_policy_qwen.json
python3 learning/train_agent_policy.py --dump-features
```

示例 JSON 见 `learning/policies/*.example.json`。

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
