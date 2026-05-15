# 向量检索 I/O 优化参考文献 · 算法总结

> 15 篇论文的算法核心摘要，按技术方向分类整理。

---

## 一、向量检索缓存与 I/O 优化（4 篇）

### 1. GoVector — 静态-动态混合缓存 + 向量相似性重排
**来源**: GoVector: An I/O-Efficient Caching Strategy for High-Dimensional Vector Nearest Neighbor Search

**核心问题**: 磁盘型图索引（如 DiskANN）中 I/O 占查询延迟 90%+，现有静态缓存仅在搜索第一阶段有效，第二阶段命中率仅 4%-9%。

**算法要点**:
- **静态缓存**: 预加载入口节点及其多跳邻居，加速第一阶段的快速收敛导航
- **动态缓存**: 第二阶段自适应保留查询路径上被访问的节点及其相似邻居页面
  - 基于"环形区域"观察：第二阶段扩展节点聚集在查询向量周围一个窄距离范围内 [d_min, d_max]
  - 缓存未命中时，批量加载扩展节点所在页面 + 相邻页面，利用空间局部性
  - 淘汰策略：LFU（默认）
- **向量相似性重排（Similarity Reordering）**:
  - 第一阶段：k-means 将向量按欧氏距离聚类
  - 第二阶段：在簇内按图拓扑局部重排，使相似向量位于同一或相邻磁盘页
- **阶段切换检测**: 参数 θ（0<θ<1），当候选队列中前 θ·k 个已访问时切换到第二阶段

**效果**: 90% recall 下 I/O 减少 46%，吞吐提升 1.73×，延迟降低 42%

---

### 2. TC-HNSW — 分层分类缓存
**来源**: Tiered Cache-HNSW: Using Hierarchical Caching System in HNSW

**核心问题**: 数据集增大后 HNSW 搜索速度显著下降。

**算法要点**:
- 按访问频次和类别将预分类数据集分层缓存
- 高频访问节点优先缓存到更快存储层
- 基于类别（categorization）的缓存策略：同类向量聚在一起缓存

**效果**: 指定搜索主题时速度提升 4 倍

---

### 3. SHINE — 分离式内存下的可扩展 HNSW
**来源**: SHINE: A Scalable HNSW Index in Disaggregated Memory

**核心问题**: 单机 HNSW 无法扩展到十亿级向量；传统分区方法破坏图结构导致精度下降。

**算法要点**:
- **图保留索引**: 不分区图，保留所有边，达到与单机 HNSW 相同精度
- **分离式内存架构**: 计算节点（CPU 多、内存少）+ 内存节点（内存大、无计算），通过 RDMA 互连
- **缓存机制**: 计算节点缓存高频访问的向量数据，缓解网络带宽瓶颈
- **逻辑组合缓存**: 将多个计算节点的缓存逻辑合并，提升整体命中率

**效果**: 可扩展至任意多内存节点，保持单机 HNSW 精度

---

### 4. Graph Reordering — 图重排优化缓存命中率
**来源**: Graph Reordering for Cache-Efficient Near Neighbor Search (NeurIPS 2022)

**核心问题**: HNSW 遍历时内存访问模式差，~40% 查询时间花在内存访问向量数据上。

**算法要点**:
- 将图遍历问题建模为**缓存命中率最大化任务**
- 图重排（Graph Reordering）：将经常一起被访问的节点在内存中相邻排列
- 数学形式化图布局与搜索缓存复杂度之间的联系
- 多种重排算法对比：BFS、RCM（Reverse Cuthill-McKee）、METIS 等

**效果**: 查询时间提升最高 40%，重排时间相比索引构建可忽略

---

## 二、LSM-Tree 写优化与 Compaction（4 篇）

### 5. GroupCache — 基于 KV Group 的范围扫描缓存
**来源**: Improving Range Scan Performance in LSM-trees with Group Caching (SIGMOD 2026)

**核心问题**: LSM-tree 多层结构导致范围扫描效率低，现有块缓存/查询缓存内存效率差。

**算法要点**:
- **KV Group**: 定义为一次查询中某 block 内需要的最小 KV 子集，作为基本缓存单元
- **大小感知策略（Size-aware Policy）**: 优先缓存小而高价值的 KV Group
  - 关键洞察：消除上层 LSM-level 的 I/O 只需缓存少量 KV，而下层需要缓存更多
  - 单位内存 I/O 节省最大化
- **实际挑战处理**: Compaction 管理、组内热度差异、可扩展性

**效果**: 同内存预算下查询快 3×，或同性能用 75% 更少空间

---

### 6. NV-Cache — NVM 加速 LSM-tree 写路径
**来源**: Improving Write Performance for LSM-tree-based Key-Value Stores with NV-Cache

**核心问题**: DRAM-SSD 速度差导致写停顿（write stall）；Compaction 导致写放大。

**算法要点**:
- **分裂 LSM-tree**: 高频访问的高层放 NVM，冷数据大层放 SSD
- **关键技术**:
  - 非排序 LSM-tree 节点布局：降低 NVM 上排序开销
  - 轻量 MemTable 刷写：利用 NVM 字节寻址直接持久化
  - 写最优列表 Compaction（Write-optimal List Compaction）：NVM 上链表式合并
  - LRU 淘汰策略管理 NV-Cache

**效果**: 防止写停顿，写减少 2.3×，整体性能比 LevelDB/NoveLSM/MatrixKV 提升 54%

---

### 7. NobLSM — 无阻塞写的 LSM-tree
**来源**: NobLSM: An LSM-tree with Non-blocking Writes for SSDs

**核心问题**: LSM-tree 频繁调用 fsync 保证崩溃一致性，fsync 是阻塞操作严重拖慢性能。

**算法要点**:
- **核心发现**: Ext4 日志系统（data=ordered 模式）通过异步提交（asynchronous commit）已隐式持久化文件
- **NobLSM 设计**: LSM-tree 与 Ext4 协作，用非阻塞异步提交替代大部分 fsync
  - 每个 KV pair 只需一次 sync，后续不再重新持久化
  - 利用 Ext4 的日志机制保证崩溃一致性
- **关键机制**: 异步提交替代阻塞 sync

**效果**: 在普通 SSD 上吞吐显著超过现有 LSM-tree

---

### 8. RemapCom — SSD 重映射减少 Compaction 写放大
**来源**: RemapCom: Optimizing Compaction Performance of LSM Trees via Data Block Remapping in SSDs

**核心问题**: LSM Compaction 中大量数据块未被修改但仍被读写回去（Unchanged Data Blocks, UDB），造成写放大。

**算法要点**:
- **UDB 识别**: 轻量级状态机跟踪每个数据块中 KV 项的修改状态
- **UDB 保留策略**: 防止相邻交叉块导致 UDB 被拆分，提高 UDB 比率
- **SSD 重映射原语**:
  - `getLPN`: 获取逻辑页号
  - `remap`: 将新逻辑地址映射到 UDB 的旧物理地址，避免重复写入
- **惰性回写方案（Lazy Write-back）**: 辅助状态跟踪

**效果**: 写放大最高减少 53%

---

## 三、缓存替换与预取策略（4 篇）

### 9. PAIC — 预取自适应智能缓存替换
**来源**: A Prefetch-Adaptive Intelligent Cache Replacement Policy Based on Machine Learning

**核心问题**: 硬件预取与缓存替换策略之间存在干扰；现有替换策略不区分预取请求和需求请求。

**算法要点**:
- **双预测器**: 基于整数支持向量机（ISVM），分别为预取请求和需求请求设计独立预测器
- **DMINgen**: 模拟 Demand-MIN 算法生成训练标签
  - 区分四种使用区间：D-D、P-D、P-P、D-P
  - Demand-MIN 优先驱逐以预取结尾的区间（D-P、P-P）
- **PC 历史寄存器（PCHR）**: 维护最近 k 个唯一 PC（k=5），提取特征
- **三级插入优先级**: RRPV=0（cache-friendly）/ RRPV=2（低置信度友好）/ RRPV=7（cache-averse）

**效果**: 单核比 LRU 提升 37.22%，四核提升 20.99%

---

### 10. SARC — 序列预取自适应替换缓存
**来源**: SARC: Sequential Prefetching in Adaptive Replacement Cache

**核心问题**: 顺序工作负载与随机工作负载混合时，预取与缓存替换交互存在异常（类似 Belady 异常）。

**算法要点**:
- **自适应分区**: 动态将缓存空间在顺序流和随机流之间分配
- **异常修复**: 揭示并修复顺序预取集成缓存时的异常行为
- **自调优**: 无需手动参数调整，根据工作负载自动适应
- **实现**: IBM Shark 存储控制器硬件上实现

**效果**: 峰值吞吐下平均响应时间 5.18ms（对比 LRU 变体的 33.35ms / 8.92ms）

---

### 11. Baleen — ML 驱动的闪存缓存准入与预取
**来源**: Baleen: ML Admission & Prefetching for Flash Caches (FAST 2024)

**核心问题**: 闪存缓存需限制写入速率以保证 SSD 寿命，传统准入策略未充分利用预取。

**算法要点**:
- **Episode 模型**: 新的缓存驻留模型，指导 ML 训练
  - 解决早期 ML 策略的"痛苦教训"
- **协调 ML 准入 + 预取**:
  - 准入策略：ML 判断哪些对象值得写入闪存
  - 预取策略：ML 预测即将需要的对象提前加载
- **优化目标**: Disk-head Time（端到端系统指标），比 IO miss rate / byte miss rate 更准确反映后端负载

**效果**: 峰值 Disk-head Time 降低 12%；Baleen-TCO 总拥有成本降低 17%

---

### 12. 基于机器学习的固态缓存优化 — 一次性访问排除
**来源**: 基于机器学习的固态缓存优化研究（易锌波）

**核心问题**: 社交网络工作负载下 61% 的图片仅被访问一次但仍被写入缓存，浪费 SSD 写入寿命。

**算法要点**:
- **一次性访问准则（One-time-access Criteria）**: 识别只访问一次的对象
- **一次性访问排除策略（One-time-access-exclusion）**: 预测并阻止一次性对象进入缓存
- **预测分类器**:
  - 非历史导向预测（首次访问即判断，无历史信息）
  - 集成决策树，提取社交相关特征
  - 历史表 + 代价敏感学习 + 每日模型更新

**效果**: 预测准确率 >80%，缓存性能在多数指标上显著提升

---

## 四、LLM 缓存与通用缓存管理（3 篇）

### 13. IC-Cache — LLM 上下文缓存服务系统
**来源**: IC-Cache: Efficient Large Language Model Serving via In-context Caching (SOSP 2025)

**核心问题**: 70%+ 的用户请求存在语义相似的历史请求，但直接缓存复用会导致质量下降。

**算法要点**:
- **上下文缓存**: 利用大模型的历史请求-响应对作为小模型的 in-context 示例
  - 小模型通过模仿大模型达到甚至超越大模型的组合推理能力
- **选择性卸载**: 将部分请求路由到小模型，降低成本和延迟
- **效用感知示例选择**: 为新请求高效选择高相似度、高价值的示例附加到输入
- **自适应路由**: 根据响应质量和服务负载跨不同能力 LLM 路由请求
- **代价感知缓存重放**: 离线优化示例质量，最大化在线缓存效用

**效果**: 吞吐提升 1.4-5.9×，延迟降低 28-71%，不损害响应质量

---

### 14. Cache What You Need to Cache — 智能缓存选择
**来源**: Cache What You Need to Cache (ToS 2020)

**核心问题**: 传统缓存策略缺乏对数据访问模式差异的感知。

**算法要点**:
- 提出"一次性访问排除"策略（与论文 15 相似技术路线）
- 基于访问模式分类，只缓存真正会被再次访问的数据
- 代价敏感的缓存准入决策

---

### 15. Learning-based Dynamic Cache Management — 云环境 ML 缓存管理
**来源**: Learning-based Dynamic Cache Management in a Cloud

**核心问题**: 云存储系统中缓存管理需适应动态多变的工作负载。

**算法要点**:
- 基于机器学习的动态缓存管理策略
- 在云环境中自适应调整缓存配置
- （PDF 文本提取质量问题，细节有限）

---

## 总结对照表

| # | 论文 | 核心算法 | 技术方向 | 关键指标提升 |
|---|------|---------|---------|-------------|
| 1 | GoVector | 静态+动态混合缓存 + 相似性重排 | 向量检索 I/O | I/O↓46%, 延迟↓42% |
| 2 | TC-HNSW | 分层分类缓存 | 向量检索缓存 | 速度↑4× |
| 3 | SHINE | 图保留索引 + 逻辑合并缓存 | 分布式向量检索 | 保持单机精度 |
| 4 | Graph Reordering | 图重排最大化缓存命中 | 向量检索缓存 | 查询↑40% |
| 5 | GroupCache | KV Group + size-aware 策略 | LSM 范围扫描 | 查询↑3× |
| 6 | NV-Cache | NVM 分裂 LSM-tree + 轻量刷写 | LSM 写优化 | 写↓2.3×, 整体↑54% |
| 7 | NobLSM | 异步提交替代 fsync | LSM 写优化 | 消除写阻塞 |
| 8 | RemapCom | UDB 重映射 + 保留策略 | LSM Compaction | 写放大↓53% |
| 9 | PAIC | 双 ISVM 预测器 + DMINgen | 缓存替换 | 单核↑37.22% |
| 10 | SARC | 自适应顺序/随机分区 | 缓存预取 | 响应时间 5.18ms vs 33.35ms |
| 11 | Baleen | Episode 模型 + 协调 ML 准入预取 | 闪存缓存 | 峰值负载↓12%, TCO↓17% |
| 12 | 易锌波 | 一次性访问排除 + 决策树分类 | SSD 缓存 | 准确率>80% |
| 13 | IC-Cache | 上下文缓存 + 自适应路由 | LLM 服务 | 吞吐↑1.4-5.9× |
| 14 | Cache What You Need | 访问模式分类准入 | 缓存策略 | — |
| 15 | Learning-based Cache | ML 动态缓存管理 | 云缓存 | — |

---

## 对 proj47 (agent-memory-io) 的技术映射

| proj47 任务 | 可参考论文 | 核心借鉴点 |
|-------------|-----------|-----------|
| T01 存储布局 | GoVector (相似性重排), Graph Reordering | 向量聚类 + 图拓扑局部重排 |
| T02 图感知缓存 | GoVector (混合缓存), TC-HNSW, SHINE | 两阶段缓存策略 + 高层常驻 |
| T03 io_uring 后端 | SARC (批量预取) | 顺序预取 + 自适应分区 |
| T04 拓扑预取 | PAIC (双预测器), Baleen (ML预取) | 区分预取/需求请求 + Episode 模型 |
| T05 LSM 写优化 | NV-Cache, NobLSM, RemapCom, GroupCache | NVM 分层 + 异步提交 + UDB 重映射 |
| T06 集成评测 | 各论文评估方法 | Recall@10, QPS, P99 延迟 |
