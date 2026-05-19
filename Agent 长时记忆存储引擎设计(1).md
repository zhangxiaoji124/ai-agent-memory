# **面向大模型Agent动态长时记忆的高性能底层I/O优化向量存储引擎架构设计**

### **文档导读**

| 部分 | 章节 |
| :---- | :---- |
| 背景与目标架构 | 赛题背景 → 顶层系统架构 → 读优化（一）（二）→ 写优化 → 透明集成 → 性能评估 |
| **工程落地（必读）** | [待实现清单 R1–R12](#待实现功能清单设计--工程落地路线图) · [bvec 500G/100G](#大规模-bvec-向量集与-500gb-数据--100gb-内存运行模型) · [区分器 M0–M4](#内存划分模式区分器memory-partition-discriminator) · [VectorDataset / 数据流](#统一数据平面vectordataset-与存储布局演进) · [Phase 1–5](#实施阶段建议排期) |
| 实现差距 | [工程实现状态与仓库对照](#工程实现状态与文档对照仓库-agent-memory-io-cpp) · [`docs/Agent长时记忆-实现状态对照.md`](docs/Agent长时记忆-实现状态对照.md) |

---

## **赛题背景与架构挑战深刻剖析**

在大语言模型（LLM）Agent的实际工业级应用场景中，长时记忆（Long-term Memory）系统是赋予模型持续进化、深度推理与个性化交互能力的核心基础设施。与传统的文档检索或关系型数据库不同，Agent的长时记忆数据呈现出高度的高维流形特性与极端的时效动态性。在人机交互的整个生命周期中，Agent会持续不断地产生新的经验记忆并触发实时的高并发写入，同时在执行复杂规划与推理时，又会以极高的频次向这些记忆发起基于语义相似度的多跳检索与召回。

当这些海量的高维向量数据被组织成图索引结构（例如分层可导航小世界图，HNSW）并下沉至固态硬盘（SSD）等非易失性外部存储设备时，传统基于操作系统的标准I/O文件机制与页面管理策略面临着全面的失效，形成了一道难以逾越的“I/O性能墙”。这一性能瓶颈的本质根源，可以归结为以下三个维度的系统性矛盾：

第一，传统操作系统页面缓存（Page Cache）的空间局部性假设与高维图路由的离散性之间存在不可调和的数学矛盾。操作系统的预读与缓存机制建立在单维逻辑块地址（LBA）的连续性之上，默认程序会顺序读取相邻的磁盘数据块。然而，HNSW图索引的近似最近邻（ANNS）检索本质上是在高维空间中执行贪婪路由（Greedy Routing）。这种路由在物理磁盘上的访问轨迹表现为极端离散、不可预测的随机跳跃。因此，操作系统默认的顺序预读不仅无法命中实际需求，反而会将大量无用数据拉入内存，引发极其严重的缓存污染，导致Page Cache的命中率呈断崖式下跌，最终致使检索线程因频繁的同步磁盘等待而被彻底阻塞 1。

第二，高并发的实时随机写入与图索引的持久化重构之间存在严重的I/O争用。为了维持图索引的拓扑连通性与召回精度，实时插入的向量需要触发复杂的邻居节点重连操作。如果将这些状态更新直接穿透至底层文件系统，将引发灾难性的随机写延迟（Random Write Latency）。工业界通常借鉴日志结构合并树（LSM-Tree）的思想，将随机写转化为对SSD的顺序追加写。然而，在LSM-Tree的后台合并（Compaction）过程中，需要将大量多层级的数据文件（SSTable）读入内存进行解包、归并排序与重新刷盘。这一过程会产生剧烈的写放大（Write Amplification），过度消耗有限的SSD编程/擦除（P/E）周期，并使得底层的I/O队列高度拥塞，最终导致前台检索的读延迟发生严重的非确定性抖动 1。

第三，在内存受限环境下的“预取-缓存”拮抗效应。本架构设计的核心约束条件要求系统在仅占数据集大小10%-20%的极低内存配额下运行。在此苛刻的约束下，主动的I/O预取（Prefetching）与被动的缓存替换（Caching）形成了一对矛盾体。盲目的预取会迅速耗尽宝贵的内存预算，将极具价值的高频热点数据强行驱逐出局；而完全不进行预取，则无法隐藏物理磁盘的访问延迟，无法满足高并发下的查询吞吐量（QPS）要求。这种平衡的丧失会直接导致召回率（Recall）跌破85%的及格线，并引发系统吞吐量的雪崩 1。

综上所述，打破传统操作系统的I/O性能墙，必须在用户态彻底重构从缓存管理、预取调度到异步合并写入的全链路I/O机制。本报告将详细论述一种专为Agent动态记忆场景设计的高性能底层存储引擎架构方案，通过深度融合机器学习自适应替换、图拓扑空间重排、块级物理重映射以及非阻塞异步提交机制，在受限内存下实现读写混合负载的极致优化。

## **顶层系统架构设计与资源拓扑全景**

为实现对传统I/O路径的全面超越，所设计的向量存储引擎架构在逻辑上被划分为三个高度协同的核心子系统：基于阶段感知与特征提取的用户态智能读缓存子系统、基于物理图重排与机器学习门控的拓扑感知异步预取子系统，以及基于零拷贝块重映射的非阻塞日志结构写优化子系统。

引擎的底层存储介质依托于支持高级异步I/O框架（如Linux内核的 io\_uring 接口）及可提供逻辑-物理地址重映射（L2P Remapping）指令的现代NVMe SSD。系统向上则提供一层标准化的向量操作API封装，确保对主流Agent应用框架（如LangChain、LlamaIndex等）的绝对透明性与平滑集成能力。

在核心的物理内存（10%-20%限制）资源调度上，架构采取了精细化的隔离与动静切分策略。总内存预算被划分为三大功能区：分配约10%的配额用作写缓冲（MemTable），以LSM-Tree的机制吸收高频瞬时的写入洪峰并作为最高效的热数据读缓存；分配约20%用于构建“静态拓扑缓存（Static Cache）”，强制锚定图索引的全局导航入口点及最顶层核心邻居网络，杜绝关键路径的缓存颠簸；剩余的70%则全部注入“动态相似性缓存（Dynamic Cache）”，并通过内置的轻量级支持向量机（ISVM）与上下文多臂老虎机（Contextual Bandit）模型驱动其进行自适应的精细化流转。这种架构层面的物理与逻辑隔离，从根本上阻断了不同生命周期数据之间的缓存污染。

> **落地说明**：三区比例由 **内存划分模式区分器**（M0–M4）在进程启动时根据向量文件类型与体量自动选择；十亿级 **bvec**（如 500GB 数据 / 100GB 内存）固定走 **M3/M4**，详见下文「大规模 bvec」与「内存划分模式区分器」章节。

## **读优化（一）：颠覆传统Page Cache的用户态智能混合缓存架构**

面对极度离散的向量检索随机读，传统的操作系统Page Cache显得极其低效。本架构放弃了OS层的缺省机制，在用户态自主实现了一套针对向量图路由特征深度定制的高效缓存池。该缓存池的设计涵盖了宏观的搜索阶段感知、微观的空间粒度控制，以及时序层面的智能替换逻辑。

### **动静混合的阶段感知缓存架构 (Phase-Aware Hybrid Caching)**

在大规模图索引的近似最近邻搜索（ANNS）过程中，查询路径在数学几何空间上表现出明显的“两阶段”特征。阶段一被称为“快速收敛期（Rapid Convergence Phase）”，算法从固定的或随机选取的全局入口节点出发，利用图结构中的长程边跨越巨大的向量空间，快速逼近查询向量所在的宏观邻域。阶段二被称为“细粒度探索期（Fine-grained Exploration Phase）”，此时算法已抵达目标局部空间，转而展开多跳的集束搜索（Beam Search）或贪婪扩展，以精确圈定Top-K邻居 1。

研究表明，如果在统一的缓存池中同时处理这两个阶段的数据请求，极易导致缓存失效。这是因为第一阶段涉及的节点虽然被高频访问，但单次查询中仅涉及少数几个；而第二阶段则会在极短时间内引发大量密集但低复用率的邻居读取。为此，架构借鉴了GoVector模型的设计思想，引入了动静混合缓存策略（Static-Dynamic Hybrid Caching） 1。

在快速收敛期，系统依赖“静态缓存”。在初始化时，引擎主动提取HNSW图的高层网络（即入口节点及其最初的几层多跳邻居），按照预设的容量阈值将其长久预置于内存中。这保证了每一次新的检索启动时，都能以纯内存速度完成最初始的空间定位跨越，大幅削减了导航初期的随机I/O盲搜。

当搜索进入第二阶段时，引擎切换至“动态缓存”。动态缓存负责自适应地捕获查询路径上的新节点及其周边关联空间。然而，准确识别阶段切换的转折点至关重要。传统的识别方法往往存在严重的迟滞，导致动态缓存介入过晚而产生大量的Cache Miss。为消除这一延迟，架构引入了一个可学习的动态参数 ![][image1] (![][image2])。引擎在运行初期对少量查询进行采样，记录真实到达Top-K最近邻的时间（迭代轮数 ![][image3]），并与算法理论预估的收敛轮数 ![][image4] 进行比对，计算出 ![][image5] 1。在后续的在线检索中，系统利用 ![][image1] 提前介入，极大前置了混合缓存模式的切换节点，使得动态缓存能够尽早拦截底层磁盘的穿透。

### **KV组与范围访问不对称性优化 (Group Cache & Range-Access Asymmetry)**

解决了搜索阶段的宏观调度后，架构在动态缓存的物理存储粒度上进行了根本性革新。传统块缓存（Block Cache）要求无论查询涉及多少向量，都必须将整个物理块（如4KB或以上）加载并驻留在内存中。当底层采用多层LSM-Tree结构存储图索引数据时，这种方式存在极其严重的浪费。

在多层LSM-Tree架构中，存在一个关键的“范围访问不对称性（Range-access Asymmetry）”物理现象。由于每一层的数据量呈指数级递增（例如以容量放大倍数 ![][image6] 递增），针对特定范围的查询所匹配的键值对（KVs）数量在不同层级分布极其不均。顶层的数据文件（Sorted Runs）通常只包含极其稀疏的相关向量，而最底层的文件则包含大量的密集匹配项 1。消除一个上层磁盘I/O只需要缓存很少的向量，而消除一个底层I/O则需要缓存海量数据 1。传统Block Cache或将完整查询结果全部缓存的Range Cache无视了这一不对称性，平等地对待所有I/O，导致缓存被底层的海量结果塞满，造成严重挤兑。

为深度利用这一物理规律，本架构引入了以“向量组（KV Groups）”为最小动态管理单元的缓存设计——Group Cache。KV Group被定义为：在单次检索涉及的一个特定数据块中，仅提取那些真正满足条件及拓扑需要的连续相邻向量构成的一个逻辑集合 1。这彻底打破了必须完整缓存4KB物理块的限制。

结合这种极细粒度的动态分离，架构摒弃了盲目的LRU，转而采用一种大小感知的淘汰策略（LRU-S, Size-aware LRU）。由于上层的KV Group体积微小但具有独立免除一次磁盘I/O的高效用，LRU-S策略在分配内存时，会优先截留和驻留这些体积小、命中收益极高的“上层微型组”，而对占据巨大空间但单位体积I/O免除率低下的“底层巨型组”采取更激进的淘汰动作 1。在实现并发控制时，为防止缓存链表的全局锁竞争，引擎还对LRU链表进行了部分分片（Partial Sharding），基于KV Group起始键值的哈希进行分发，兼顾了精准的全局淘汰顺序与极高的多线程可扩展性。

### **PAIC：基于机器学习的自适应预取感知淘汰策略**

Group Cache解决了缓存粒度的空间分配问题，而针对时序层面更为复杂的“真实命中”与“无效预取”冲突，架构部署了PAIC（预取自适应智能缓存替换，Prefetch-Adaptive Intelligent Cache Replacement Policy）技术机制 1。

在异步预取被激活的系统中，内存不仅要服务CPU的立即需求（Demand Requests），还要容纳后台提前拉取的预取需求（Prefetch Requests）。大量的历史研究表明，向量检索中很大一部分预取最终未能抵达内存的有效生命周期，属于在LLC（末级缓存）或软件缓存池中完全未被触碰的“死数据”1。如果不加区分地对待两者，激进的预取不仅不能加速检索，反而会将极其重要的真实需求历史特征强行冲刷掉。

为此，PAIC构建了基于整数支持向量机（ISVM, Integer Support Vector Machine）的双子预测器网络——分别针对真实需求与预取需求进行建模。PAIC在硬件或高层软件态维护了一个微型程序计数器（或查询哈希）历史寄存器（PCHR, PC History Register），用于记录最近发生的检索行为上下文。模型训练的理论基础并非简单的最近最少使用（LRU），而是基于更先进的 Demand-MIN 算法推演。系统通过重构事后访问的历史区间（将其分为四类模式：D-D真实到真实、P-D预取到真实、P-P预取到预取、D-P真实到预取），精确量化每一次驱逐操作对未来真实命中的损益 1。

具体而言，能够导向 P-D（预取后被真实访问）的数据被奖励，而陷入 P-P 或 P-Open（无尽的未命中预取）的行为模式则受到惩罚。在推理时，由于采用了线性的整数计算，每次缓存决策仅涉及简单的向量内积与阈值比较，其计算延迟能够完美隐藏在图节点高维距离计算的间隙之中。通过这种区分特征、细粒度惩罚无效预取行为的机器学习淘汰机制，引擎能在仅有不到原占用空间三分之一的超小物理开销（约29KB状态存储）下，提供极高精准度的替换决策 1，最大化降低I/O下发次数。

| 优化维度 | 传统机制缺陷 | 本架构解决方案 (读缓存子系统) | 效果与提升理论 |
| :---- | :---- | :---- | :---- |
| **阶段特征** | 全程混用，缓存颠簸 | **GoVector动静混合**：静态驻留入口节点，动态管理细粒度局部探索 | 确保前置路由纯内存极速响应，准确切分生命周期 |
| **空间粒度** | 锁定4KB Page，充斥无效向量 | **Group Cache 向量组**：基于范围不对称性，以KV Group为单元提取有效片段 | 避免底层庞大冗余结果污染内存，同等内存承载量跃升数倍 |
| **时序淘汰** | 盲目LRU导致预取污染真实数据 | **PAIC自适应机制**：ISVM双模型区分Demand与Prefetch，依据Demand-MIN训练 | 精准剔除“死预取”向量，内存利用率逼近理论极限最佳点 |

## **读优化（二）：拓扑感知、物理重排与非阻塞智能预取机制**

被动的内存拦截缓存虽然极大降低了I/O次数，但在严苛的内存配额下必然存在未命中现象（Cache Misses）。此时，必须依靠主动的磁盘预取技术。传统存储系统的顺序LBA预读对于散乱的向量图拓扑检索毫无用处，必须打通图索引逻辑拓扑与SSD物理存储介质的深层关联。

### **向量图结构底层布局重排 (Graph Reordering)**

在标准的LSM-Tree写入流程中，向量与节点数据是按照时间插入顺序（Insertion-order）追加落盘的。这种布局忽略了节点在高维空间中的距离属性和互相连通性，导致在图中执行一跳紧密相邻的寻路时，却需要在磁盘上进行跨越多个物理页的长距离随机寻道。

为了化解这种物理不连续性带来的巨大惩罚，本引擎在后台合并与初始化重组时，深度融合了图重排（Graph Reordering）技术 1。其核心思想是将逻辑图上连接紧密、被联合访问概率极高的节点族群，强制压缩进相邻的或者同一个4KB物理页（SSD的最小读取单元）之中。架构吸收了Gorder、Corder与Porder等前沿重排算法的优势 1。

以Gorder算法为基础，系统在后台建模一个滑动窗口内的节点收益优化函数，力求最大化被打包在同一个物理块内的节点之间的共享边数量与公共邻居数量。而在更为复杂的长时记忆访问中，部分历史向量的活跃度远超其他节点。因此，引擎在此基础上引入了Porder（Profile-guided Reordering）模型。系统在后台定期追踪并采样高频交互的边缘跳跃流量，为图的边赋予不同的权重（Weights），从而将“拓扑近邻”与“时序高频近邻”结合，执行加权最大化分配组合 1。经过此类重排后，原本零散的每一次随机I/O，都能附带拉取大量未来可能访问的关联节点簇，使得单次磁盘I/O的数据利用效率实现了实质性的突破。

### **Episode缓存驻留模型与io\_uring并发拉取**

为了将重排优化与异步框架结合，引擎引入了Episodes概念模型（源自Baleen系统的研究）1。Episode被定义为与一个数据块生命周期相关联的连续访问事件组合。系统通过抽象评估各个节点的Episode停留长度与时间间隔，构建了一套更适用于闪存缓存离线评估与预取判断的分析框架 1。

依托于Linux最新的高级非阻塞I/O接口——io\_uring，存储引擎设计了彻底无锁化的下一跳并发预取流水线。io\_uring 采用共享环形缓冲区（Submission/Completion Queues），彻底避免了由于大量系统调用（Syscalls）和内核上下文切换带来的高昂CPU开销。在主计算线程加载当前批次的向量并使用AVX-512/AMX指令集进行大规模SIMD高维距离计算时，I/O调度器在后台并发地向 io\_uring 提交前瞻性读取请求。

### **基于ML-When与ML-Range的预取门控节流机制**

毫无节制的并发预取会迅速打爆SSD的吞吐带宽极限（Bandwidth Wall）。预取过浅（Underfetch）会导致CPU陷入I/O等待，而预取过深（Overfetch）则不仅浪费带宽，还会挤占其他有效查询的I/O响应通道并引发缓存覆盖灾难 1。因此，在将预取请求推送至 io\_uring 之前，引擎内嵌了一个极轻量级的多变量决策阀门，结合ML-Range和ML-When思想实现动态节流。

首先，ML-Range模块通过分析当前贪婪搜索在图结构中所处的层级深度以及候选队列的离散程度，输出一个自适应的预取范围边界，动态决定拉取相邻节点的最优深度 1。 其次，ML-When模块进行成本收益评估。引擎定义了一个量化指标——“基于磁盘磁头时间的边际预取收益（Marginal DT Gained）”。只有当模型预测“提前拉取该节点集合所能节省的磁盘等待时间”明确大于一个被定义为 ![][image7] 的机会成本阈值（该阈值用于表征消耗当前有限闪存带宽带来的宏观损耗风险）时，预取才会被真正授权执行 1。在遇到突发性高并发读取导致系统整体负载飙升时，引擎会借鉴多臂老虎机（Contextual Multi-Armed Bandit）理论 1，实时计算系统负载的指数移动平均（EMA）。当负载越过安全红线时，通过基于双曲正切函数（tanh）的反馈控制回路，动态平滑地降低 ![][image7] 阈值的门限宽容度，强制削弱预取力度，确保在极端压力下读请求依然不发生雪崩。

## **写优化：高并发下的无阻塞异步追加与零拷贝LSM重整机制**

大模型Agent不仅依赖对历史的高频召回，同时也会在长周期的多轮对话中源源不断地生成新记忆（事实插入与更新）。传统的HNSW图索引更新是典型的高代价原位（In-place）随机写，若直接映射到SSD，则会因为NAND Flash异地更新的物理限制引发极其严重的底层擦除损耗，其高昂的锁竞争延迟更是灾难性的。

本架构采用了在键值存储与分布式数据库中已被广泛验证的LSM-Tree思想，将所有并发的实时写入请求截流在内存的MemTable中，积累到一定阈值后以批量顺序追加的形式落盘，从根本上缓解了并发插入瓶颈。然而，原生LSM-Tree为了清除垃圾并维护多层有序性，必须执行繁重的后台合并操作（Compaction）。这种合并操作会引起大规模的数据反复读写（Write Amplification），并严重侵占SSD的前台查询带宽。

为打破这一桎梏，本架构结合了NVM、底层重映射（L2P Remapping）及文件系统协同技术，进行了一场深度的写入重构。

### **NVTable：面向NVM与轻量化内存刷写的日志融合设计**

在传统的LSM-Tree架构中，为了防止断电引发的内存数据丢失，所有写入操作必须先强制追加至磁盘的预写日志（Write-Ahead Log, WAL）中。这个前置I/O虽然是顺序写，但依然拖慢了超高并发的响应速度。

若服务器配备了非易失性内存（NVM），或者系统采用掉电保护型内存缓冲池，架构将采用NV-Cache的进阶设计 1。通过深度利用字节寻址特性，架构设计了一种称之为“NVTable”的无序链表组织格式 1。新向量的插入仅是对NVTable链表的简单追加，从而**彻底消灭了独立的WAL开销（Write-ahead-log-free）** 1。为了不影响读取速度，系统会在后台轻量级地为NVTable挂载一个偏移量数组（Offset Array），使得二分查找可以无缝衔接。 当内存中的MemTable写满转变为Immutable状态触发向L0层SSD刷写时，架构同样进行列表级合并（List Compaction）。相比传统庞大的键值拷贝重写，该机制仅仅是调整更新底层的八字节指针链路来完成逻辑上的合并 1，极大地降低了数据在缓存层间的倒腾成本。

### **RemapCom：基于零拷贝块重映射彻底消减合并写放大**

针对必须在多层SSD文件之间执行重整与合并的核心矛盾，架构深度集成了基于底层块重映射的合并优化技术（RemapCom） 1。

在LSM合并归并阶段，大量旧文件的KV键值对其实未发生任何修改（或是被删除更新），它们只是由于区间交叠被强制读取，与新数据拼接后又原封不动地重新写回另一块磁盘区域。架构创新地提出了“未变更数据块（UDB, Unchanged Data Blocks）”的识别与截留机制。

当后台进行SSTable解包合并时，内存中会实例化一个极轻量级的状态机，用于逐一追踪每个4KB数据块的处理流水。该状态机存在四种确定状态：Begin, InProgress, Changed, Unchanged 1。同时，系统启用“延迟写回（Lazy Write Back）”通道，在处理过程中先将数据缓存至微型缓冲区内。

* 如果扫描发现该数据块中有任何KV项目被标记为丢弃（Dropped）或者与其他层级数据产生了穿插混排，状态机立即判定为 Changed，此时该数据块正常写回磁盘。  
* 如果连续扫描确认直到块末尾，所有内容完整保留（Reserved）且内部顺序不变，状态机标记其为 Unchanged，即确认其为一个UDB。

更为精妙的是，为了防止纯天然UDB被由于闪存页面对齐要求而强行拼凑的相邻孤散数据打碎，架构采用了“UDB强制保留策略（UDB Retention）” 1。即便会产生极轻微的内部碎片，引擎也坚决杜绝UDB与外部数据的混合。

一旦判定为UDB，引擎绝不发出真实的物理写请求。相反，架构调用为文件系统专门定制的拓展接口指令（例如针对 nvme\_ioctl 封装的 getLPN 和 remap 原语） 1。存储引擎仅仅是将新的文件逻辑页号（LPN）告知给SSD控制器，而SSD的固件转换层（FTL）直接将这个新的逻辑页号映射到该UDB原本驻留的旧物理页框（Physical Page）上 1。 这一操作在SSD内部表现为纯粹的元数据（映射表）修改，**实现了物理层面的零拷贝（Zero-copy）合并**。多项评测研究证明，借助于这一重映射机制，系统在不同层级的合并中能够削减高达 52%-53% 的冗余写放大（Write Amplification） 1。不仅将SSD的使用寿命延长了数倍，更解放了本应用于写入的底层信道带宽，保障了混合负载下读请求的极致平稳。

### **NobLSM：打破文件系统同步屏障的非阻塞异步提交机制**

尽管RemapCom消灭了大量的无效重写，但在Compaction过程中，不可避免地会产生一批确需重新落盘的新生成的变更数据文件。为了向整个存储栈宣布这批新合并生成的SSTable已具备崩溃一致性（Crash Consistency），传统的LSM存储引擎（如RocksDB、LevelDB）必须显式调用具有极强阻塞副作用的系统函数（如 fsync 或 fdatasync） 1。这些同步调用会产生巨大的底层屏障（Barrier），强行排空OS缓存并将一切并发I/O暂停，直至数据与元数据完全落盘。这就是导致向量检索长尾延迟抖动（Latency Spike）的最终罪魁祸首 1。

为了根除这一阻塞点，架构引入了 NobLSM 的协同思想，使得存储引擎能与底层日志文件系统（如运行在默认 data=ordered 模式的Ext4）进行高级别的非阻塞交互协作 1。

在 Ext4 的 data=ordered 模式下，系统内置了一个严格的时序保证：任何文件的元数据（Inode信息）在被异步提交到系统的元数据日志区（Journal）之前，该文件所有被修改的底层数据块必须已经被预先刷入磁盘 1。这就意味着，一旦底层日志成功收录了Inode信息，该文件在崩溃后绝对安全。

引擎巧妙地利用了这一系统特性。架构向内核拓展注入了两个低开销系统调用：check\_commit 和 is\_committed。当后台合并完成一批新SSTable的异步落盘请求后，引擎根本不调用 fsync，而是立即调用 check\_commit 将这些新文件的Inode告知内核态的“等待判定表（Pending Table）” 1。 此时，合并线程立刻放行释放锁，继续处理其它图更新逻辑，丝毫不受磁盘I/O物理速度的阻塞。在底层操作系统的后台，Ext4自身的内核回写线程（Writeback Thread）会根据内存压力或固定的超时周期（如5秒间隔），从容、批量、异步地将数据安全沉淀到固态闪存，并顺带将对应的Inode提交至日志区，随即将其标记在内核态的“已提交表（Committed Table）”中 1。

在用户态，存储引擎维护一个轻量级的关联集合，定期通过 is\_committed 轮询检查。确认新文件已安全持久化后，再稳妥、非阻塞地删除冗余的历史垃圾文件 1。 凭借“底层重映射零拷贝”与“非阻塞异步日志提交”这两项绝杀技的组合，写优化子系统把原本沉重、耗时、锁争用极度惨烈的磁盘索引重整过程，彻底降维为轻量级的内存指针重定位与异步状态机轮询。在承受 Agent 高并发向量记忆持续注入的严酷环境下，检索（读）延迟呈现出近乎绝对的线性平稳，不再有任何灾难性的抖动。

## **透明集成生态与多架构硬件底层适配策略**

本设计方案不仅是一套存在于理论模型层面的极限优化内核，更被严谨地构架为可落地的通用型基础设施底座。

### **零代码修改的标准Agent生态接入层**

架构在外围封装了一层统一且透明的标准API接口。无论是 LangChain 的 VectorStore 抽象，还是 LlamaIndex 的底层集成桥，系统均暴露出原生兼容的 add\_texts(), similarity\_search\_by\_vector(), max\_marginal\_relevance\_search() 等高阶功能。

所有涉及到重排、缓存穿透计算、零拷贝合并与非阻塞调用的复杂工作，完全被隔离在用户态守护进程之下。为了进一步贴合大模型Agent的长时记忆遗忘（Forgetting）与局部上下文检索需求，接口内部支持直接将Session ID标识及时间戳随同向量载入，并在图索引边结构中携带过滤属性，使得向量距离计算与标量属性匹配在I/O提交阶段之前便能完成物理剪枝。

### **面向国产芯片与先进存储设备的泛化适配**

架构高度脱离了对于特定厂商闭源硬件指令集的深度强绑定。基于 io\_uring 环形队列驱动的并行状态机，在Intel的Hyper-Threading或AMD的多CCX架构下可自动利用无锁队列的并发优势；而在针对国产处理器（如鲲鹏ARMv8、海光等）进行编译部署时，其避免重度互斥锁与跨核中断切换的特性，同样能够发挥出极高的多核扩展能力。

针对外挂高速设备，系统特别优化了对新一代企业级非易失性存储接口的支持协议。利用存储系统的开放通道技术（Open-Channel SSD）或分区命名空间技术（ZNS SSD），架构甚至可以将上述在块设备控制器层完成的Remap命令与ZNS区域的Zone Append特性联合打通，彻底消灭传统SSD主控中繁重的FTL映射负担。这一层面的硬件深度榨取，进一步巩固了其作为未来最高规格Agent引擎基座的地位。

## **全域性能评估机制与核心技术指标理论预测**

为了验证系统在严格限制条件下的运行状态，整体性能评估被严格锁定在包含实时读写混合密集负载的沙箱测试中运行。核心考量指标囊括了吞吐率、长尾延迟以及精准度基线。基于各优化子模块数学边界与理论模型的联合测算，该架构的设计预计将在评测中展现出统治级的优势：

1. **极度受限环境下的召回率坚守 (Recall@10 \>= 85%)**  
   在仅占总数据规模10%-20%的极端内存束缚中，传统的页面驱逐会令搜索通道随时中断，丧失图连通性。本架构通过静态锁定全局入口保障绝对路由起步效率，结合PAIC对预取无效数据的智能甄别及主动剔除（规避了死数据对有效探索历史的挤兑），确保了在深度受限空间中内存资源的绝对纯净度。配合高置信度门控调度的下一跳预取（ML-Range预测），保证了搜索宽度的充足填充，从数学下界上封死了召回率崩溃的可能，确保严格且持续地稳定在85%以上的预设基准线。  
2. **抗干扰读写混合吞吐率飞升 (QPS)**  
   原有的串行随机读在此架构下被图重排（Gorder/Porder）降维成了物理区块内部的高概率顺序扫描。io\_uring深度重叠隐藏了I/O传输消耗与CPU高维距离计算。尤其是在并发写入极度活跃的高频交互高峰期，原本因原位更新和阻塞式合并导致CPU长时间处于I/O wait的恶劣状况，被零拷贝重映射（RemapCom减少一半合并代价）及非阻塞提交通信（NobLSM免除 fsync 暂停）所根除。这使得系统的计算流水线可以全速奔跑，预测读写混合状态下的QPS指标将取得较开源基线方案数倍乃至数量级层面的几何跨越。  
3. **读延迟长尾效应的终极平抑 (P99/P99.9 Tail Latency)**  
   在生产级Agent应用中，用户体感的最大破坏者往往并非平均延迟，而是偶发性的极高长尾等待。由于消灭了传统LSM-Tree中最具破坏性的文件同步屏障（Barrier），并由用户态精细化的Group Cache向量组缓冲拦截底层的高昂开销，原本不可预知的极端访问长尾被彻底平滑削顶。检索请求的响应耗时分布将呈现出高度集中的正态波峰，极大幅度地降低了高分位段下的延迟毛刺现象。

| 评估核心维度 | 传统机制性能塌陷成因剖析 | 本架构部署的核心破局技术 | 预期系统表现改善幅度 |
| :---- | :---- | :---- | :---- |
| **混合吞吐极值 (QPS)** | OS Page Cache顺序预读失败，CPU阻塞干等单一乱序I/O响应 | 物理页面图重排 (Reordering) 叠加 io\_uring 异步并发批量拉取 | I/O排队拥塞大幅消解，全时段吞吐极值跃升数倍 |
| **P99长尾检索延迟** | 内存容量瓶颈引发剧烈缺页中断；合并同步指令屏障引发读停滞 | PAIC自适应有效汰换拦截磁盘穿透；NobLSM免 fsync 非阻塞提交解绑 | 命中率在限额下逼近理论极值，P99长尾等待削减达量级水平 |
| **并发大批次更新抖动** | 索引结构强同步重组引发高倍率读写冲突与介质级写入风暴 | 提取UDB实现RemapCom原生物理零拷贝；日志态异步归并重整 | 合并写放大骤降50%以上，全面消除更新抖动峰值 |
| **强约束边界召回准确率** | 暴力随机预取与局部集束搜索爆发内讧，导致关键拓扑强行失忆 | GoVector多相动静混合拦截调度与Episodes预取置信成本门控调节 | 在10-20%内存下捍卫完整探索宽度，稳保 85% 基线不破 |

---

## **待实现功能清单（设计 → 工程落地路线图）**

> 下列条目为在上文目标架构基础上，**必须在代码库中补齐**的能力；与当前仓库差距见文末「工程实现状态」一节。

| 编号 | 模块 | 待实现内容 | 验收标准 |
| :---- | :---- | :---- | :---- |
| **R1** | 数据集 I/O | **bvec** 流式读取、mmap 视图、按条 `seek`；与 fvecs/ivecs 统一 `VectorDataset` 抽象 | 无需整文件载入 RAM 即可建索引/检索 |
| **R2** | 内存划分区分器 | 根据**文件类型 + 文件大小 + 可用内存预算**选择划分模式（见下节） | 同一二进制对不同数据集自动切换配置 |
| **R3** | 三区内存预算 | 按选定模式将 `ram_budget` 切分为 MemTable / Static / Dynamic（10% / 20% / 70% 为默认大盘模式） | 进程启动日志打印各池字节上限与模式名 |
| **R4** | GoVector 静态区 | 从磁盘索引提取入口点及高层邻居，**pin** 至 Static 池 | 冷启动首跳无磁盘读 |
| **R5** | GoVector 动态区 + θ | 在线阶段检测，动态缓存路径节点；learn/轨迹驱动策略 | Recall@10 在 20% 内存下 ≥ 85%（约定口径） |
| **R6** | bvec 距离与块布局 | `NodeBlock` 支持 `uint8` 载荷或分块向量；L2/ADC 与 float 路径统一接口 | dim=128 bvec 与 SIFT1B 子集 recall 可测 |
| **R7** | 流式建索引 | 分片扫描 bvec → 分批构图/落盘，峰值内存 < `ram_budget` | 500GB 级文件在 100GB 机器上可完成构建（允许耗时） |
| **R8** | Group Cache / PAIC | KV Group 粒度缓存；demand/prefetch 分流淘汰（可先计数 useful/wasted） | 预取污染可量化下降 |
| **R9** | 图重排 | 建索引后 Gorder/RCM 简化版重排 `node_id→LBA` | 单次 I/O 多节点命中率提升 |
| **R10** | 写路径 | RemapCom / NobLSM / NVTable 择一落地或仿真接口 | 混合写时 P99 检索抖动下降 |
| **R11** | Agent 特征 | 真实 search trace 或图邻接游走；bvec 数据集专用 learn 策略 | `agent_io_policy.json` 含 `memory_profile` 字段 |
| **R12** | 评测 | `eval_disk --format auto --ram-budget-gb 100`；500GB bvec 子集与全量口径文档 | 一条脚本产出 QPS / P99 / Recall |

---

## **大规模 bvec 向量集与 500GB 数据 / 100GB 内存运行模型**

### **bvec 文件格式（TexMex / BIGANN 系）**

与 **fvecs**（`uint32 dim` + `dim × float32`）并列，**bvec** 为：

- 每条记录：`uint32 dim`（小端）+ `dim × uint8` 分量（通常 0–255，SIFT 为 8bit 量化灰度特征）。
- 单条字节数：`record_bytes = 4 + dim`。
- 条数估计：`num_vectors ≈ file_size / record_bytes`（需用首条 dim 校验全文件一致）。

| 类型 | 扩展名 | 载荷 | 典型场景 |
| :---- | :---- | :---- | :---- |
| float 向量 | `.fvecs` | float32 | SIFT 子集、learn/query、实验复现 |
| 字节向量 | `.bvecs` | uint8 | **SIFT1B**、十亿级 Agent 记忆底库 |
| 整数 GT | `.ivecs` | int32 | groundtruth，**不参与**内存划分，仅评测 |

**当前仓库状态**：`VectorDataset` 支持 fvecs/bvecs 任意 `dim`（如 **960**）；**dim≤128** 时向量内联 `NodeBlock`（v1），**dim>128** 时向量写入索引文件尾部连续区（v2，`external_vectors.*`），图结构仍为 4KB 块。

### **500GB bvec + 约 100GB 内存：约束与目标**

| 指标 | 目标值 | 说明 |
| :---- | :---- | :---- |
| 向量原始文件 | ≈ **500GB**（`.bvecs` 或分片 `base_*.bvec`） | 不允许整表 `malloc` 进内存 |
| 进程可用内存预算 | ≈ **100GB** | 含缓存 + 建索引缓冲 + 元数据；约为数据量的 **20%** |
| 数据/内存比 ρ | ρ = `data_bytes / ram_budget` ≥ **5** | 强制 **磁盘优先（Disk-First）** 划分模式 |
| 索引形态 | 磁盘 **HNSW/NSW**（每节点 4KB `NodeBlock`）+ 可选内存 MemTable | 索引文件通常另占磁盘，规模与 `num_vectors` 成正比 |
| 召回 | Recall@10 ≥ **85%**（赛题基线，固定 `ef_search` 与评测子集口径） | 在 100GB 预算下通过 Static+Dynamic+预取保证 |

**容量粗算（dim=128 的 bvec）**

- 单条 ≈ 132B → 500GB 约 **3.8×10⁹** 条（若为单一连续 base；实际常以分片或子集评测）。
- 若仅 **10⁹** 条（SIFT1B 量级），原始 bvec ≈ **132GB**；500GB 可能含多分片、备份或更高维（如 GIST960 单条约 964B，5×10⁸ 条即超 450GB）。

**100GB 内存推荐用途（模式 `BVEC_DISK_TIERED`，见下表）**

| 池 | 占比 | ≈ 字节 | 职责 |
| :---- | :---- | :---- | :---- |
| MemTable / 写缓冲 | 10% | 10GB | 吸收 Agent 实时写入；WAL 顺序刷盘 |
| Static Cache | 20% | 20GB | 入口点 + 高层图导航节点常驻 |
| Dynamic Cache | 70% | 70GB | 查询路径热节点 + 拓扑预取回填；LRU-S / PAIC 目标落地区 |

其余 **向量主体与冷节点** 均在 SSD；依赖 **io_uring 预取 + 图重排** 隐藏延迟，**禁止**依赖 OS Page Cache 顺序预读。

### **bvec 专用读路径（待实现）**

1. **mmap 只读映射** bvec 文件，按 `record_bytes` 随机访问，避免拷贝全量。
2. **距离计算**：检索时对查询向量 float 与块内 `uint8` 做 L2（或先 float 化再算）；SIMD 批量。
3. **建索引**：流式多 pass——Pass1 统计条数/dim；Pass2 分批读入 `batch_size` 条构图并 append `NodeBlock`；峰值内存由 `batch_size` 与 `ram_budget` 共同约束。
4. **learn 策略**：`train_agent_policy.py` 增加 `--data-source learn-bvecs`（mmap 子采样 + 游走），输出 JSON 中写入 `memory_profile: "BVEC_DISK_TIERED"`。

---

## **内存划分模式区分器（Memory Partition Discriminator）**

### **设计目标**

在进程启动或 `open(base_path)` 时，根据：

1. **向量文件类型**（扩展名 / 魔数探测）；
2. **向量文件总字节** `data_bytes`（多文件则求和）；
3. **配置或 cgroup 给出的内存预算** `ram_budget_bytes`；

自动选择 **内存划分模式（MemoryPartitionProfile）**，并生成 `Config`（各池上限、是否允许全内存构图、预取激进程度等）。**禁止** 对 500GB bvec 误用「小集全内存」模式。

### **划分模式枚举**

| 模式 ID | 名称 | 适用条件（摘要） | MemTable | Static | Dynamic | 构图 |
| :---- | :---- | :---- | :----: | :----: | :----: | :---- |
| **M0** | `HOST_RESIDENT` | fvecs/小文件，且 `data_bytes ≤ 0.5×ram` | 10% | 15% | 75% | 允许全量载入内存 HNSW |
| **M1** | `HYBRID_FLOAT` | fvecs，`0.5×ram < data_bytes ≤ 2×ram` | 10% | 20% | 70% | 内存索引 + 磁盘镜像 |
| **M2** | `DISK_FIRST` | fvecs，`data_bytes > 2×ram` 或 `ρ>2` | 10% | 20% | 70% | 仅磁盘检索为主 |
| **M3** | `BVEC_DISK_TIERED` | **bvec** 且 `ρ≥3` 或 `data_bytes≥100GB` | 10% | 20% | 70% | 流式落盘索引，禁止全量载入 |
| **M4** | `BVEC_ULTRA_500G_100G` | **bvec** 且 `data_bytes≥400GB` 且 `ram_budget≈100GB`（ρ≥4） | 10% | 20% | 70% | 同 M3 + 更激进 Static pin、预取深度由 learned 策略上限 |

**默认比例**：MemTable 10% + Static 20% + Dynamic 70% = 100% `ram_budget`（不含索引文件 cache 与 io_uring 队列，二者另计上限）。

### **区分器决策逻辑（规范伪代码）**

```text
输入: paths[], ram_budget_bytes, 可选 override_profile
输出: MemoryPartitionProfile, pool_bytes{memtable, static, dynamic}, DatasetDescriptor

1. 解析类型 kind = detect_kind(path)   // FVECS | BVECS | IVECS | UNKNOWN
2. 若 kind==IVECS 且 仅 GT → 不参与划分，只挂评测
3. data_bytes = sum(file_size(path))
4. dim, num_vectors = probe_first_record(path)  // 读首条 dim，校验后续一致
5. ρ = data_bytes / ram_budget_bytes

6. 若 override_profile 已设置 → 使用该模式
7. 否则若 kind==BVECS:
       若 data_bytes≥400GB 且 ram_budget∈[80GB,128GB] → M4 BVEC_ULTRA_500G_100G
       否则若 data_bytes≥100GB 或 ρ≥3              → M3 BVEC_DISK_TIERED
       否则                                         → M2 DISK_FIRST
8. 否则若 kind==FVECS:
       若 data_bytes≤0.5×ram                         → M0 HOST_RESIDENT
       否则若 data_bytes≤2×ram                      → M1 HYBRID_FLOAT
       否则                                         → M2 DISK_FIRST
9. 否则 → M2 DISK_FIRST（保守）

10. pool.memtable = ram_budget * ratio(memtable)
    pool.static    = ram_budget * ratio(static)
    pool.dynamic   = ram_budget - memtable - static
11. 写入日志与 agent_io_policy.json 的 _meta.memory_profile
```

### **文件类型检测规则**

| 优先级 | 规则 |
| :---- | :---- |
| 1 | 扩展名：`.bvecs` → BVECS；`.fvecs` → FVECS；`.ivecs` → IVECS |
| 2 | 首记录：读 `dim`，再读 `dim` 个 float；若长度匹配且后续可重复 → FVECS |
| 3 | 首记录：读 `dim` 个 uint8；若 `4+dim` 步进全文件一致 → BVECS |
| 4 | 失败 → UNKNOWN，降级 M2，`stderr` 告警 |

### **规划中的代码落点（待实现）**

| 组件 | 建议路径 |
| :---- | :---- |
| 类型与模式枚举 | `include/runtime/memory_partition.h` |
| 区分器 + 探测 | `src/runtime/memory_partition.cpp` |
| 与 Config 集成 | `vector_store.h`：`Config::ram_budget_bytes`、`partition_profile` |
| CLI | `eval_disk` / `build_index`：`--format auto`、`--ram-budget-gb 100` |
| 策略 JSON | `memory_profile`、`pool_*_mb` 字段由 Python 脚本写出 |

### **配置示例（500GB bvec，100GB RAM）**

```json
{
  "memory_profile": "BVEC_ULTRA_500G_100G",
  "ram_budget_gb": 100,
  "data_bytes_gb": 500,
  "compression_ratio_rho": 5.0,
  "pool_memtable_gb": 10,
  "pool_static_gb": 20,
  "pool_dynamic_gb": 70,
  "vector_file_kind": "bvecs",
  "allow_full_memory_graph": false,
  "stream_index_batch_vectors": 500000
}
```

### **各模式下的运行时参数缺省（区分器输出 → Config）**

区分器选定 `MemoryPartitionProfile` 后，除三区字节上限外，还应**联动**下列运行时缺省（可被 `agent_io_policy.json` 覆盖）：

| 参数 | M0 | M1 | M2 | M3 | M4 |
| :---- | :----: | :----: | :----: | :----: | :----: |
| `allow_full_memory_graph` | 1 | 1 | 0 | 0 | 0 |
| `search_backend` | memory | hybrid | disk | disk | disk |
| `cache_size_mb`（≈Dynamic 池） | 按 75% ram | 70% | 70% | 70% | 70% |
| `static_cache_mb` | 15% ram | 20% | 20% | 20% | 20% |
| `memtable_limit_mb` | 10% ram | 10% | 10% | 10% | 10% |
| `prefetch_depth` 基线 | 1 | 1 | 2 | 2 | 2 |
| `enable_wal` | 1 | 1 | 1 | 1 | 1 |
| `stream_index_batch_vectors` | 0（全量） | 1e6 | 5e5 | 5e5 | **5e5** |
| `io_uring` 优先 | 可选 | 开 | **开** | **开** | **开** |
| `hot_insert_min_prior_misses` | 1 | 2 | 2 | 2 | **2** |
| 推荐 `ef_search` 起点 | 128 | 160 | 192 | 192 | **224** |

M4 在 M3 基础上：**Static 池优先 pin 全层 ≤2 的导航子图**（见下节），预取扇出取 learned 策略上限，评测时固定记录 `memory_profile` 与 `rho` 便于复现。

### **模式 × 向量类型决策矩阵（速查）**

|  | 数据 < 50GB | 50GB–100GB | 100GB–400GB | ≥ 400GB（如 500GB bvec） |
| :---- | :---- | :---- | :---- | :---- |
| **.fvecs** | M0（ram 充足时） | M1 | M2 | M2 |
| **.bvecs** | M2（保守） | M3 | M3 | **M4**（ram≈100GB） |
| **.ivecs** | — | — | — | 仅 GT，不参与 |

---

## **统一数据平面：VectorDataset 与存储布局演进**

### **VectorDataset 抽象（R1 目标接口）**

为统一 fvecs / bvecs / 未来分片，读路径不经过「整文件 `vector<vector<float>>`」：

```cpp
// 规划：include/dataset/vector_dataset.h
enum class VectorEncoding { Float32, UInt8 };

struct VectorRecordView {
  const void *payload;  // float* 或 uint8*
  uint32_t dim;
  VectorEncoding enc;
};

class VectorDataset {
 public:
  bool open(const std::string &path, VectorFileKind kind);
  uint64_t size_est() const;
  uint32_t dim() const;
  bool get(uint64_t i, VectorRecordView *out);  // mmap / pread
  bool iterate_batch(uint64_t begin, uint64_t count,
                     const std::function<bool(uint64_t, VectorRecordView)> &fn);
};
```

- **fvecs**：`enc=Float32`，`payload` 指向 dim 个 float。  
- **bvecs**：`enc=UInt8`，检索 QUERY 仍为 fvecs 时，在距离内核内将 query 量化或 promote 为 float 再与 uint8 做 L2。  
- **ivecs**：实现 `VectorDataset` 子类 `GroundTruthDataset`，仅用于 `compute_recall_at_k`。

### **NodeBlock 与磁盘索引布局（R6）**

当前 `NodeBlock`（4096B）硬编码 `float vector[128]`，面向 SIFT-128。**bvec / 高维**需分版本：

| 版本 | 适用 | 向量区 | 说明 |
| :---- | :---- | :---- | :---- |
| **v1（现状）** | dim≤128 fvecs | 128×float | 与现有索引兼容 |
| **v2（规划）** | dim≤128 **bvec** | 128×uint8 + 对齐 | 距离用 uint8 L2；query 为 float 时在线转换 |
| **v3（规划）** | dim>128（如 960） | **不在块内存向量** | 块内仅存 `vector_file_offset`；向量外置连续 bvec/fvec 文件，检索二次读 |

索引头 `IndexFileHeader.version` 区分 v1/v2/v3；区分器在 M3/M4 下**默认要求 v2 或 v3**，避免误用 v1 强转 float。

**索引磁盘占用粗算**（v1/v2，每节点 4KB）：

\[
\text{index\_bytes} \approx \text{num\_vectors} \times 4096
\]

十亿节点约 **4TB** 索引，与 500GB 原始 bvec **分离存储**；100GB 内存**只缓存热节点块**，不缓存全量向量文件。

### **端到端数据流（500GB bvec + 100GB RAM）**

```text
                    ┌─────────────────────────────────────────┐
  base.bvecs(500G)  │ VectorDataset (mmap, UInt8)             │
                    └───────────────┬─────────────────────────┘
                                    │ 流式 batch
                    ┌───────────────▼─────────────────────────┐
                    │ build_index_streaming                   │
                    │  HNSW 构图 → NodeBlock v2 追加 .index     │
                    └───────────────┬─────────────────────────┘
                                    │
  query.fvecs       ┌───────────────▼─────────────────────────┐
                    │ VectorStore::open                       │
                    │  ① select_memory_partition → M4         │
                    │  ② 初始化 MemTable/Static/Dynamic 池     │
                    │  ③ merge agent_io_policy.json           │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │ search_disk                             │
                    │  Static: 入口+高层 pin                  │
                    │  Dynamic: GraphAwareCache + 策略准入    │
                    │  Prefetch: TopologyPrefetcher+io_uring  │
                    │  Distance: L2(float_q, uint8_node)      │
                    └─────────────────────────────────────────┘
```

---

## **三区内存池与现有模块映射**

设计上的 MemTable / Static / Dynamic 需在 `VectorStore` 构造时**显式切分**（当前仅单一 `cache_size_mb`，待 R3 落地）：

| 逻辑池 | 规划结构 | 现有代码映射 | 待增能力 |
| :---- | :---- | :---- | :---- |
| **MemTable 10%** | 写入缓冲 + 热读 | `write::MemTable`、`memtable_limit_mb` | 与 `ram_budget` 联动上限 |
| **Static 20%** | 入口+高层常驻 | `GraphAwareCache::pin` | **启动时** `extract_static_subgraph()` 批量 pin |
| **Dynamic 70%** | 路径热块 LRU | `GraphAwareCache` hot 区 | 容量=`dynamic_cache_bytes`；未来挂 PAIC |

### **Static 子图提取算法（R4 规范）**

1. 读 `IndexFileHeader.entry_point`、`max_layer`。  
2. BFS/DFS 从入口出发，仅沿 **layer ≥ 1** 的边扩展，直到 Static 池字节达到 `static_cache_bytes` 或前沿为空。  
3. 对每个访问到的 `node_id` 调用 `cache_.pin(id, block)`。  
4. 记录 `static_pin_count` 写入启动日志；M4 下允许提高至占满 20GB。

### **阶段切换 θ（R5 规范，对齐 GoVector）**

- 维护候选队列长度 `ef`、已访问集合 `visited`、当前最优距离 `d_best`。  
- 定义 \(t^\*\)：首次某候选满足「进入 Top-k 邻域」的步数（实现可用距离阈值或排名）。  
- 离线标定比例 \(\theta \in (0,1)\)：使 \(\theta \cdot ef\_search \approx t^\*\) 的样本均值。  
- **在线**：当 `visited_count ≥ θ · ef_search` 时，将后续未命中节点优先写入 **Dynamic** 池，并提高 `prefetch_depth_layer0`（由策略 JSON 上限裁剪）。  
- θ 与 `memory_profile` 一并写入策略 `_meta`（M3/M4 默认 θ=0.35，M0 可关闭阶段切换）。

---

## **bvec 距离计算与查询格式**

| 场景 | base | query | 距离 |
| :---- | :---- | :---- | :---- |
| SIFT1B 标准 | bvecs uint8 | fvecs float | \( \sum_i (q_i - b_i)^2 \)，\(b_i\) 转 float 或 query 量化到 uint8 |
| 全 float 实验 | fvecs | fvecs | 现有 `l2_sq` |
| 子集评测 | bvecs 子集 | fvecs | 与 BIGANN 一致，保证 recall 口径可对照 |

**实现注意**：uint8 L2 可用 AVX2/AVX-512 批量；与磁盘块内 v2 布局对齐。M4 下建议在 `eval_disk` 日志中增加 `distance_kernel=u8_l2` 字段。

---

## **分片、多文件与磁盘规划**

### **多分片 bvec**

- 路径模式：`base_000.bvecs` … `base_127.bvecs` 或 manifest 列表。  
- 区分器：`data_bytes = sum(shard_size)`；`num_vectors_est = sum(shard_vectors)`。  
- `VectorDataset::open` 接受 manifest；`global_id = shard_id << 32 | local_id`（或外部 id 文件）。  
- **M4** 下建索引**顺序消费分片**，单分片峰值内存仍受 `batch_size` 约束。

### **推荐磁盘布局（单机 500G+ 场景）**

| 路径 | 内容 | 量级（示例） |
| :---- | :---- | :---- |
| `/data/vectors/base*.bvecs` | 原始向量 | ~500GB |
| `/data/index/graph.index` | NodeBlock 索引 | 与节点数×4KB 同级，另盘 |
| `/data/wal/` | WAL 段 | 递增，可轮转 |
| `/data/agent_io_policy.json` | 策略 + memory_profile | KB 级 |
| `/logs/eval_*.log` | 评测 | — |

内存 **100GB** 仅进程 RSS 上限；**不**要求 OS 对 500GB 文件 cache 全量。

---

## **Agent 策略 JSON 扩展字段（R11）**

在现有预取/缓存字段之外，增加与区分器对齐的元数据：

```json
{
  "agent_profile": "cursor",
  "prefetch_depth_upper": 2,
  "memory_profile": "BVEC_ULTRA_500G_100G",
  "partition_theta": 0.35,
  "pool_memtable_mb": 10240,
  "pool_static_mb": 20480,
  "pool_dynamic_mb": 71680,
  "vector_file_kind": "bvecs",
  "stream_index_batch_vectors": 500000,
  "_meta": {
    "trainer": "train_agent_policy.py",
    "data_source": "learn-bvecs",
    "aggregate_trace_stats": { }
  }
}
```

`train_agent_policy.py` 在 `--data-source learn-bvecs` 时：mmap 子采样（如 200 万条）做游走，**禁止** 全量 500GB 载入；`memory_profile` 由脚本根据输入文件大小与 `--ram-budget-gb` 自动填写。

---

## **命令行与对外 API 契约（R12）**

| 工具 | 新增/变更参数 | 说明 |
| :---- | :---- | :---- |
| `build_index` | `--format auto\|fvecs\|bvecs` | auto 走区分器 + 探测 |
| `build_index` | `--ram-budget-gb 100` | 流式 batch 与内存上限 |
| `build_index` | `--shard-manifest path` | 多分片列表 |
| `eval_disk` | 同上 + `--memory-profile override` | 强制 M0–M4 |
| `eval_disk` | `--base-limit` / `--query-limit` | 500G 场景先子集再全量 |
| `linux_run.sh` | `fetch-sift1b` / `eval-bvec` | 规划子命令 |

**环境变量**：`AMIO_RAM_BUDGET_GB`、`AMIO_MEMORY_PROFILE`、`AMIO_STREAM_BATCH_VECTORS`。

---

## **实施阶段（建议排期）**

| 阶段 | 周期目标 | 交付物 | 依赖 |
| :---- | :---- | :---- | :---- |
| **Phase 1** | 区分器 + VectorDataset bvec mmap | `memory_partition.cpp`、`load_bvec`、启动日志打印 M* | R1–R3 |
| **Phase 2** | NodeBlock v2 + 流式 `build_index` + u8 L2 | 子集 bvec 建索引与 recall | R6–R7 |
| **Phase 3** | 三区池化 + Static 提取 + θ 骨架 | M3/M4 下 recall@10 趋势达标 | R4–R5 |
| **Phase 4** | learn-bvecs 策略 + 500G 子集评测脚本 | `agent_io_policy`、linux_run 文档 | R11–R12 |
| **Phase 5**（研究） | Group Cache / PAIC / 图重排 / 写优化 | 论文级对照实验 | R8–R10 |

Phase 1–2 即可在 **100GB 机器**上对 **十亿级子集**（如 1e7–1e8 条）完成端到端验证；全量 500GB 构图耗时接受数天级离线任务。

---

## **异常、降级与可观测性**

| 风险 | 检测 | 降级策略 |
| :---- | :---- | :---- |
| 误判 fvecs/bvecs | 探测 dim 与文件长度不一致 | 降级 M2 + 告警退出码 |
| ram_budget 未配置 | 默认 `0.2 × data_bytes` 或物理内存 80% 较小者 | 日志 WARN |
| Static 提取超限 | pin 字节 > static_cache_bytes | 按 BFS 层截断，记录 `static_truncated=1` |
| mmap 失败（32 位 / 映射限制） | `open` 返回错误 | 回退 `pread` 流式 |
| dim>128 用 v1 索引 | header.version 不匹配 | 拒绝加载，提示需 v3 |
| OOM | RSS 超预算 | 减小 `stream_index_batch_vectors`、Dynamic 池 |

**可观测性字段**（写入 `eval_disk` 汇总与 CSV）：`memory_profile`、`rho`、`pool_*_mb`、`static_pins`、`theta_phase_switches`、`distance_kernel`、`bvec_mmap=1`。

---

## **与参考文献及赛题意见的映射（bvec / 区分器）**

| 设计能力 | 参考文献方向 | 赛题评审点 |
| :---- | :---- | :---- |
| M3/M4 磁盘优先 + 20% 内存 | GoVector、DiskANN 类 | 受限内存 recall |
| Static/Dynamic 划分 | GoVector 动静混合 | Base cache 分层 |
| 区分器自动配置 | PAIC/Baleen 参数化思想 | ML 调节缓存/预取 |
| bvec 流式 | LSM / 大规模向量工程 | 异步读、数据量分级 |
| io_uring + 顺序预取 | SARC、拓扑预取 | 图索引 + 顺序化读 |

---

## **工程实现状态与文档对照（仓库 agent-memory-io-cpp）**

> **重要**：上文各节为面向赛题与参考文献的**目标架构**；并非每一项均已在 C++ 仓库中完整实现。下表为当前代码落地摘要（2026-05），**完整对照表、代码路径与后续优先级**见 [`docs/Agent长时记忆-实现状态对照.md`](docs/Agent长时记忆-实现状态对照.md)，交付计划见 [`docs/赛题剩余工作与交付计划.md`](docs/赛题剩余工作与交付计划.md)。

### **实现状态总览**

| 设计模块（本文档章节） | 状态 | 仓库已实现要点 |
| :---- | :---- | :---- |
| 读优化（一）动静混合 + Group Cache + PAIC | **部分 / 未** | `GraphAwareCache`：pinned + hot LRU；`hot_insert_min_prior_misses` 近似二次准入；**无** θ、KV Group、ISVM |
| 读优化（二）图重排 + Episode + ML 预取门控 | **部分 / 未** | `TopologyPrefetcher` + Linux `io_uring`；离线 JSON 调分层预取/扇出/按 offset；**无** Gorder/Porder、Episode |
| 写优化 NVTable / RemapCom / NobLSM | **未（写骨架 only）** | `MemTable` + `WAL` + `CompactionWorker` 骨架 |
| Agent ML 访问特征 → 策略 | **离线/代理** | `learning/train_agent_policy.py` → `data/agent_io_policy.json` → `AgentIoPolicy::merge_json_file` |
| 透明集成 LangChain / LlamaIndex | **未** | C++ 库 + `eval_disk`；无 Python VectorStore 封装 |
| 性能评估 Recall≥85%、混合 QPS | **待 Linux 固化** | `scripts/linux_run.sh compare`（builtin vs learned） |
| **bvec 流式 I/O + 500G/100G 模式** | **未** | 仅 fvecs 全量加载；无 `MemoryPartitionDiscriminator` |
| **内存划分区分器 M0–M4** | **未** | `Config::cache_size_mb` 为单一池，未按类型/体积分区 |

### **当前已贯通的技术闭环（可答辩演示）**

1. **磁盘 HNSW 检索**：`index/hnsw.*`、`storage_layout.*`、`VectorStore::search_disk`  
2. **读路径优化（工程版）**：图感知缓存 + 拓扑预取 + io_uring（Linux）  
3. **Agent 策略 AB**：合成或 SIFT learn 特征 → JSON → `--policy-mode learned`  
4. **评测与日志**：`eval_disk` 汇总 QPS/延迟分位/Recall + 逐查询 CSV（缓存、磁盘读、预取块数）

### **与设计差距最大的未完成块（建议后续）**

| 优先级 | 未完成内容 | 对应设计 |
| :---- | :---- | :---- |
| P0 | Linux 实机 io_uring 与 learned 对比数据；评测口径文档 | § 性能评估 |
| P0 | 用真实检索轨迹或图邻接游走替代 learn 随机 id 池，修正 `seq_locality` | § Agent ML、GoVector |
| P1 | θ 阶段切换、Group Cache / PAIC 或预取 useful/wasted 统计 | § 读优化（一） |
| P1 | 建索引后图重排（可先 BFS/RCM） | § 读优化（二） |
| P2 | RemapCom、NobLSM、NVTable | § 写优化 |
| **P0** | **bvec** 流式读、mmap、L2(uint8)；`build_index` 分批 | § bvec / R1、R6、R7 |
| **P0** | **MemoryPartitionDiscriminator**（文件类型+大小+100GB 预算） | § 区分器 / R2、R3 |
| **P0** | 500GB bvec 评测口径与 `--ram-budget-gb` | § 500G/100G / R12 |

```bash
# 复现已实现部分（fvecs / SIFT 子集）
python learning/train_agent_policy.py --data-source auto --out data/agent_io_policy.json
./scripts/linux_run.sh compare   # Linux
```

---

## **结语**

面向未来通用人工智能应用，大语言模型Agent不仅需要算力巅峰的支持，更需要一个足够庞大、且能够瞬时存取与更新的长时记忆数据底座。当百亿级高维向量检索遭遇物理内存容量枷锁与存储介质硬件边界的双重制约时，修缝补漏式的优化已无法化解日益严重的性能矛盾。

本报告提出了一种彻底打破操作系统默认I/O性能墙的高性能向量检索存储引擎架构。该架构采取了彻底的自顶向下的重构策略：在用户态，利用精准划分阶段的多相缓存与针对随机无效预取的深度机器学习甄别替换逻辑，实现了对苛刻内存配额的压榨式利用；在I/O调度层，依托底层的图空间物理重排与非阻塞异步拉取网络，将延迟隐藏于计算之中；在持久化结构层，利用创新的无日志轻量化设计、底层介质零拷贝映射重定位以及无阻塞通信协议，全面剿灭了妨碍系统平滑运行的写入抖动与并发风暴。这一涵盖算法结构、软件系统与硬件介质全栈深层耦合的存储底座架构，必将为处理海量动态Agent记忆数据的未来系统确立一套具有压倒性优势的全新设计范式。

#### **Works cited**

1. GoVector An IO-Efficient Caching Strategy for High-Dimensional Vector Nearest Neighbor Search.pdf

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAA20lEQVR4XmNgGAUQIATEB4D4PBAXAzEjiiwUyADxKQaIpA4QPwRiSxQVQMAJxJsZIKaBgCQDRGErTAEMZADxXyD2gPJhChfCVQCBOBBfB+KrQCwCFTMG4q8MaAojgPg/EE9CEgtCFwOZADLpJxAvBeJZUBrEB5kIMhkMYFYcAGIeNLETQMwPFWNwAeJ/DKjW5jBArAV5EA5gCtORxEAmXQFiMSQxBn0g/gTEvkhiP4DYD4kPBiA3gEyogvLlgbiMAUfUmQHxfSBeDsTnGHAoggEOBkjAs6JLDBUAADuvKhiY8CLUAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFEAAAAYCAYAAACC2BGSAAACkElEQVR4Xu2Xz8tMURjHv0IRkR+9UiQKiVioV7aSlEg2yI8tC6UoSrYWNm8SKTtK/gBZWHmzEoXkx0oiJQsbpVj48f323PP23PvOmTn3zJixOJ/6NjPPvTP3O895znnOAQqFQqFQKPzfzKbWUvObF0bEYuoONUk9p85SM/wNI0Ceouyg3lNXqRfV67zaHcNlBfWEOgdL3CbqA7Xd3zQkZsL8XKT2Nq5NsZH6Qh2tPi+B/YErGM3Iz6XuwSowzIrlsCReqj4Pi53UR+ox9RBdkqiq+wTLdkBmFVvtYrlomdDDJ5BW3SepX9RuFwtJvOViubT1EziPSBIXUU+pd9QyF9cX/lAHXCyHBdQb6jTSDMvDW+o1tdTFt1Lf0V8S5UXLQxs/nmgS11CfUZ864gwsifpiDiup67CGoJFP5RDsuZodHg1mp3gK3st+tPPjiSYxjPAk6knUzTKdMvJaePfA1tHb1Pr65WRUearAn7CufLOS3ismn/LbC+8n10sn/kkSZ1GHqZfUZVhD6oeYlxDX4r7QxZsM2k+TaBI3UF8x3fgRWBIvuFgTTYtjGJxpdcLfmD5lT8G8qOF0Y9B+mkSTGKaQOp86YCA0FiWzF5o+2mf2O51DEk+4mCpPFfiKGnPxbng/uV46EU2i9oHXYM1FTSagalCFqlJT0W9tpu5Tj6htVSyVLdQ31I2qofyg9rlYKnp2rpdORJMotObIfHOzfQO2zuSgjqh9mLZPqd0wVF1YQlbBtjvh1JKL99JPd5av6JZPBo/DTi06n2r0HqDHOTER7cV0jEzdl43Djp93qWfUQfSXQI+eLx9t/ISmpqXNq7n8TaFK2EWtw+CMi7YnhDmwTXduxfSirZ9CoVAoJPIX0uGJuHIIaFQAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAA4ElEQVR4XmNgGAXYgSkQ7wHiR0B8AU0OA/AAsT4QXwfi02hyWIEmEL8F4jnoEthANBD/h9J4ASMQz2eAmAyyAS8QAeKrQHwYiHmBWAuI9zNA3G+ApA4M0hkgTggE4gIGiE0gt4PEIpDUMbAA8Rog/scAMckTKs4BxMIMEI1wAHPCcwZIGH8DYm5kBcjABoh/A3ErlO8ExA1wWTRQxABxgguUDwoNWFjnMCAFJcy9D4BYGipmDMTlDBD3LgViWag43L3LGSAaQYATiG8A8Q4gDoCKwQE/ELOiiYFMBaWXIQ0Ahv8kbTdnRaoAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAABF0lEQVR4Xu2RP0tCURjGHyEhSokaAqdQaBDqA7g31ODS0uAHcMkPIM5O0uI3cJAGR8VJwjFoaWh0KYomaY6gep7OOXh6vXKHaPMHP7j3PNxz3z/Amn+nQQc0Z4M0dukdvaU7cXDmD59on27FoadM5/TSBiqjSj9o12SBGn2mRRsI/f2LntvAo0uv6YYNRBuuLJVnydMbemoDobKncAPRYCy6cAgzqECJvmLRbxZuMKpGz9v00GdLhJIrtAP3p2MkrMUSSn6jYyx6ztBN/7ySUPI9ndEXegH3cSrxivRBk77DtZCKXZEu+aQntIfk6f+gKU7we0V1uP6PaMufJVKgj/QqOjugD3SElEmrxz24Xcbofd+crfkL3zHmLU8cDFbVAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAZCAYAAABjNDOYAAADHklEQVR4Xu2XTahNURTH/0IR8hkJSTEgRclAFMmAAUWSklIGDIgoykBKkgyUoUiSDEjJkLhlIgYMiIlCSimKIpKP9Xvr7Hv22fecd+/kHvLOr/7d7t5nn3fX2uvrSQ0NDQ0NDf8ge03XTGPTjcAkU8v02HTQNKywWw9LTXdMb0xPTAuK213ZY9oVfd9geiB/X8s0JdoLzDd9kJ8tZabpodwhC02vTcsKT9QDN7fI9Nz0yDSxuD0o4+W3Pydam2Baafpkum4aEe0FtpneqniuzWjTLblnYbrcOSfCAzUTbvJ8utGFFaYL6nTAGtMv04FkPXDWdFWd5wbYbfppWpt9D8651H6iXrjJ39lnrxDxZ5TbEMMl/5A7r4y7Kj+nafIQfqY8H5eYvujvOAcjL8ojhwjqFVLihjrTkDRtqWhfCllDSnawVX5LhFZgU8laXWAAhtw3jcvWKMr35DVocbaWgh1H00XlKXpF7ni0Wv6u2/JaO6/9dAQ5Rq7hiFOmzZluZmtx1U+Zrfz5XsWZbtCtvsovBkO2m46Ytpjemw7nj7ahZmI8jSQlXDS2jJSf32naJ/876/NHi4Rb+i5/+bnsk++kFelVRb+cgxEYs9G0X+4cnERxZp0IScEp/G6clIKTsWW56aQ8asbIxwXsrOzIoba0lA8/YY3ZoDQP+wiRTLulsxD266K9UabJKp+9SKcyp1F/eA9OIC3jlMRe2nwlocXFtYVBiBuig9VNiOR38gGQsKcDcdNVcIaCOiPdUF5vXppemD7Kh1vSqyvBOXFtIWKemqZGa2X0I61otbTcMF+RAtz6sfBACbRgHFgWUfFIwP4Oub1lUdYBk+hnFYvSN/nI3Y1+OIchjR/PpUE6DBLV8exDGjL0Vc0vnItHgngYxFmnVX12oKYQKXQD4McfUvkt1AH15pXyFAn1jw5DvaHozsr2gNmGfxeqaiP1pqW8nhIERBKfc+Xvq/xHM0CxYxjsKRf7CEamv4GLwjGpEXSmyxp8UKQgD0/W+E7JwOb/FlowaVP6/9BQ57h6q41DDkoA9YnPhoRVyptIQ938AT9Kpf31aK/VAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAZCAYAAACPQVaOAAAByklEQVR4Xu2WPyhFURzHf8IghCiJEpsoSRYpSSmDRQbFoAzKaFEGKRmwiA0lAwuZjcrGLBZFkQyYWBS+X+cejnP/9N7Lu+/S+dRnuN937n33d/7dI+JwOByO5FAJF+F6Ci7BHpj/eWf8FMF6O7TohHUS8o7D8A3uwRE4BG+9bNa7psvwGW6r22KlHA7Ccwn//2K4BcfhGjyD3WaDArgLx2CekT+IalxlZGQBTltZtjmGF/AAvkpwsXz3FXhiZKPwHjbroEHUiJbpwOMd7ovqDJN52G9lcdEu4TOrSdQAzRhZI7yDczrgFJ7UFx7sJRY7YeUsfAO2WnlcRBXL5cd3njKyGngtamaUMuA8LzQakApRD+XDTdgJ/C1w4cdAVLFcXix2wMiq4aWogll4IHzoqajC0oGd1ivfG1oqsr3d2WFEFcvMLrYEHknwwH3B6btphynw54rluuTGxDWQNH692Fp4JWp3SxpRxa6Kv1guQy7HG1GHDB9dor5l7JV0yeU05tJjseYZQO/GofsPG/OmTMhlsS3wSdSurNHfXo66D547DyXzYrNNB3yBO/LztEe413BTtU9Qj7DNyP4VPANwxvSJOkM4HA5H8vkAbjh1lbzUQWQAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAbCAYAAABBTc6+AAAAjElEQVR4XmNgGAWDFXADcSgQTwPiKUCsBpNgBGIfIH4CxJOA2AGIxYGYFabAD4h/A3EMTAAZ8APxCSC+AsQaQCwJxRwwBfpA/AmIbwDxLCRsClNgDMRfgbgcJoAOYCbgVMAJxJuBeBUDwtUgX8nCVQCBBBBvB+KDDBD7DwNxARCzICsCAQEGNP8PCQAAOl0Ty/tjjz8AAAAASUVORK5CYII=>