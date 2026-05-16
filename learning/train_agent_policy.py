#!/usr/bin/env python3
"""
离线学习 Agent 向量检索 I/O 访问模式，导出 C++ 运行时策略 JSON。

两种数据来源（--data-source）：
  - synthetic：按 Agent 画像生成合成轨迹（原行为）。
  - learn-fvecs：读取官方 learn 集（如 data/sift/sift_learn.fvecs），在向量上做多步
    「随机候选池 + 近邻贪心游走」，统计 mean_fanout / seq_locality 等，再映射到策略 JSON。

建模典型 Agent（synthetic 模式，与赛题意见一致）：
  - cursor / qwen-ai / qwen3：见各 _synthetic_*_trace。

用法:
  python learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json
  python learning/train_agent_policy.py --data-source learn-fvecs --learn-fvecs data/sift/sift_learn.fvecs
  python learning/train_agent_policy.py --profile qwen-ai --out data/agent_io_policy_qwen.json
  python learning/train_agent_policy.py --dump-features
  python learning/train_agent_policy.py --dump-features --dump-profiles cursor,qwen3
  python learning/train_agent_policy.py --dump-features --dump-traces 200 --dump-samples 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Tuple

FEATURE_NAMES = ["mean_fanout", "std_fanout", "seq_locality", "high_layer_share"]

FEATURE_DESCRIPTIONS_ZH = {
    "mean_fanout": "每步扩展邻居数的均值，反映图遍历扇出规模",
    "std_fanout": "邻居数标准差，反映扇出波动（并行/多工具时更大）",
    "seq_locality": "顺序局部性得分 [0,1]，邻居在 id 上是否成簇、利于顺序读盘",
    "high_layer_share": "高层（layer>0）步数占比，反映上层导航 vs 底层精搜比重",
}


@dataclass
class TraceStats:
    mean_fanout: float
    std_fanout: float
    seq_locality: float
    high_layer_share: float


def _synthetic_cursor_trace(steps: int, rng: random.Random) -> TraceStats:
    fanouts: List[int] = []
    prev = rng.randint(0, 10_000)
    seq_hits = 0
    high_hits = 0
    for _ in range(steps):
        layer = 0 if rng.random() < 0.72 else rng.randint(1, 3)
        if layer > 0:
            high_hits += 1
        f = int(rng.gauss(6, 1.2))
        f = max(2, min(24, f))
        fanouts.append(f)
        nbrs = [prev + rng.randint(-3, 3) for _ in range(f)]
        prev = rng.choice(nbrs)
        if any(abs(a - b) <= 2 for a in nbrs for b in nbrs if a != b):
            seq_hits += 1
    mean_f = sum(fanouts) / len(fanouts)
    var = sum((x - mean_f) ** 2 for x in fanouts) / len(fanouts)
    return TraceStats(
        mean_fanout=mean_f,
        std_fanout=math.sqrt(var),
        seq_locality=seq_hits / steps,
        high_layer_share=high_hits / steps,
    )


def _synthetic_qwen_trace(steps: int, rng: random.Random) -> TraceStats:
    fanouts: List[int] = []
    prev = rng.randint(0, 50_000)
    seq_hits = 0
    high_hits = 0
    for _ in range(steps):
        layer = 0 if rng.random() < 0.45 else rng.randint(1, 6)
        if layer > 0:
            high_hits += 1
        f = int(rng.gauss(18, 4.0))
        f = max(4, min(32, f))
        fanouts.append(f)
        nbrs = [rng.randint(0, 100_000) for _ in range(f)]
        prev = rng.choice(nbrs)
        if abs(prev - rng.choice(nbrs)) < 500:
            seq_hits += 1
    mean_f = sum(fanouts) / len(fanouts)
    var = sum((x - mean_f) ** 2 for x in fanouts) / len(fanouts)
    return TraceStats(
        mean_fanout=mean_f,
        std_fanout=math.sqrt(var),
        seq_locality=seq_hits / steps,
        high_layer_share=high_hits / steps,
    )


def _synthetic_qwen3_trace(steps: int, rng: random.Random) -> TraceStats:
    """Qwen3 类：推理段内 id 局部连贯 + 工具调用时随机跳转，扇出中等偏高。"""
    fanouts: List[int] = []
    prev = rng.randint(0, 30_000)
    seq_hits = 0
    high_hits = 0
    for _ in range(steps):
        layer = 0 if rng.random() < 0.52 else rng.randint(1, 5)
        if layer > 0:
            high_hits += 1
        f = int(rng.gauss(12.0, 2.2))
        f = max(4, min(28, f))
        fanouts.append(f)
        tool_burst = rng.random() < 0.28
        nbrs: List[int] = []
        if tool_burst:
            nbrs = [rng.randint(0, 80_000) for _ in range(f)]
        else:
            for _i in range(f):
                if rng.random() < 0.68:
                    nbrs.append(prev + rng.randint(-8, 8))
                else:
                    nbrs.append(rng.randint(0, 60_000))
        prev = rng.choice(nbrs)
        if len(nbrs) >= 2:
            a, b = rng.choice(nbrs), rng.choice(nbrs)
            if abs(a - b) < 400:
                seq_hits += 1
    mean_f = sum(fanouts) / len(fanouts)
    var = sum((x - mean_f) ** 2 for x in fanouts) / len(fanouts)
    return TraceStats(
        mean_fanout=mean_f,
        std_fanout=math.sqrt(var),
        seq_locality=seq_hits / steps,
        high_layer_share=high_hits / steps,
    )


def _stats_to_features(s: TraceStats) -> List[float]:
    return [s.mean_fanout, s.std_fanout, s.seq_locality, s.high_layer_share]


def load_fvecs(path: str, max_vectors: int | None = None) -> List[List[float]]:
    """读取 TexMex fvecs：每行 uint32 dim + dim 个 float32（小端）。"""
    out: List[List[float]] = []
    with open(path, "rb") as fp:
        while True:
            hdr = fp.read(4)
            if len(hdr) < 4:
                break
            (dim,) = struct.unpack("<I", hdr)
            payload = fp.read(dim * 4)
            if len(payload) < dim * 4:
                break
            vec = list(struct.unpack(f"<{dim}f", payload))
            out.append(vec)
            if max_vectors is not None and len(out) >= max_vectors:
                break
    return out


def _l2_sq(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        d = a[i] - b[i]
        s += d * d
    return s


def _learn_random_walk_trace(
    vecs: List[List[float]],
    steps: int,
    rng: random.Random,
    pool_size: int = 256,
) -> TraceStats:
    """
    在 learn 向量集上模拟「每步从随机候选池中取近邻」的访问轨迹，
    用 id 空间邻近度近似顺序读盘局部性，用距离分布近似高层/底层步。
    """
    n = len(vecs)
    if n < 50:
        raise ValueError("learn 向量过少")
    fanouts: List[int] = []
    seq_hits = 0
    high_hits = 0
    curr = rng.randrange(n)
    pool_sz = min(pool_size, n)
    id_span = max(500, n // 200)

    for _ in range(steps):
        q = vecs[curr]
        pool = {curr}
        while len(pool) < pool_sz:
            pool.add(rng.randrange(n))
        pool_list = list(pool)
        scored = [(j, _l2_sq(q, vecs[j])) for j in pool_list]
        scored.sort(key=lambda x: x[1])
        nf = int(rng.gauss(14.0, 4.0))
        nf = max(4, min(32, nf))
        top = scored[:nf]
        nbr_ids = [j for j, _ in top]
        fanouts.append(len(nbr_ids))
        if len(nbr_ids) >= 2:
            span = max(nbr_ids) - min(nbr_ids)
            if span < id_span:
                seq_hits += 1
        dists = [d for _, d in scored[: min(32, len(scored))]]
        if len(dists) >= 4 and dists[0] > 1e-12:
            med = dists[len(dists) // 2]
            if med / dists[0] > 4.0:
                high_hits += 1
        curr = rng.choice(nbr_ids)

    mean_f = sum(fanouts) / len(fanouts)
    var = sum((x - mean_f) ** 2 for x in fanouts) / len(fanouts)
    return TraceStats(
        mean_fanout=mean_f,
        std_fanout=math.sqrt(var),
        seq_locality=seq_hits / steps,
        high_layer_share=high_hits / steps,
    )


def _average_trace_stats(traces: List[TraceStats]) -> TraceStats:
    if not traces:
        raise ValueError("empty traces")
    k = len(traces)
    return TraceStats(
        mean_fanout=sum(t.mean_fanout for t in traces) / k,
        std_fanout=sum(t.std_fanout for t in traces) / k,
        seq_locality=sum(t.seq_locality for t in traces) / k,
        high_layer_share=sum(t.high_layer_share for t in traces) / k,
    )


def _cluster_from_learn_aggregate(s: TraceStats) -> int:
    """
    由 learn 集聚合特征选择策略簇（与 _policy_for_cluster 对齐）：
    扇出较低、顺序局部性较高 → 偏 cluster0（cursor 类 I/O）；否则偏 cluster1。
    """
    score = s.mean_fanout / 32.0 - s.seq_locality * 0.55 + s.high_layer_share * 0.35
    return 0 if score < 0.42 else 1


_PROFILE_GENERATORS: Dict[str, Tuple[str, Callable[[int, random.Random], TraceStats]]] = {
    "cursor": ("Cursor（IDE 小步、窄扇出）", _synthetic_cursor_trace),
    "qwen-ai": ("Qwen / 长对话（宽扇出、高随机）", _synthetic_qwen_trace),
    "qwen3": ("Qwen3（推理链局部 + 工具突发，合成模型）", _synthetic_qwen3_trace),
}


def dump_extracted_features(
    n_traces: int, seed: int, sample_rows: int, profile_keys: List[str]
) -> None:
    """打印合成轨迹上提取的 4 维特征：名称、含义、各 Agent 模型的分布与若干样本。"""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng = random.Random(seed)
    print("=== AI-agent 访问模式：提取特征说明 ===\n")
    print(
        "说明：以下为**按行为假设生成的合成轨迹**上统计的四维特征，"
        "用于对比不同 Agent 画像下的 I/O/图遍历形态；并非 Cursor/Qwen 官方遥测。\n"
    )
    for name in FEATURE_NAMES:
        print(f"  - {name}")
        print(f"    {FEATURE_DESCRIPTIONS_ZH[name]}\n")

    def summarize(
        label: str, gen: Callable[[int, random.Random], TraceStats], n: int
    ) -> tuple[List[TraceStats], List[List[float]]]:
        traces = [gen(200, rng) for _ in range(n)]
        X = [_stats_to_features(t) for t in traces]
        return traces, X

    for key in profile_keys:
        if key not in _PROFILE_GENERATORS:
            raise SystemExit(f"未知 profile: {key!r}，可选: {list(_PROFILE_GENERATORS)}")
        label, gen = _PROFILE_GENERATORS[key]
        _, X = summarize(label, gen, n_traces)
        print(f"=== 分布统计: {label}  [profile={key}] (n={n_traces}, steps=200/trace, seed={seed}) ===")
        for j, fname in enumerate(FEATURE_NAMES):
            col = [row[j] for row in X]
            mean_c = sum(col) / len(col)
            var_c = sum((x - mean_c) ** 2 for x in col) / len(col)
            std_c = math.sqrt(var_c)
            print(
                f"  {fname}: mean={mean_c:.4f}  std={std_c:.4f}  "
                f"min={min(col):.4f}  max={max(col):.4f}"
            )
        print("\n  样本特征向量 [mean_fanout, std_fanout, seq_locality, high_layer_share]:")
        for i, row in enumerate(X[:sample_rows]):
            print(f"    #{i + 1}: {[round(x, 4) for x in row]}")
        print()


def _policy_for_cluster(cluster: int, profile_name: str) -> Dict[str, Any]:
    """策略簇 → JSON 字段（与 C++ merge_json_file 对齐）。"""
    if cluster == 0:
        return {
            "agent_profile": profile_name,
            "prefetch_depth_upper": 1,
            "prefetch_depth_layer0": 2,
            "max_neighbor_fanout_layer0": 24,
            "use_layer_aware_prefetch": True,
            "sort_prefetch_by_disk_offset": True,
            "hot_insert_min_prior_misses": 2,
        }
    return {
        "agent_profile": profile_name,
        "prefetch_depth_upper": 2,
        "prefetch_depth_layer0": 2,
        "max_neighbor_fanout_layer0": 32,
        "use_layer_aware_prefetch": False,
        "sort_prefetch_by_disk_offset": False,
        "hot_insert_min_prior_misses": 1,
    }


def _maybe_print_tree(X: List[List[float]], y: List[int]) -> str:
    try:
        from sklearn import tree  # type: ignore

        clf = tree.DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        imp = clf.feature_importances_.tolist()
        return f"sklearn_tree feature_importances={imp}"
    except Exception as e:
        return f"sklearn 不可用，跳过决策树拟合 ({e})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        choices=("cursor", "qwen-ai", "qwen3"),
        default="cursor",
        help="目标 Agent 画像（写入 JSON 的 agent_profile；learn 模式下仍用于命名）",
    )
    ap.add_argument("--out", default="data/agent_io_policy.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--data-source",
        choices=("auto", "synthetic", "learn-fvecs"),
        default="auto",
        help="auto：若存在 --learn-fvecs 则读 learn 集，否则合成轨迹",
    )
    ap.add_argument(
        "--learn-fvecs",
        default="data/sift/sift_learn.fvecs",
        help="SIFT learn 集路径（fvecs）",
    )
    ap.add_argument(
        "--learn-max-vectors",
        type=int,
        default=80000,
        metavar="N",
        help="最多载入的 learn 向量条数（控内存/时间）",
    )
    ap.add_argument(
        "--learn-traces",
        type=int,
        default=120,
        metavar="T",
        help="learn 模式下随机游走的轨迹条数",
    )
    ap.add_argument(
        "--trace-steps",
        type=int,
        default=200,
        metavar="S",
        help="每条轨迹的步数",
    )
    ap.add_argument(
        "--learn-pool-size",
        type=int,
        default=256,
        metavar="P",
        help="learn 模式下每步随机候选池大小",
    )
    ap.add_argument(
        "--dump-features",
        action="store_true",
        help="仅打印提取的特征说明与统计，不写策略文件",
    )
    ap.add_argument(
        "--dump-traces",
        type=int,
        default=80,
        metavar="N",
        help="与 --dump-features 配合：每类 Agent 合成轨迹条数",
    )
    ap.add_argument(
        "--dump-samples",
        type=int,
        default=8,
        metavar="K",
        help="与 --dump-features 配合：每类打印前 K 条特征向量",
    )
    ap.add_argument(
        "--dump-profiles",
        type=str,
        default="cursor,qwen3",
        help="与 --dump-features 配合：逗号分隔 profile 键，如 cursor,qwen3",
    )
    args = ap.parse_args()

    if args.dump_features:
        keys = [k.strip() for k in args.dump_profiles.split(",") if k.strip()]
        dump_extracted_features(args.dump_traces, args.seed, args.dump_samples, keys)
        return

    rng = random.Random(args.seed)
    data_source = args.data_source
    if data_source == "auto":
        data_source = "learn-fvecs" if os.path.isfile(args.learn_fvecs) else "synthetic"

    meta: Dict[str, Any] = {
        "trainer": "train_agent_policy.py",
        "data_source": data_source,
        "papers_mapping": "GoVector/SARC/PAIC/Baleen/一次性访问排除 → 分层预取+顺序化+二次准入",
    }

    if data_source == "learn-fvecs":
        if not os.path.isfile(args.learn_fvecs):
            raise SystemExit(f"--data-source=learn-fvecs 但未找到文件: {args.learn_fvecs}")
        vecs = load_fvecs(args.learn_fvecs, args.learn_max_vectors)
        if len(vecs) < 100:
            raise SystemExit(f"learn 向量过少: {len(vecs)}，请检查 {args.learn_fvecs}")
        d0 = len(vecs[0])
        if any(len(v) != d0 for v in vecs[: min(2000, len(vecs))]):
            raise SystemExit("learn fvecs 行维度不一致")
        traces = [
            _learn_random_walk_trace(vecs, args.trace_steps, rng, args.learn_pool_size)
            for _ in range(args.learn_traces)
        ]
        s_agg = _average_trace_stats(traces)
        cluster = _cluster_from_learn_aggregate(s_agg)
        med = sorted(t.mean_fanout for t in traces)[len(traces) // 2]
        labels = [0 if t.mean_fanout < med else 1 for t in traces]
        X = [_stats_to_features(t) for t in traces]
        tree_info = _maybe_print_tree(X, labels)
        policy = _policy_for_cluster(cluster, args.profile)
        meta["tree_probe"] = tree_info
        meta["learn_fvecs"] = args.learn_fvecs
        meta["learn_vectors_loaded"] = len(vecs)
        meta["learn_traces"] = args.learn_traces
        meta["trace_steps"] = args.trace_steps
        meta["learn_pool_size"] = args.learn_pool_size
        meta["aggregate_trace_stats"] = {k: round(v, 6) for k, v in asdict(s_agg).items()}
        meta["policy_cluster"] = cluster
    else:
        traces: List[TraceStats] = []
        labels: List[int] = []
        for _ in range(40):
            traces.append(_synthetic_cursor_trace(200, rng))
            labels.append(0)
        for _ in range(40):
            traces.append(_synthetic_qwen_trace(200, rng))
            labels.append(1)
        for _ in range(40):
            traces.append(_synthetic_qwen3_trace(200, rng))
            labels.append(2)

        X = [_stats_to_features(t) for t in traces]
        tree_info = _maybe_print_tree(X, labels)
        cluster = 0 if args.profile == "cursor" else 1
        policy = _policy_for_cluster(cluster, args.profile)
        meta["tree_probe"] = tree_info
        meta["policy_cluster"] = cluster

    policy["_meta"] = meta

    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(policy, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    print(f"written: {args.out}")
    print(meta.get("tree_probe", ""))


if __name__ == "__main__":
    main()
