#!/usr/bin/env python3
"""
离线学习 Agent 向量检索 I/O 访问模式，导出 C++ 运行时策略 JSON。

建模典型 Agent（与赛题意见一致，均为**合成统计轨迹**，非厂商遥测）：
  - cursor：IDE 内短上下文、多轮小步编辑 → 图遍历更「窄」、顺序局部性更强。
  - qwen-ai：长对话、宽扇出、跳转更随机。
  - qwen3：在「推理链局部连贯」与「工具/检索突发跳转」之间折中，扇出介于 cursor 与 qwen-ai。

特征（由合成轨迹统计）：
  mean_fanout, std_fanout, seq_locality_score, high_layer_share

算法：
  在合成数据上拟合 sklearn.tree.DecisionTreeClassifier（若已安装），
  打印特征重要性；最终导出策略仍按 --profile 显式映射到可解释参数，
  保证可复现、与答辩材料一致。

用法:
  python learning/train_agent_policy.py --profile cursor --out data/agent_io_policy.json
  python learning/train_agent_policy.py --profile qwen-ai --out data/agent_io_policy_qwen.json
  python learning/train_agent_policy.py --profile qwen3 --out data/agent_io_policy_qwen3.json
  python learning/train_agent_policy.py --dump-features
  python learning/train_agent_policy.py --dump-features --dump-profiles cursor,qwen3
  python learning/train_agent_policy.py --dump-features --dump-traces 200 --dump-samples 5
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
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
        help="目标 Agent 画像（导出 JSON 时用；qwen3 与 qwen-ai 共用宽扇出策略簇）",
    )
    ap.add_argument("--out", default="data/agent_io_policy.json")
    ap.add_argument("--seed", type=int, default=42)
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
    policy["_meta"] = {
        "trainer": "train_agent_policy.py",
        "tree_probe": tree_info,
        "papers_mapping": "GoVector/SARC/PAIC/Baleen/一次性访问排除 → 分层预取+顺序化+二次准入",
    }

    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(policy, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    print(f"written: {args.out}")
    print(tree_info)


if __name__ == "__main__":
    main()
