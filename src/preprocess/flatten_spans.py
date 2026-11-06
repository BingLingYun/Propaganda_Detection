"""训练侧标签压平工具:将重叠片段压平为互不重叠的片段集,供 --single_label 模式使用。

仅作用于 Train / Dev,输出 train_flat.jsonl / dev_flat.jsonl。
Eval 与 ID Eval 的 gold 永远保持原始重叠标注,本脚本拒绝处理它们。

策略:
    longest     - 冲突时保留较长片段
    rarest      - 冲突时保留训练集中较稀有类别的片段
    drop_nested - 丢弃被完全覆盖的内层片段;剩余的部分重叠再按 longest 兜底,
                  保证输出严格无重叠

用法:
    python -m src.preprocess.flatten_spans --processed_dir data/processed/custom --strategy longest
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.common.utils import read_jsonl, write_jsonl, sort_spans, spans_overlap, span_len


def _resolve_greedy(spans: list[dict], priority_key) -> list[dict]:
    """按优先级贪心保留片段,丢弃与已保留片段相交者。priority_key 越小越优先。"""
    kept: list[dict] = []
    for s in sorted(spans, key=priority_key):
        if not any(spans_overlap(s, k) for k in kept):
            kept.append(s)
    return sort_spans(kept)


def flatten_doc(spans: list[dict], strategy: str, tech_freq: Counter) -> list[dict]:
    if len(spans) <= 1:
        return sort_spans(list(spans))
    if strategy == "longest":
        # 长者优先;同长按 start 稳定
        return _resolve_greedy(spans, lambda s: (-span_len(s), s["start"]))
    if strategy == "rarest":
        # 稀有类别优先(训练集频次小者优先)
        return _resolve_greedy(spans, lambda s: (tech_freq[s["technique"]], -span_len(s), s["start"]))
    if strategy == "drop_nested":
        # 先丢弃被完全覆盖的内层片段
        kept = []
        for i, s in enumerate(spans):
            covered = any(
                j != i and o["start"] <= s["start"] and s["end"] <= o["end"]
                and (span_len(o) > span_len(s) or j < i)
                for j, o in enumerate(spans))
            if not covered:
                kept.append(s)
        # 剩余部分重叠按 longest 兜底,保证严格无重叠
        return _resolve_greedy(kept, lambda s: (-span_len(s), s["start"]))
    raise ValueError(f"未知策略: {strategy}")


def main():
    ap = argparse.ArgumentParser(description="训练侧重叠片段压平(仅 Train/Dev)")
    ap.add_argument("--processed_dir", required=True, help="预处理输出目录(含 train.jsonl/dev.jsonl)")
    ap.add_argument("--strategy", default="longest", choices=["longest", "rarest", "drop_nested"])
    args = ap.parse_args()

    pdir = Path(args.processed_dir)
    train = read_jsonl(pdir / "train.jsonl")
    tech_freq = Counter(s["technique"] for d in train for s in d["spans"])

    for split in ("train", "dev"):
        src = pdir / f"{split}.jsonl"
        if not src.exists():
            continue
        docs = read_jsonl(src)
        n_before = sum(len(d["spans"]) for d in docs)
        for d in docs:
            d["spans"] = flatten_doc(d["spans"], args.strategy, tech_freq)
            # 输出自检:严格无重叠
            for i in range(len(d["spans"]) - 1):
                assert d["spans"][i]["end"] <= d["spans"][i + 1]["start"] or \
                    not spans_overlap(d["spans"][i], d["spans"][i + 1]), \
                    f"压平后仍有重叠: doc {d['id']}"
        n_after = sum(len(d["spans"]) for d in docs)
        out = pdir / f"{split}_flat.jsonl"
        write_jsonl(out, docs)
        print(f"{split}: {n_before} -> {n_after} 片段 (策略 {args.strategy}) -> {out}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
