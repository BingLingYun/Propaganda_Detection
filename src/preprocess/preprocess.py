"""模块一:数据预处理与质检。

用法:
    python -m src.preprocess.preprocess --config configs/examples/custom_raw_dataset.yaml
    python -m src.preprocess.preprocess --config ... --id_eval_ratio 0.1 --seed 42

流程:
    读取原始数据(配置驱动,无数据集特有硬编码)
    -> 偏移量归一化(统一为 字符级 / 0 起 / 左闭右开)
    -> (可选)同类别重叠片段合并
    -> dev 划分(若无官方 dev)
    -> ID Eval 按文档级抽取(固定种子,id 列表落盘)
    -> 目标域隔离校验 / 文档 id 跨划分校验
    -> 质检(偏移一致性、重叠统计、规模核对、分布统计、标签闭集)
    -> 写出标准化 JSONL + eval_groups.json + 质检报告(json + md)

重要约束:全程不对 text 做任何改变字符数的处理(不 strip、不做 Unicode 规范化),
所有输出偏移均相对原始文本。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from src.common.utils import (
    load_yaml, read_jsonl, write_jsonl, write_json,
    merge_same_technique_spans, sort_spans, spans_overlap, span_len, quantiles,
    get_labels,
)

MAX_FAIL_EXAMPLES = 50


# ============================================================ 原始数据读取

def _read_table(path: str, fmt: str) -> list[dict]:
    """读取一个文档/标注文件为字典列表。fmt: jsonl | json | csv | tsv"""
    if fmt == "jsonl":
        return read_jsonl(path)
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), f"{path}: json 格式要求顶层为数组"
        return data
    if fmt in ("csv", "tsv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t" if fmt == "tsv" else ",")
            return list(reader)
    raise ValueError(f"不支持的格式: {fmt}")


def _get(row: dict, field: str, path: str):
    if field not in row:
        raise KeyError(f"{path}: 记录缺少字段 '{field}',请检查配置 field_map。已有字段: {list(row)[:20]}")
    return row[field]


def _guess_span_file_format(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {"jsonl": "jsonl", "json": "json", "csv": "csv", "tsv": "tsv", "txt": "tsv"}.get(ext, "tsv")


def load_raw_split(cfg: dict, split: str, path: str, group: str | None = None) -> list[dict]:
    """读取一个划分,返回标准结构文档列表(偏移未归一化):
    [{id, text, spans:[{start,end,technique,_raw_text?}], _group?, _domain?}]
    """
    raw = cfg["raw"]
    fm = raw["field_map"]
    rows = _read_table(path, raw["docs_format"])

    docs = []
    for row in rows:
        doc = {
            "id": str(_get(row, fm["id"], path)),
            "text": _get(row, fm["text"], path),
            "spans": [],
        }
        if fm.get("group"):
            doc["_group"] = row.get(fm["group"])
        if group is not None:                       # 文件级分组优先
            doc["_group"] = group
        if fm.get("domain"):
            doc["_domain"] = row.get(fm["domain"])
        if raw["spans_source"] == "inline":
            for sp in (_get(row, fm["spans"], path) or []):
                doc["spans"].append(_convert_span_row(sp, fm, path))
        docs.append(doc)

    if raw["spans_source"] == "file":
        span_path = raw["spans_files"].get(split)
        if span_path:
            fmt = _guess_span_file_format(span_path)
            by_id = {d["id"]: d for d in docs}
            assert fm.get("span_doc_id"), "spans_source=file 需要配置 field_map.span_doc_id"
            for sp in _read_table(span_path, fmt):
                did = str(_get(sp, fm["span_doc_id"], span_path))
                if did not in by_id:
                    raise ValueError(f"{span_path}: 标注指向不存在的文档 id {did}")
                by_id[did]["spans"].append(_convert_span_row(sp, fm, span_path))
    return docs


def _convert_span_row(sp: dict, fm: dict, path: str) -> dict:
    out = {
        "start": int(_get(sp, fm["span_start"], path)),
        "end": int(_get(sp, fm["span_end"], path)),
        "technique": str(_get(sp, fm["span_technique"], path)),
    }
    if fm.get("span_text") and fm["span_text"] in sp:
        out["_raw_text"] = sp[fm["span_text"]]   # 仅用于偏移校验,落盘前删除
    return out


# ============================================================ 偏移量归一化

_WORD_RE = re.compile(r"\S+")


def normalize_offsets(doc: dict, conv: dict) -> None:
    """将偏移统一为 字符级 / 0 起 / 左闭右开。就地修改 doc['spans']。"""
    base = int(conv.get("base", 0))
    inclusive = bool(conv.get("end_inclusive", False))
    unit = conv.get("unit", "char")

    if unit == "char":
        for s in doc["spans"]:
            s["start"] -= base
            s["end"] -= base
            if inclusive:
                s["end"] += 1
    elif unit == "word":
        # 词级偏移:借助原文以 \S+ 分词重建字符偏移
        words = [(m.start(), m.end()) for m in _WORD_RE.finditer(doc["text"])]
        for s in doc["spans"]:
            ws = s["start"] - base
            we = s["end"] - base
            if inclusive:
                we += 1                      # 统一为词级左闭右开 [ws, we)
            if not (0 <= ws < we <= len(words)):
                raise ValueError(f"文档 {doc['id']} 词级偏移越界: [{s['start']}, {s['end']}], 共 {len(words)} 词")
            s["start"] = words[ws][0]
            s["end"] = words[we - 1][1]
    else:
        raise ValueError(f"不支持的偏移单位: {unit}")


# ============================================================ 质检

def qc_split(docs: list[dict], split: str, labels: set[str], cfg: dict) -> dict:
    """对单个划分执行全部质检项,返回统计字典(可 JSON 序列化)。"""
    report: dict = {"split": split, "n_docs": len(docs)}

    # ---- 1. 偏移一致性校验(逐条,不抽样)
    n_spans, n_fail = 0, 0
    fail_examples = []
    for d in docs:
        tlen = len(d["text"])
        for s in d["spans"]:
            n_spans += 1
            ok = 0 <= s["start"] < s["end"] <= tlen
            expected = s.get("_raw_text")
            actual = d["text"][s["start"]:s["end"]] if ok else None
            if ok and expected is not None:
                ok = (actual == expected)
            if not ok:
                n_fail += 1
                if len(fail_examples) < MAX_FAIL_EXAMPLES:
                    fail_examples.append({
                        "id": d["id"], "start": s["start"], "end": s["end"],
                        "expected_text": expected, "actual_slice": actual,
                    })
    report["n_spans"] = n_spans
    report["offset_check"] = {
        "n_fail": n_fail,
        "pass_rate": 1.0 if n_spans == 0 else 1 - n_fail / n_spans,
        "fail_examples": fail_examples,
    }

    # ---- 2. 重叠统计(文档内两两比较)
    n_overlap_pairs = n_cross_class_pairs = n_same_class_pairs = 0
    overlapping_span_ids: set[tuple] = set()
    n_fully_covered = 0
    max_nest_depth = 0
    for d in docs:
        sp = sort_spans(d["spans"])
        for i in range(len(sp)):
            covered = False
            for j in range(len(sp)):
                if i == j:
                    continue
                if spans_overlap(sp[i], sp[j]):
                    if j > i:
                        n_overlap_pairs += 1
                        if sp[i]["technique"] == sp[j]["technique"]:
                            n_same_class_pairs += 1
                        else:
                            n_cross_class_pairs += 1
                    overlapping_span_ids.add((d["id"], i))
                    # 完全覆盖:i 被 j 包含(含相等时取序靠后者视为被覆盖)
                    if (sp[j]["start"] <= sp[i]["start"] and sp[i]["end"] <= sp[j]["end"]
                            and (span_len(sp[j]) > span_len(sp[i]) or j < i)):
                        covered = True
            if covered:
                n_fully_covered += 1
        # 最大嵌套深度:扫描线统计同一字符位置上的最大片段层数
        events = sorted([(s["start"], 1) for s in sp] + [(s["end"], -1) for s in sp])
        depth = 0
        for _, delta in events:
            depth += delta
            max_nest_depth = max(max_nest_depth, depth)
    report["overlap"] = {
        "n_overlap_pairs": n_overlap_pairs,
        "same_class_pairs": n_same_class_pairs,
        "cross_class_pairs": n_cross_class_pairs,
        "same_class_ratio": n_same_class_pairs / n_overlap_pairs if n_overlap_pairs else 0.0,
        "cross_class_ratio": n_cross_class_pairs / n_overlap_pairs if n_overlap_pairs else 0.0,
        "overlapping_span_ratio": len(overlapping_span_ids) / n_spans if n_spans else 0.0,
        "fully_covered_span_ratio": n_fully_covered / n_spans if n_spans else 0.0,
        "max_nest_depth": max_nest_depth,
    }

    # ---- 3. 规模核对
    expected = (cfg.get("qc", {}).get("expected_counts") or {}).get(split) or {}
    warns = []
    for key, actual in (("docs", len(docs)), ("spans", n_spans)):
        exp = expected.get(key)
        if exp is not None and exp > 0 and abs(actual - exp) / exp > 0.01:
            warns.append(f"{split}.{key}: 实际 {actual} vs 预期 {exp},偏差超 ±1%")
    report["count_check"] = {"expected": expected, "warnings": warns}

    # ---- 4. 分布统计
    tech_counter = Counter(s["technique"] for d in docs for s in d["spans"])
    report["distributions"] = {
        "technique_counts": dict(tech_counter.most_common()),
        "text_len_quantiles": quantiles([len(d["text"]) for d in docs]),
        "spans_per_doc_quantiles": quantiles([len(d["spans"]) for d in docs]),
        "span_len_quantiles": quantiles([span_len(s) for d in docs for s in d["spans"]]),
    }

    # ---- 5. 标签闭集校验
    unknown = sorted(set(tech_counter) - labels)
    report["label_check"] = {"unknown_labels": unknown}
    return report


# ============================================================ 划分与隔离

def doc_key(doc_id: str, pattern: re.Pattern | None) -> str:
    if pattern is None:
        return doc_id
    m = pattern.search(doc_id)
    if not m:
        raise ValueError(f"doc_key_regex 无法匹配 id: {doc_id}")
    return m.group(1)


def split_by_doc(docs: list[dict], ratio: float, seed: int, key_re: re.Pattern | None
                 ) -> tuple[list[dict], list[dict], list[str]]:
    """按文档 key 随机抽取 ratio 比例,返回 (剩余, 抽出, 抽出的 key 列表)。"""
    keys = sorted({doc_key(d["id"], key_re) for d in docs})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_pick = max(1, round(len(keys) * ratio)) if ratio > 0 and keys else 0
    picked = set(keys[:n_pick])
    rest = [d for d in docs if doc_key(d["id"], key_re) not in picked]
    out = [d for d in docs if doc_key(d["id"], key_re) in picked]
    return rest, out, sorted(picked)


def check_isolation(splits: dict[str, list[dict]], cfg: dict, key_re: re.Pattern | None) -> list[str]:
    """目标域隔离 + 文档 id 跨划分校验。返回违规描述列表(空 = 通过)。"""
    violations = []
    # 同一文档 key 不得跨划分
    seen: dict[str, str] = {}
    for split, docs in splits.items():
        for d in docs:
            k = doc_key(d["id"], key_re)
            if k in seen and seen[k] != split:
                violations.append(f"文档 {k} 同时出现在 {seen[k]} 与 {split}")
            seen.setdefault(k, split)
    # Train/Dev 不得含目标域样本
    targets = set(cfg.get("target_domains") or [])
    if targets:
        for split in ("train", "dev"):
            for d in splits.get(split, []):
                if d.get("_domain") in targets:
                    violations.append(f"{split} 含目标域样本: id={d['id']} domain={d['_domain']}")
    return violations


# ============================================================ 报告输出

def render_md(dataset: str, reports: dict[str, dict], extra: dict) -> str:
    lines = [f"# 质检报告: {dataset}", ""]
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("| 划分 | 文档数 | 片段数 | 偏移通过率 | 重叠片段占比 | 最大嵌套深度 | 表外标签 |")
    lines.append("|---|---|---|---|---|---|---|")
    for split, r in reports.items():
        lines.append(
            f"| {split} | {r['n_docs']} | {r['n_spans']} "
            f"| {r['offset_check']['pass_rate']:.4f} "
            f"| {r['overlap']['overlapping_span_ratio']:.4f} "
            f"| {r['overlap']['max_nest_depth']} "
            f"| {', '.join(r['label_check']['unknown_labels']) or '-'} |")
    lines.append("")
    for split, r in reports.items():
        lines.append(f"## {split} 技巧类别分布")
        lines.append("| 类别 | 片段数 |")
        lines.append("|---|---|")
        for t, c in r["distributions"]["technique_counts"].items():
            lines.append(f"| {t} | {c} |")
        for w in r["count_check"]["warnings"]:
            lines.append(f"\n> ⚠️ {w}")
        lines.append("")
    return "\n".join(lines)


# ============================================================ 主流程

def run(cfg: dict, args: argparse.Namespace) -> int:
    name = cfg["name"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    id_eval_ratio = args.id_eval_ratio if args.id_eval_ratio is not None else cfg.get("id_eval_ratio", 0.1)
    out_dir = Path(cfg["processed_dir"])
    report_dir = Path("outputs") / name / "preprocess_reports"
    key_re = re.compile(cfg["doc_key_regex"]) if cfg.get("doc_key_regex") else None
    labels = set(get_labels(cfg))
    conv = cfg["raw"]["offset_convention"]

    # ---- 读取原始划分
    files = cfg["raw"]["files"]
    splits: dict[str, list[dict]] = {}
    eval_docs: list[dict] = []
    ev = files["eval"]
    if isinstance(ev, list):
        for item in ev:
            eval_docs += load_raw_split(cfg, "eval", item["path"], group=item.get("group"))
    else:
        eval_docs = load_raw_split(cfg, "eval", ev)
    splits["eval"] = eval_docs
    splits["train"] = load_raw_split(cfg, "train", files["train"])
    if files.get("dev"):
        splits["dev"] = load_raw_split(cfg, "dev", files["dev"])

    # ---- 偏移量归一化
    for docs in splits.values():
        for d in docs:
            normalize_offsets(d, conv)

    # ---- 同类别重叠合并(仅在配置开启时;ZenPropaganda 场景)
    n_merged_total = 0
    if cfg.get("merge_same_label_overlaps"):
        for docs in splits.values():
            for d in docs:
                if d["spans"]:
                    # _raw_text 在合并后不再与新区间对应,合并前先完成文字校验由 QC 负责,
                    # 此处直接丢弃 _raw_text 以免误报
                    merged, n = merge_same_technique_spans(
                        [{k: v for k, v in s.items() if k != "_raw_text"} for s in d["spans"]])
                    if n:
                        d["spans"] = merged
                        n_merged_total += n

    # ---- dev 划分(无官方 dev 时)
    if "dev" not in splits:
        dev_ratio = cfg.get("dev_ratio", 0.1)
        splits["train"], splits["dev"], _ = split_by_doc(splits["train"], dev_ratio, seed + 1, key_re)
        print(f"[{name}] 无官方 dev,从 train 按 {dev_ratio} 划分: dev={len(splits['dev'])} 文档")

    # ---- ID Eval 抽取(从 Train+Dev 池,文档级,固定种子,id 列表落盘)
    pool = splits["train"] + splits["dev"]
    _, id_eval_docs, picked_keys = split_by_doc(pool, id_eval_ratio, seed, key_re)
    picked_set = set(picked_keys)
    splits["train"] = [d for d in splits["train"] if doc_key(d["id"], key_re) not in picked_set]
    splits["dev"] = [d for d in splits["dev"] if doc_key(d["id"], key_re) not in picked_set]
    splits["id_eval"] = id_eval_docs
    write_json(out_dir / "id_eval_ids.json",
               {"seed": seed, "ratio": id_eval_ratio, "doc_keys": picked_keys,
                "ids": sorted(d["id"] for d in id_eval_docs)})

    # ---- 隔离校验
    violations = check_isolation(splits, cfg, key_re)
    if violations:
        print(f"[{name}] 隔离校验失败,违规列表:", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        write_json(report_dir / f"{name}_isolation_violations.json", violations)
        return 2

    # ---- 质检
    reports = {split: qc_split(docs, split, labels, cfg) for split, docs in splits.items()}
    unknown = sorted({u for r in reports.values() for u in r["label_check"]["unknown_labels"]})
    total_spans = sum(r["n_spans"] for r in reports.values())
    total_fail = sum(r["offset_check"]["n_fail"] for r in reports.values())
    fail_rate = total_fail / total_spans if total_spans else 0.0
    threshold = cfg.get("qc", {}).get("offset_fail_threshold", 0.01)

    extra = {
        "总片段数": total_spans,
        "偏移校验总失败率": f"{fail_rate:.4%} (阈值 {threshold:.2%})",
        "同类别重叠合并数(merge_same_label_overlaps)": n_merged_total,
        "ID Eval 抽取": f"ratio={id_eval_ratio}, seed={seed}, 文档数={len(id_eval_docs)}",
    }
    write_json(report_dir / f"{name}_qc.json",
               {"dataset": name, "summary": {k: str(v) for k, v in extra.items()},
                "same_label_merge_count": n_merged_total, "splits": reports})
    (report_dir / f"{name}_qc.md").write_text(render_md(name, reports, extra), encoding="utf-8")

    # ---- 写出标准化 JSONL(落盘前删除内部字段)
    def clean(d: dict) -> dict:
        return {"id": d["id"], "text": d["text"],
                "spans": sort_spans([{k: v for k, v in s.items() if not k.startswith("_")}
                                     for s in d["spans"]])}

    for split, fname in (("train", "train.jsonl"), ("dev", "dev.jsonl"),
                         ("eval", "eval.jsonl"), ("id_eval", "id_eval.jsonl")):
        write_jsonl(out_dir / fname, (clean(d) for d in splits[split]))

    # ---- eval_groups.json(存在分组信息时)
    groups = {d["id"]: d["_group"] for d in splits["eval"] if d.get("_group")}
    if groups:
        write_json(out_dir / "eval_groups.json", groups)

    for split, docs in splits.items():
        print(f"[{name}] {split}: {len(docs)} 文档, {reports[split]['n_spans']} 片段")

    # ---- 退出码
    if unknown:
        print(f"[{name}] 错误: 出现表外标签: {unknown}", file=sys.stderr)
        return 3
    if fail_rate > threshold:
        print(f"[{name}] 错误: 偏移校验失败率 {fail_rate:.4%} 超过阈值 {threshold:.2%},"
              f"失败样例见 {report_dir / f'{name}_qc.json'}", file=sys.stderr)
        return 1
    print(f"[{name}] 预处理完成 -> {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="数据预处理与质检(模块一)")
    ap.add_argument("--config", required=True, help="数据集配置 yaml 路径")
    ap.add_argument("--seed", type=int, default=None, help="覆盖配置中的 seed")
    ap.add_argument("--id_eval_ratio", type=float, default=None, help="覆盖配置中的 ID Eval 抽取比例")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    sys.exit(run(cfg, args))


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
