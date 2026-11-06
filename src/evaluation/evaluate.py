"""Span-level propaganda technique evaluation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from src.common.utils import read_json, read_jsonl
from src.evaluation.span_metrics import (
    SCORING_VERSION, compute_grr, is_unit_score, partial_match_prf, write_json,
)


def _overlap(s: dict[str, Any], t: dict[str, Any]) -> int:
    return max(0, min(int(s["end"]), int(t["end"])) - max(int(s["start"]), int(t["start"])))


def _span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["technique"])


def _normalize_doc_rows(
    rows: list[dict[str, Any]],
    role: str,
    warnings: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int | None]]:
    docs: dict[str, list[dict[str, Any]]] = {}
    text_lengths: dict[str, int | None] = {}
    duplicate_doc_ids = 0
    invalid_spans = 0
    duplicate_spans = 0

    for row in rows:
        doc_id = str(row["id"])
        if doc_id in docs:
            duplicate_doc_ids += 1
        text = row.get("text")
        text_len = len(text) if isinstance(text, str) else None
        if doc_id not in text_lengths or text_lengths[doc_id] is None:
            text_lengths[doc_id] = text_len

        seen_spans = {_span_key(span) for span in docs.get(doc_id, [])}
        for span_index, raw_span in enumerate(row.get("spans", []) or []):
            try:
                start = int(raw_span["start"])
                end = int(raw_span["end"])
                technique = str(raw_span["technique"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{role} doc={doc_id!r} span #{span_index} is malformed") from exc
            if start < 0 or start >= end or (text_len is not None and end > text_len):
                invalid_spans += 1
                continue
            span = {"start": start, "end": end, "technique": technique}
            key = _span_key(span)
            if key in seen_spans:
                duplicate_spans += 1
                continue
            docs.setdefault(doc_id, []).append(span)
            seen_spans.add(key)
        docs.setdefault(doc_id, [])

    if duplicate_doc_ids:
        warnings.append(f"{role}: {duplicate_doc_ids} duplicate document ids were merged")
    if invalid_spans:
        warnings.append(f"{role}: {invalid_spans} invalid spans were ignored")
    if duplicate_spans:
        warnings.append(f"{role}: {duplicate_spans} duplicate spans were ignored")
    return docs, text_lengths


def _align_prediction_docs(
    gold_docs: dict[str, list[dict[str, Any]]],
    pred_docs: dict[str, list[dict[str, Any]]],
    pred_text_lengths: dict[str, int | None],
    gold_text_lengths: dict[str, int | None],
    warnings: list[str],
) -> dict[str, list[dict[str, Any]]]:
    missing = sorted(set(gold_docs) - set(pred_docs))
    extra = sorted(set(pred_docs) - set(gold_docs))
    if missing:
        warnings.append(f"{len(missing)} gold documents missing from predictions; treated as empty, examples={missing[:5]}")
    if extra:
        warnings.append(f"{len(extra)} prediction documents are not in gold and were ignored, examples={extra[:5]}")

    gold_labels = {span["technique"] for spans in gold_docs.values() for span in spans}
    unknown_labels = sorted(
        {
            span["technique"]
            for doc_id, spans in pred_docs.items()
            if doc_id in gold_docs
            for span in spans
            if span["technique"] not in gold_labels
        }
    )
    if unknown_labels:
        warnings.append(f"{len(unknown_labels)} predicted labels are absent from gold labels, examples={unknown_labels[:5]}")

    aligned: dict[str, list[dict[str, Any]]] = {}
    out_of_bounds = 0
    for doc_id in gold_docs:
        text_len = gold_text_lengths.get(doc_id) or pred_text_lengths.get(doc_id)
        spans = []
        for span in pred_docs.get(doc_id, []):
            if text_len is not None and int(span["end"]) > text_len:
                out_of_bounds += 1
                continue
            spans.append(span)
        aligned[doc_id] = spans
    if out_of_bounds:
        warnings.append(f"pred: {out_of_bounds} spans exceeded gold document length and were ignored")
    return aligned


def _merge_location_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping spans after dropping technique labels for location-only SI."""
    intervals = sorted(
        ({"start": int(s["start"]), "end": int(s["end"]), "technique": "PROPAGANDA"} for s in spans),
        key=lambda s: (s["start"], s["end"]),
    )
    merged: list[dict[str, Any]] = []
    for span in intervals:
        if not merged or span["start"] >= merged[-1]["end"]:
            merged.append(dict(span))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
    return merged


def merge_location_docs(docs: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {doc_id: _merge_location_spans(spans) for doc_id, spans in docs.items()}


def _merge_same_technique_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping spans per technique before SemEval-style partial scoring."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        by_label.setdefault(str(span["technique"]), []).append(span)

    merged_all: list[dict[str, Any]] = []
    for technique, label_spans in by_label.items():
        merged: list[dict[str, Any]] = []
        for span in sorted(label_spans, key=lambda item: (int(item["start"]), int(item["end"]))):
            item = {"start": int(span["start"]), "end": int(span["end"]), "technique": technique}
            if not merged or item["start"] >= merged[-1]["end"]:
                merged.append(item)
            else:
                merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        merged_all.extend(merged)
    return sorted(merged_all, key=lambda item: (item["start"], item["end"], item["technique"]))


def merge_same_technique_docs(docs: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {doc_id: _merge_same_technique_spans(spans) for doc_id, spans in docs.items()}




def per_class_f1(
    gold_docs: dict[str, list[dict[str, Any]]],
    pred_docs: dict[str, list[dict[str, Any]]],
    labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    classes = list(labels) if labels is not None else sorted({span["technique"] for spans in gold_docs.values() for span in spans})
    out = {}
    for label in classes:
        gold_label = {
            doc_id: [span for span in spans if span["technique"] == label]
            for doc_id, spans in gold_docs.items()
        }
        pred_label = {
            doc_id: [span for span in spans if span["technique"] == label]
            for doc_id, spans in pred_docs.items()
        }
        out[label] = partial_match_prf(gold_label, pred_label, use_class=True)
    return out


def macro_f1(
    gold_docs: dict[str, list[dict[str, Any]]],
    pred_docs: dict[str, list[dict[str, Any]]],
    labels: list[str] | None = None,
) -> float:
    by_class = per_class_f1(gold_docs, pred_docs, labels=labels)
    return sum(value["f1"] for value in by_class.values()) / len(by_class) if by_class else 0.0


def _group_values(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and str(v) != ""]
    if isinstance(value, str) and value.lstrip().startswith("["):
        import json

        decoded = json.loads(value)
        if not isinstance(decoded, list):
            raise ValueError("Encoded group must be a JSON array")
        return _group_values(decoded)
    return [str(value)]


def grouped_metrics(gold_docs: dict[str, list[dict[str, Any]]], pred_docs: dict[str, list[dict[str, Any]]], groups: dict[str, Any]) -> dict[str, Any]:
    group_names = sorted(
        {
            name
            for doc_id, value in groups.items()
            if doc_id in gold_docs
            for name in _group_values(value)
        }
    )
    per_group = {}
    for group_name in group_names:
        ids = {doc_id for doc_id, value in groups.items() if group_name in _group_values(value)}
        group_gold = {doc_id: gold_docs[doc_id] for doc_id in ids if doc_id in gold_docs}
        group_pred = {doc_id: pred_docs.get(doc_id, []) for doc_id in group_gold}
        score = partial_match_prf(group_gold, group_pred, use_class=True)
        score["n_docs"] = len(group_gold)
        per_group[group_name] = score
    f1_values = [score["f1"] for score in per_group.values()]
    return {
        "per_group": per_group,
        "GMF1": sum(f1_values) / len(f1_values) if f1_values else None,
        "WorstF1": min(f1_values) if f1_values else None,
    }




def evaluate(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    groups: dict[str, Any] | None = None,
    id_eval_micro_f1: float | None = None,
    merge_same_label_overlaps: bool = False,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    gold_docs, gold_text_lengths = _normalize_doc_rows(gold, "gold", warnings)
    raw_pred_docs, pred_text_lengths = _normalize_doc_rows(pred, "pred", warnings)
    pred_docs = _align_prediction_docs(gold_docs, raw_pred_docs, pred_text_lengths, gold_text_lengths, warnings)

    if groups:
        grouped_gold_ids = {doc_id for doc_id, value in groups.items() if _group_values(value)}
        missing_group_ids = sorted(set(gold_docs) - grouped_gold_ids)
        extra_group_ids = sorted(grouped_gold_ids - set(gold_docs))
        if missing_group_ids:
            warnings.append(f"{len(missing_group_ids)} gold documents have no eval group, examples={missing_group_ids[:5]}")
        if extra_group_ids:
            warnings.append(f"{len(extra_group_ids)} group entries are not in gold and were ignored, examples={extra_group_ids[:5]}")

    if merge_same_label_overlaps:
        scored_gold_docs = merge_same_technique_docs(gold_docs)
        scored_pred_docs = merge_same_technique_docs(pred_docs)
    else:
        scored_gold_docs = gold_docs
        scored_pred_docs = pred_docs

    micro = partial_match_prf(scored_gold_docs, scored_pred_docs, use_class=True)
    location = partial_match_prf(
        merge_location_docs(gold_docs),
        merge_location_docs(pred_docs),
        use_class=False,
    )
    by_class = per_class_f1(scored_gold_docs, scored_pred_docs, labels=labels)
    macro = sum(value["f1"] for value in by_class.values()) / len(by_class) if by_class else 0.0

    report: dict[str, Any] = {
        "counts": {
            "n_docs": len(gold_docs),
            "n_gold_spans": micro["n_gold"],
            "n_pred_spans": micro["n_pred"],
            "n_raw_gold_spans": sum(len(spans) for spans in gold_docs.values()),
            "n_raw_pred_spans": sum(len(spans) for spans in pred_docs.values()),
        },
        "micro": micro,
        "location": location,
        "macro_f1": macro,
        "per_class": by_class,
        "scoring": {
            "version": SCORING_VERSION,
            "partial_match_mode": "per_span_union_coverage",
            "merge_same_label_overlaps": merge_same_label_overlaps,
            "macro_labels": labels or sorted({span["technique"] for spans in gold_docs.values() for span in spans}),
        },
        "warnings": warnings,
        "id_eval_micro_f1": id_eval_micro_f1 if is_unit_score(id_eval_micro_f1) else None,
    }

    per_group = {}
    if groups:
        grouped = grouped_metrics(scored_gold_docs, scored_pred_docs, groups)
        report["groups"] = grouped
        per_group = grouped["per_group"]
    grr = compute_grr(per_group, id_eval_micro_f1)
    report["GRR"] = grr
    if grr["warning"] and (groups is not None or id_eval_micro_f1 is not None):
        warnings.append(grr["warning"])
    return report


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return "NaN" if math.isnan(x) else f"{x:.4f}"
    return str(x)


def render_md(report: dict[str, Any], title: str) -> str:
    lines = [f"# Evaluation Report: {title}", ""]
    counts = report["counts"]
    lines.append(
        f"Documents: {counts['n_docs']}; gold spans: {counts['n_gold_spans']}; "
        f"predicted spans: {counts['n_pred_spans']}"
    )
    lines.append("")
    lines.append("| Metric | P | R | F1 |")
    lines.append("|---|---:|---:|---:|")
    micro = report["micro"]
    location = report["location"]
    lines.append(f"| Partial-match Micro | {_fmt(micro['precision'])} | {_fmt(micro['recall'])} | {_fmt(micro['f1'])} |")
    lines.append(f"| Location Only | {_fmt(location['precision'])} | {_fmt(location['recall'])} | {_fmt(location['f1'])} |")
    lines.append(f"| Macro-F1 | - | - | {_fmt(report['macro_f1'])} |")
    lines.append("")
    lines.append("## Per-Class F1")
    lines.append("| Class | P | R | F1 | gold | pred |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for class_name, value in sorted(report["per_class"].items(), key=lambda item: -item[1]["f1"]):
        lines.append(
            f"| {class_name} | {_fmt(value['precision'])} | {_fmt(value['recall'])} | {_fmt(value['f1'])} "
            f"| {value['n_gold']} | {value['n_pred']} |"
        )
    if "groups" in report:
        groups = report["groups"]
        grr_per = (report.get("GRR") or {}).get("per_group", {})
        lines += ["", "## Groups", "| Group | docs | MicroF1 | GRR |", "|---|---:|---:|---:|"]
        for group_name, value in groups["per_group"].items():
            lines.append(f"| {group_name} | {value['n_docs']} | {_fmt(value['f1'])} | {_fmt(grr_per.get(group_name))} |")
        lines.append("")
        lines.append(f"- **GMF1** = {_fmt(groups['GMF1'])}")
        lines.append(f"- **WorstF1** = {_fmt(groups['WorstF1'])}")
    if report.get("GRR"):
        lines.append(
            f"- **GRR** = {_fmt(report['GRR']['GRR'])} "
            f"(ID Eval MicroF1 = {_fmt(report.get('id_eval_micro_f1'))})"
        )
    if report["warnings"]:
        lines += ["", "## Warnings"] + [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate span-level propaganda technique predictions.")
    parser.add_argument("--gold", required=True, help="gold JSONL")
    parser.add_argument("--pred", required=True, help="prediction JSONL")
    parser.add_argument("--groups", default=None, help="optional eval_groups.json")
    parser.add_argument("--id_eval_report", default=None, help="ID Eval report JSON; its micro.f1 is used for GRR")
    parser.add_argument("--id_eval_micro_f1", type=float, default=None, help="ID Eval Micro-F1 used for GRR")
    parser.add_argument(
        "--merge_same_label_overlaps",
        action="store_true",
        help="Legacy/custom mode: merge same-technique overlapping gold and predicted spans before scoring.",
    )
    parser.add_argument("--label_file", default=None, help="Optional labels.json used as the Macro-F1 label universe.")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_md", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    pred = read_jsonl(args.pred)
    groups = read_json(args.groups) if args.groups else None
    label_file = Path(args.label_file) if args.label_file else Path(args.gold).parent / "labels.json"
    labels = None
    if label_file.exists():
        label_data = read_json(label_file)
        labels = list(label_data.get("labels", label_data) if isinstance(label_data, dict) else label_data)
    id_f1 = args.id_eval_micro_f1
    if id_f1 is None and args.id_eval_report:
        id_f1 = read_json(args.id_eval_report)["micro"]["f1"]

    report = evaluate(
        gold,
        pred,
        groups=groups,
        id_eval_micro_f1=id_f1,
        merge_same_label_overlaps=args.merge_same_label_overlaps,
        labels=labels,
    )
    title = args.title or Path(args.pred).stem

    print(
        f"Micro-F1={report['micro']['f1']:.4f}  "
        f"Macro-F1={report['macro_f1']:.4f}  "
        f"Loc-F1={report['location']['f1']:.4f}"
    )
    if "groups" in report:
        print(f"GMF1={_fmt(report['groups']['GMF1'])}  WorstF1={_fmt(report['groups']['WorstF1'])}")
    if report.get("GRR"):
        print(f"GRR={_fmt(report['GRR']['GRR'])}")
    for warning in report["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)

    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_md(report, title), encoding="utf-8")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
