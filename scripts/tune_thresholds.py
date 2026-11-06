"""Tune dev thresholds for two-stage baseline checkpoints."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.inference.checkpoint import load_classifier, load_proposal
from src.common.device import detect_device
from src.common.experiment import update_checkpoint_threshold
from src.common.utils import read_json, read_jsonl, set_seed, write_json
from src.evaluation.evaluate import merge_location_docs, partial_match_prf
from src.inference.span_predict import predict_proposal_docs
from src.inference.two_stage_predict import classify_proposals
from src.training.stage2_data import read_proposal_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a single global threshold on dev data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gold", required=True, help="Dev JSONL with gold spans.")
    parser.add_argument("--proposal_file", default=None, help="Required for ts_classifier tuning.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_docs", type=int, default=None, help="Smoke-test only: tune on first N docs.")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--stop", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--target_recall", type=float, default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_md", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    return parser.parse_args()


def load_resolved_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint.parent / "resolved_config.json"
    return read_json(path) if path.exists() else {}


def grid_values(args: argparse.Namespace, resolved: dict[str, Any]) -> list[float]:
    grid = resolved.get("threshold_grid") or {}
    start = float(args.start if args.start is not None else grid.get("start", 0.10))
    stop = float(args.stop if args.stop is not None else grid.get("stop", 0.90))
    step = float(args.step if args.step is not None else grid.get("step", 0.05))
    values = []
    cur = start
    while cur <= stop + 1e-9:
        values.append(round(cur, 6))
        cur += step
    return values


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return 0.0 if denom == 0 else (1 + beta2) * precision * recall / denom


def proposal_gold_docs(docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for doc in docs:
        seen = set()
        spans = []
        for span in doc.get("spans", []):
            key = (int(span["start"]), int(span["end"]))
            if key in seen:
                continue
            seen.add(key)
            spans.append({"start": key[0], "end": key[1], "technique": "PROPAGANDA"})
        out[str(doc["id"])] = spans
    return out


def proposals_as_span_docs(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(row["id"]): [
            {"start": int(p["start"]), "end": int(p["end"]), "technique": "PROPAGANDA"}
            for p in row.get("proposals", [])
        ]
        for row in rows
    }


def tune_proposal(checkpoint: Path, docs: list[dict[str, Any]], thresholds: list[float], batch_size: int, device, target_recall: float):
    model, tokenizer, meta = load_proposal(checkpoint, device)
    gold_docs = proposal_gold_docs(docs)
    records = []
    best = None
    recalled = []
    for threshold in thresholds:
        rows = predict_proposal_docs(
            model,
            tokenizer,
            docs,
            int(meta["max_length"]),
            int(meta["stride"]),
            int(meta.get("max_span_width", 128)),
            threshold,
            batch_size,
            device,
            candidate_chunk_size=int(meta.get("candidate_chunk_size", 32768)),
            max_proposals_per_document=int(meta.get("max_proposals_per_document", 200)),
        )
        metrics = partial_match_prf(
            merge_location_docs(gold_docs),
            merge_location_docs(proposals_as_span_docs(rows)),
            use_class=False,
        )
        f2 = fbeta(metrics["precision"], metrics["recall"], beta=2.0)
        avg_props = sum(len(row.get("proposals", [])) for row in rows) / max(1, len(rows))
        record = {"threshold": threshold, **metrics, "f2": f2, "avg_proposals_per_doc": avg_props}
        records.append(record)
        if metrics["recall"] >= target_recall:
            recalled.append((threshold, record, rows))
        key = (f2, metrics["recall"], metrics["precision"], threshold)
        if best is None or key > best[0]:
            best = (key, record, rows)
    if recalled:
        chosen = max(recalled, key=lambda item: (item[0], item[1]["precision"]))
        return chosen[1], records, chosen[2], "target_recall_then_highest_threshold"
    return best[1], records, best[2], "dev_proposal_f2"


def tune_classifier(checkpoint: Path, docs: list[dict[str, Any]], proposal_file: str, thresholds: list[float], batch_size: int, device):
    model, tokenizer, meta = load_classifier(checkpoint, device)
    proposals_by_doc = read_proposal_file(proposal_file)
    gold_docs = {str(doc["id"]): doc.get("spans", []) for doc in docs}
    records = []
    best = None
    for threshold in thresholds:
        pred = classify_proposals(
            model,
            tokenizer,
            docs,
            proposals_by_doc,
            meta["labels"],
            int(meta["max_length"]),
            threshold,
            batch_size,
            device,
            nms_iou=float(meta.get("nms_iou", 0.5)),
            sentence_context_sentences=int(meta.get("sentence_context_sentences", 0)),
            candidate_chunk_size=int(meta.get("candidate_chunk_size", 32768)),
        )
        metrics = partial_match_prf(
            gold_docs,
            {str(row["id"]): row["spans"] for row in pred},
            use_class=True,
        )
        record = {"threshold": threshold, **metrics}
        records.append(record)
        key = (metrics["f1"], metrics["precision"], threshold)
        if best is None or key > best[0]:
            best = (key, record, pred)
    return best[1], records, best[2], "dev_partial_micro_f1"


def render_threshold_md(report: dict[str, Any]) -> str:
    lines = [f"# Threshold Tuning: {report['run_name']}", ""]
    lines.append(f"- model_type: `{report['model_type']}`")
    lines.append(f"- selected_threshold: `{report['selected_threshold']}`")
    lines.append(f"- selection_metric: `{report['selection_metric']}`")
    lines.append("")
    lines.append("| threshold | P | R | F1 | extra |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report["grid_results"]:
        extra = row.get("f2", "")
        if isinstance(extra, float) and not math.isnan(extra):
            extra = f"F2={extra:.4f}"
        lines.append(
            f"| {row['threshold']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {extra} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    meta = read_json(checkpoint / "meta.json")
    resolved = load_resolved_config(checkpoint)
    docs = read_jsonl(args.gold)
    if args.max_docs:
        docs = docs[: args.max_docs]
    thresholds = grid_values(args, resolved)
    device = detect_device(args.device)
    model_type = meta.get("model_type")
    if model_type == "ts_proposal":
        target = float(args.target_recall if args.target_recall is not None else resolved.get("target_dev_recall", 0.95))
        selected, records, best_pred, selection_metric = tune_proposal(checkpoint, docs, thresholds, args.batch_size, device, target)
    elif model_type == "ts_classifier":
        if not args.proposal_file:
            raise ValueError("--proposal_file is required for ts_classifier threshold tuning")
        selected, records, best_pred, selection_metric = tune_classifier(
            checkpoint, docs, args.proposal_file, thresholds, args.batch_size, device
        )
    else:
        raise ValueError(f"Threshold tuning is only needed for two-stage checkpoints, got {model_type}")

    run_name = meta["run_name"]
    threshold = float(selected["threshold"])
    report = {
        "run_name": run_name,
        "model_type": model_type,
        "selected_threshold": threshold,
        "selection_metric": selection_metric,
        "dev_docs": len(docs),
        "selected_metrics": selected,
        "grid_results": records,
    }
    update_checkpoint_threshold(checkpoint, threshold, report)
    out_json = Path(args.out_json) if args.out_json else checkpoint / "threshold_tuning.json"
    out_md = Path(args.out_md) if args.out_md else checkpoint / "threshold_tuning.md"
    write_json(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_threshold_md(report), encoding="utf-8")
    print(f"selected_threshold={threshold:.4f} metric={selection_metric} -> {checkpoint / 'thresholds.json'}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
