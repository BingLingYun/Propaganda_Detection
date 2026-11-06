"""Dependency-free span scoring, mirrored in the standalone platform evaluator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCORING_VERSION = "span-coverage-v2"


def _coverage_by_label(
    spans: list[dict[str, Any]], use_class: bool,
) -> dict[str | None, list[tuple[int, int]]]:
    """Build interval unions for coverage only; keep original span counts."""
    by_label: dict[str | None, list[tuple[int, int]]] = {}
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        if start < end:
            label = str(span["technique"]) if use_class else None
            by_label.setdefault(label, []).append((start, end))
    coverage = {}
    for label, intervals in by_label.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        coverage[label] = merged
    return coverage


def _covered_length(span: dict[str, Any], intervals: list[tuple[int, int]]) -> int:
    """Count each character in this span at most once."""
    start, end = int(span["start"]), int(span["end"])
    covered = 0
    for left, right in intervals:
        if left >= end:
            break
        if right > start:
            covered += min(end, right) - max(start, left)
    return covered


def partial_match_prf(
    gold_docs: dict[str, list[dict[str, Any]]],
    pred_docs: dict[str, list[dict[str, Any]]],
    use_class: bool = True,
) -> dict[str, float]:
    """Average each span's coverage by the opposite same-label interval union.

    This is a coverage metric, not one-to-one matching: several fragments can
    jointly cover a gold span, and redundant correct fragments are not penalized
    beyond their own uncovered characters. Original spans and denominators stay
    intact; only overlapping character contributions are deduplicated.
    """
    doc_ids = sorted(set(gold_docs) | set(pred_docs))
    n_pred = sum(len(pred_docs.get(doc_id, [])) for doc_id in doc_ids)
    n_gold = sum(len(gold_docs.get(doc_id, [])) for doc_id in doc_ids)
    p_credits, r_credits = [], []
    for doc_id in doc_ids:
        gold_spans = gold_docs.get(doc_id, [])
        pred_spans = pred_docs.get(doc_id, [])
        gold_coverage = _coverage_by_label(gold_spans, use_class)
        pred_coverage = _coverage_by_label(pred_spans, use_class)
        for spans, coverage, credits in (
            (pred_spans, gold_coverage, p_credits),
            (gold_spans, pred_coverage, r_credits),
        ):
            for span in spans:
                length = int(span["end"]) - int(span["start"])
                if length > 0:
                    label = str(span["technique"]) if use_class else None
                    credits.append(_covered_length(span, coverage.get(label, [])) / length)
    precision = math.fsum(p_credits) / n_pred if n_pred else 0.0
    recall = math.fsum(r_credits) / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_pred": n_pred, "n_gold": n_gold}


def is_unit_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
        and math.isfinite(value)
    )


def compute_grr(
    per_group: dict[str, dict[str, float]], id_micro_f1: float | None,
) -> dict[str, Any]:
    """Undefined ratios use None plus a reason; never fabricate zero or NaN."""
    def unavailable(status: str, warning: str) -> dict[str, Any]:
        return {
            "GRR": None,
            "per_group": {name: None for name in per_group},
            "status": status,
            "warning": warning,
        }

    if id_micro_f1 is None:
        return unavailable("missing_id_eval", "ID Eval result was not provided; GRR was not computed")
    if not is_unit_score(id_micro_f1):
        return unavailable("invalid_id_score", "ID Eval Micro-F1 must be finite and in [0, 1]; GRR was not computed")
    if id_micro_f1 == 0:
        return unavailable("zero_id_score", "ID Eval Micro-F1 is 0; GRR is undefined")
    if not per_group:
        return unavailable("missing_groups", "No target groups were provided; GRR was not computed")
    if any(not is_unit_score(score.get("f1")) for score in per_group.values()):
        return unavailable("invalid_group_score", "Target-group Micro-F1 must be finite and in [0, 1]; GRR was not computed")
    per = {name: score["f1"] / id_micro_f1 for name, score in per_group.items()}
    if not all(math.isfinite(value) for value in per.values()):
        return unavailable("non_finite_ratio", "GRR division overflowed; no finite ratio could be reported")
    # Divide before summing so a finite mean cannot overflow its accumulator.
    mean = math.fsum(value / len(per) for value in per.values())
    return {"GRR": mean, "per_group": per, "status": "ok", "warning": None}


def write_json(path: str | Path, payload: Any) -> None:
    """Serialize strictly before opening a file, rejecting NaN and Infinity."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")
