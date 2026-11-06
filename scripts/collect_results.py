"""Collect evaluation report JSON files into CSV and Markdown summaries.

Expected report naming:
    outputs/<dataset>/<run_name>/reports/{split}.json

Usage:
    python scripts/collect_results.py [--reports_dir outputs] [--out_prefix summary_results]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLUMNS = [
    "run",
    "split",
    "n_docs",
    "n_gold_spans",
    "n_pred_spans",
    "micro_p",
    "micro_r",
    "micro_f1",
    "macro_f1",
    "loc_f1",
    "GMF1",
    "WorstF1",
    "GRR",
    "id_eval_micro_f1",
]


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return "NaN" if math.isnan(x) else f"{x:.4f}"
    return str(x)


def load_row(path: Path) -> dict | None:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if "micro" not in report:
        return None
    stem = path.stem
    if "__" in stem:
        run, _, split = stem.rpartition("__")
    else:
        run, split = path.parent.parent.name, stem
    groups = report.get("groups") or {}
    grr = (report.get("GRR") or {}).get("GRR")
    return {
        "run": run,
        "split": split,
        "n_docs": report["counts"]["n_docs"],
        "n_gold_spans": report["counts"]["n_gold_spans"],
        "n_pred_spans": report["counts"]["n_pred_spans"],
        "micro_p": report["micro"]["precision"],
        "micro_r": report["micro"]["recall"],
        "micro_f1": report["micro"]["f1"],
        "macro_f1": report["macro_f1"],
        "loc_f1": report["location"]["f1"],
        "GMF1": groups.get("GMF1"),
        "WorstF1": groups.get("WorstF1"),
        "GRR": grr,
        "id_eval_micro_f1": report.get("id_eval_micro_f1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", default="outputs")
    parser.add_argument("--out_prefix", default="summary_results")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    rows = []
    for path in sorted(reports_dir.rglob("*.json")):
        if path.name in {"summary_results.json", "thresholds.json", "threshold_tuning.json", "runtime.json", "meta.json"}:
            continue
        try:
            row = load_row(path)
        except (json.JSONDecodeError, KeyError):
            continue
        if row:
            rows.append(row)

    if not rows:
        print(f"{reports_dir} 下未发现评测报告，命名应为 reports/{{split}}.json")
        return
    rows.sort(key=lambda row: (row["run"], row["split"]))

    csv_path = reports_dir / f"{args.out_prefix}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row[k]) for k in COLUMNS})

    md_path = reports_dir / f"{args.out_prefix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 实验记录总表\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "---|" * len(COLUMNS) + "\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row[k]) or "-" for k in COLUMNS) + " |\n")

    print(f"{len(rows)} 行 -> {csv_path} / {md_path}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
