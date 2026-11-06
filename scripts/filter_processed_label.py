"""Filter a label from already processed JSONL splits and refresh metadata."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import load_yaml, write_json


SPLITS = ["train", "dev", "id_eval", "eval"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove labels from processed split JSONL files.")
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--drop_label", action="append", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.dataset_config)
    processed_dir = Path(cfg["processed_dir"])
    drop_labels = set(args.drop_label)

    label_counts: Counter = Counter()
    summary = {
        "name": cfg["name"],
        "processed_dir": str(processed_dir),
        "filtered_labels": sorted(drop_labels),
        "splits": {},
    }
    total_removed = 0

    for split in SPLITS:
        path = processed_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        removed = 0
        kept_spans = 0
        for row in rows:
            spans = row.get("spans", [])
            new_spans = [span for span in spans if span.get("technique") not in drop_labels]
            removed += len(spans) - len(new_spans)
            kept_spans += len(new_spans)
            row["spans"] = new_spans
            label_counts.update(span["technique"] for span in new_spans)
        total_removed += removed
        summary["splits"][split] = {"documents": len(rows), "spans": kept_spans, "removed_spans": removed}
        if not args.dry_run:
            write_jsonl(path, rows)

    labels = sorted(label_counts)
    if not args.dry_run:
        write_json(processed_dir / "labels.json", {"labels": labels, "label_counts": dict(label_counts.most_common())})
        metadata_path = processed_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        for split, stats in summary["splits"].items():
            metadata.setdefault(split, {})
            metadata[split]["documents"] = stats["documents"]
            metadata[split]["spans"] = stats["spans"]
        metadata["labels"] = labels
        metadata["n_labels"] = len(labels)
        metadata["filtered_labels"] = sorted(drop_labels)
        metadata["filtered_label_removed_spans"] = total_removed
        write_json(metadata_path, metadata)

    print(f"processed_dir={processed_dir}")
    for split, stats in summary["splits"].items():
        print(f"{split}: docs={stats['documents']} spans={stats['spans']} removed={stats['removed_spans']}")
    print(f"labels={len(labels)} removed_total={total_removed}")
    if args.dry_run:
        print("dry_run=True: no files were modified")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
