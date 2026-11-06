"""Export processed JSONL splits into train/test CSV folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import load_yaml, read_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed dataset splits to CSV.")
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--out_dir", default=None, help="Default: <processed_dir>_csv")
    return parser.parse_args()


def format_group(value: str | list[str] | None) -> str:
    if isinstance(value, list):
        return value[0] if len(value) == 1 else json.dumps(value, ensure_ascii=False)
    return str(value or "")


def write_split_csv(split_path: Path, out_path: Path, groups: dict[str, str | list[str]]) -> tuple[int, int]:
    docs = read_jsonl(split_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "spans", "n_spans", "group"])
        writer.writeheader()
        for doc in docs:
            spans = doc.get("spans", [])
            writer.writerow(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "spans": json.dumps(spans, ensure_ascii=False),
                    "n_spans": len(spans),
                    "group": format_group(groups.get(str(doc["id"]))),
                }
            )
    return len(docs), sum(len(doc.get("spans", [])) for doc in docs)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.dataset_config)
    processed_dir = Path(cfg["processed_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / "csv" / str(cfg["name"])

    split_groups_path = processed_dir / "split_groups.json"
    split_groups = read_json(split_groups_path) if split_groups_path.exists() else {}
    if not split_groups:
        groups_path = processed_dir / "eval_groups.json"
        split_groups = {"eval": read_json(groups_path)} if groups_path.exists() else {}

    layout = {
        "train": ["train", "dev"],
        "test": ["id_eval", "eval"],
    }
    for folder, splits in layout.items():
        for split in splits:
            n_docs, n_spans = write_split_csv(
                processed_dir / f"{split}.jsonl",
                out_dir / folder / f"{split}.csv",
                split_groups.get(split, {}),
            )
            print(f"{split}: {n_docs} docs, {n_spans} spans -> {out_dir / folder / f'{split}.csv'}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
