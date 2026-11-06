"""Trivial baselines for propaganda span prediction."""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.utils import load_yaml, read_jsonl, sort_spans, span_len, write_json, write_jsonl


SENT_BOUNDARY = re.compile(r"[.!?;:\n\r\u3002\uff01\uff1f]+")


def train_stats(train_docs: list[dict[str, Any]]):
    spans_per_doc = [len(doc.get("spans", [])) for doc in train_docs]
    span_lens = [span_len(span) for doc in train_docs for span in doc.get("spans", [])]
    tech_counter = Counter(span["technique"] for doc in train_docs for span in doc.get("spans", []))
    techs, weights = (zip(*tech_counter.items()) if tech_counter else ((), ()))
    return spans_per_doc or [0], span_lens or [1], list(techs), list(weights), tech_counter


def predict_empty(docs, *_):
    return [{"id": doc["id"], "spans": []} for doc in docs]


def predict_random(docs, stats, rng: random.Random):
    spans_per_doc, span_lens, techs, weights, _ = stats
    preds = []
    for doc in docs:
        text_len = len(doc["text"])
        spans = []
        k = rng.choice(spans_per_doc)
        for _ in range(k):
            if text_len < 1 or not techs:
                break
            length = min(rng.choice(span_lens), text_len)
            length = max(1, length)
            start = rng.randint(0, text_len - length)
            spans.append(
                {
                    "start": start,
                    "end": start + length,
                    "technique": rng.choices(techs, weights=weights, k=1)[0],
                }
            )
        preds.append({"id": doc["id"], "spans": sort_spans(spans)})
    return preds


def sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Return non-empty sentence-like character spans without changing offsets."""
    bounds = []
    prev = 0
    for match in SENT_BOUNDARY.finditer(text):
        bounds.append((prev, match.start()))
        prev = match.end()
    bounds.append((prev, len(text)))

    clean = []
    for start, end in bounds:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            clean.append((start, end))
    return clean


def predict_majority_sentence(docs, stats, *_):
    _, _, _, _, tech_counter = stats
    if not tech_counter:
        return predict_empty(docs)
    majority = tech_counter.most_common(1)[0][0]
    preds = []
    for doc in docs:
        spans = [
            {"start": start, "end": end, "technique": majority}
            for start, end in sentence_bounds(doc["text"])
        ]
        preds.append({"id": doc["id"], "spans": spans})
    return preds


def write_majority_stats(path: Path, stats, dataset: str, run_name: str) -> None:
    _, _, _, _, tech_counter = stats
    majority = tech_counter.most_common(1)[0][0] if tech_counter else None
    write_json(
        path,
        {
            "run_name": run_name,
            "dataset": dataset,
            "method": "majority",
            "prediction_unit": "sentence",
            "majority_label": majority,
            "label_counts": dict(tech_counter.most_common()),
        },
    )


METHODS = {
    "empty": predict_empty,
    "random": predict_random,
    "majority": predict_majority_sentence,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trivial baselines.")
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--split", nargs="+", default=["eval", "id_eval"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", default="outputs")
    parser.add_argument("--run_name", default=None, help="Optional explicit output run name. Only valid when one method is selected.")
    args = parser.parse_args()

    if args.run_name and len(args.methods) != 1:
        raise ValueError("--run_name can only be used with exactly one trivial method.")

    cfg = load_yaml(args.dataset_config)
    processed_dir = Path(cfg["processed_dir"])
    stats = train_stats(read_jsonl(processed_dir / "train.jsonl"))

    for split in args.split:
        docs = read_jsonl(processed_dir / f"{split}.jsonl")
        for method in args.methods:
            rng = random.Random(args.seed)
            preds = METHODS[method](docs, stats, rng)
            dataset_root = Path(args.out_root)
            if dataset_root.name != cfg["name"]:
                dataset_root = dataset_root / cfg["name"]
            run_name = args.run_name or method
            if method == "majority":
                write_majority_stats(dataset_root / run_name / "model" / "stats.json", stats, cfg["name"], run_name)
            out = dataset_root / run_name / "predictions" / f"{split}.jsonl"
            write_jsonl(out, preds)
            n_spans = sum(len(row["spans"]) for row in preds)
            print(f"{method:18s} {split}: {len(preds)} documents, {n_spans} spans -> {out}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
