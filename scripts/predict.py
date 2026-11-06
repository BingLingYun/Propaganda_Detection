"""Unified neural baseline prediction entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.device import detect_device
from src.common.utils import read_jsonl, set_seed, write_jsonl
from src.inference.checkpoint import load_model
from src.modeling.inference import predict_docs


MODEL_TYPES = ["mhbio_crf", "ts_proposal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict propaganda spans with a neural baseline checkpoint.")
    parser.add_argument("--model_type", default=None, choices=MODEL_TYPES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_docs", type=int, default=None, help="Smoke-test only: predict first N documents.")
    parser.add_argument("--out_root", default="outputs", help="Root output directory; default writes to outputs/<dataset>/<run>/predictions.")
    parser.add_argument("--out_dir", default=None, help="Direct prediction output directory; overrides --out_root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    device = detect_device(args.device)
    model, tokenizer, meta = load_model(checkpoint, device)
    model_type = meta.get("model_type", "mhbio_crf")
    if args.model_type and args.model_type != model_type:
        raise ValueError(f"Requested model_type={args.model_type}, checkpoint has {model_type}")
    docs = read_jsonl(args.input)
    if args.max_docs:
        docs = docs[: args.max_docs]
    split = args.split or Path(args.input).stem

    if model_type == "mhbio_crf":
        predictions = predict_docs(
            model,
            tokenizer,
            docs,
            meta["labels"],
            int(meta["max_length"]),
            int(meta["stride"]),
            int(meta.get("merge_gap", 0)),
            args.batch_size,
            device,
            single_label=False,
        )
    else:
        raise ValueError(
            "ts_proposal is a diagnostic Stage 1 checkpoint. "
            "Use it to build proposals for Stage 2; it is not an end-to-end technique predictor."
        )

    if args.out_dir:
        out_path = Path(args.out_dir) / f"{split}.jsonl"
    else:
        dataset = meta.get("dataset")
        if dataset:
            out_path = Path(args.out_root) / dataset / meta["run_name"] / "predictions" / f"{split}.jsonl"
        else:
            out_path = Path(args.out_root) / meta["run_name"] / f"{split}.jsonl"
    write_jsonl(out_path, predictions)
    n_spans = sum(len(doc["spans"]) for doc in predictions)
    print(f"{len(predictions)} documents, {n_spans} spans -> {out_path}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
