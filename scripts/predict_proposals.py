"""Generate Stage 1 proposals from a ts_proposal checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.device import detect_device
from src.common.experiment import checkpoint_threshold
from src.common.utils import read_jsonl, set_seed, write_jsonl
from src.inference.checkpoint import load_proposal
from src.inference.span_predict import predict_proposal_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict span proposals for two-stage Stage 2.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_docs", type=int, default=None, help="Smoke-test only: predict first N documents.")
    parser.add_argument("--out_root", default="outputs", help="Root output directory; default writes to outputs/<dataset>/<run>/proposals.")
    parser.add_argument("--out_dir", default=None, help="Direct proposal output directory; overrides --out_root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    device = detect_device(args.device)
    model, tokenizer, meta = load_proposal(checkpoint, device)
    docs = read_jsonl(args.input)
    if args.max_docs:
        docs = docs[: args.max_docs]
    proposals = predict_proposal_docs(
        model,
        tokenizer,
        docs,
        int(meta["max_length"]),
        int(meta["stride"]),
        int(meta.get("max_span_width", 128)),
        float(args.threshold if args.threshold is not None else checkpoint_threshold(checkpoint, meta.get("threshold", 0.5))),
        args.batch_size,
        device,
        candidate_chunk_size=int(meta.get("candidate_chunk_size", 32768)),
        max_proposals_per_document=int(meta.get("max_proposals_per_document", 200)),
    )
    split = args.split or Path(args.input).stem
    if args.out_dir:
        out_path = Path(args.out_dir) / f"{split}.jsonl"
    else:
        dataset = meta.get("dataset")
        if dataset:
            out_path = Path(args.out_root) / dataset / meta["run_name"] / "proposals" / f"{split}.jsonl"
        else:
            out_path = Path(args.out_root) / meta["run_name"] / f"{split}.jsonl"
    write_jsonl(out_path, proposals)
    n_props = sum(len(row["proposals"]) for row in proposals)
    print(f"{len(proposals)} documents, {n_props} proposals -> {out_path}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
