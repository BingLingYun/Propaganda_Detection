"""End-to-end prediction for the two-stage proposal + selected-span classifier baseline."""

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
from src.inference.checkpoint import load_classifier, load_proposal
from src.inference.two_stage_predict import predict_two_stage_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a two-stage propaganda span pipeline.")
    parser.add_argument("--proposal_checkpoint", required=True)
    parser.add_argument("--classifier_checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--proposal_threshold", type=float, default=None)
    parser.add_argument("--classifier_threshold", type=float, default=None)
    parser.add_argument("--max_docs", type=int, default=None, help="Smoke-test only: predict first N documents.")
    parser.add_argument("--out_root", default="outputs", help="Root output directory; default writes to outputs/<dataset>/<run>/predictions.")
    parser.add_argument("--out_dir", default=None, help="Direct prediction output directory; overrides --out_root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = detect_device(args.device)
    proposal_model, proposal_tokenizer, proposal_meta = load_proposal(Path(args.proposal_checkpoint), device)
    classifier_model, classifier_tokenizer, classifier_meta = load_classifier(Path(args.classifier_checkpoint), device)
    docs = read_jsonl(args.input)
    if args.max_docs:
        docs = docs[: args.max_docs]
    predictions = predict_two_stage_pipeline(
        proposal_model,
        classifier_model,
        proposal_tokenizer,
        classifier_tokenizer,
        docs,
        classifier_meta["labels"],
        int(proposal_meta["max_length"]),
        int(proposal_meta["stride"]),
        int(proposal_meta.get("max_span_width", 128)),
        float(args.proposal_threshold if args.proposal_threshold is not None else checkpoint_threshold(Path(args.proposal_checkpoint), proposal_meta.get("threshold", 0.5))),
        float(args.classifier_threshold if args.classifier_threshold is not None else checkpoint_threshold(Path(args.classifier_checkpoint), classifier_meta.get("threshold", 0.5))),
        args.batch_size,
        device,
        candidate_chunk_size=int(proposal_meta.get("candidate_chunk_size", 32768)),
        max_proposals_per_document=int(proposal_meta.get("max_proposals_per_document", 200)),
        nms_iou=float(classifier_meta.get("nms_iou", 0.5)),
        sentence_context_sentences=int(classifier_meta.get("sentence_context_sentences", 0)),
    )
    split = args.split or Path(args.input).stem
    run_name = f"{proposal_meta['run_name']}__{classifier_meta['run_name']}"
    if args.out_dir:
        out_path = Path(args.out_dir) / f"{split}.jsonl"
    else:
        dataset = proposal_meta.get("dataset") or classifier_meta.get("dataset")
        if dataset:
            out_path = Path(args.out_root) / dataset / run_name / "predictions" / f"{split}.jsonl"
        else:
            out_path = Path(args.out_root) / run_name / f"{split}.jsonl"
    write_jsonl(out_path, predictions)
    n_spans = sum(len(row["spans"]) for row in predictions)
    print(f"{len(predictions)} documents, {n_spans} spans -> {out_path}")


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
