"""Unified neural baseline training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a transitive transformers dependency.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.common.device import autocast_for, detect_device, make_grad_scaler, memory_allocated_gb, optimizer_step_was_run
from src.common.experiment import make_neural_run_name, runtime_snapshot
from src.common.utils import get_labels, load_yaml, read_jsonl, set_seed, write_json
from src.evaluation.evaluate import merge_location_docs, partial_match_prf
from src.inference.span_predict import predict_proposal_docs
from src.inference.two_stage_predict import classify_proposals
from src.modeling.data import WindowDataset, collate
from src.modeling.inference import predict_docs
from src.models.mh_bio_crf import XLMRMultiHeadBIOCRF
from src.models.two_stage.classifier import ZenSpanTechniqueClassifier
from src.models.two_stage.proposal import SpanProposalDetector
from src.training.config import merge_config
from src.training.stage2_data import (
    BILOUProposalWindowDataset,
    Stage2SamplingConfig,
    ZenStage2CandidateDataset,
    bilou_collate,
    read_proposal_file,
    zen_stage2_collate,
)


MODEL_TYPES = ["mhbio_crf", "ts_proposal", "ts_classifier"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural propaganda span baselines.")
    parser.add_argument("--model_type", default=None, choices=MODEL_TYPES)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--model_config", default="configs/models/xlm_mhbio_crf.yaml")
    parser.add_argument("--train_config", default="configs/train_default.yaml")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--run_name", default=None, help="Optional explicit output run name.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accumulation", "--grad_accum", dest="grad_accumulation", type=int, default=None)
    parser.add_argument("--encoder_lr", "--lr_encoder", dest="encoder_lr", type=float, default=None)
    parser.add_argument("--head_lr", "--lr_head", dest="head_lr", type=float, default=None)
    parser.add_argument("--task_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--grad_clip", "--max_grad_norm", dest="grad_clip", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_span_width_cap", type=int, default=None)
    parser.add_argument("--candidate_chunk_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pos_weight_cap", type=float, default=None)
    parser.add_argument("--head_weighting", choices=["none", "inverse_sqrt"], default=None)
    parser.add_argument("--negative_head_weight", type=float, default=None)
    parser.add_argument("--aux_bio_ce_weight", type=float, default=None)
    parser.add_argument("--bio_positive_weight", type=float, default=None)
    parser.add_argument("--patience", "--early_stopping_patience", dest="patience", type=int, default=None)
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=None)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dev_sample", type=int, default=None)
    parser.add_argument("--max_train_docs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--proposal_file", default=None)
    parser.add_argument("--deduplicate_train_windows", dest="deduplicate_train_windows", action="store_true", default=None)
    parser.add_argument("--no_deduplicate_train_windows", dest="deduplicate_train_windows", action="store_false")
    parser.add_argument("--lazy_windows", dest="lazy_windows", action="store_true", default=None)
    parser.add_argument("--no_lazy_windows", dest="lazy_windows", action="store_false")
    parser.add_argument("--lazy_window_cache_docs", type=int, default=None)
    parser.add_argument("--negative_positive_ratio", type=int, default=None)
    parser.add_argument("--max_negative_no_positive", type=int, default=None)
    parser.add_argument("--boundary_perturb_radius", type=int, default=None)
    parser.add_argument(
        "--positive_overlap_mode",
        choices=["iou", "gold_coverage", "candidate_coverage", "either_coverage"],
        default=None,
    )
    parser.add_argument("--out_root", default="outputs", help="Root output directory; checkpoints go under outputs/<dataset>/<run>/checkpoints.")
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    return parser.parse_args()


def dataset_output_root(out_root: str | Path, dataset_name: str) -> Path:
    root = Path(out_root)
    return root if root.name == dataset_name else root / dataset_name


def log_jsonl(path: Path, record: dict[str, Any]) -> None:
    record = {**record, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def optimizer_groups(model: torch.nn.Module, encoder_lr: float, task_lr: float, weight_decay: float) -> list[dict[str, Any]]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    encoder_params = []
    task_params = []
    for name, param in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("scorer.encoder."):
            encoder_params.append((name, param))
        else:
            task_params.append((name, param))

    def split(params: list[tuple[str, torch.nn.Parameter]], lr: float) -> list[dict[str, Any]]:
        decay = [param for name, param in params if not any(key in name for key in no_decay)]
        nodecay = [param for name, param in params if any(key in name for key in no_decay)]
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": nodecay, "lr": lr, "weight_decay": 0.0},
        ]

    return split(encoder_params, encoder_lr) + split(task_params, task_lr)


def save_checkpoint(
    best_dir: Path,
    model: torch.nn.Module,
    tokenizer,
    labels: list[str],
    cfg: dict[str, Any],
    run_name: str,
    epoch: int,
    score: float,
) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    encoder = model.scorer.encoder if hasattr(model, "scorer") else model.encoder
    encoder_state = {
        name: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
        for name, tensor in encoder.state_dict().items()
    }
    encoder.save_pretrained(best_dir / "encoder", state_dict=encoder_state)
    tokenizer.save_pretrained(best_dir / "tokenizer")
    model_state = {
        name: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
        for name, tensor in model.state_dict().items()
    }
    torch.save(model_state, best_dir / "model.pt")
    write_json(
        best_dir / "meta.json",
        {
            "run_name": run_name,
            "dataset": cfg.get("dataset"),
            "model_type": cfg["model_type"],
            "model_name": cfg["model_name"],
            "epoch": epoch,
            "selection_score": score,
            "labels": labels,
            "max_length": int(cfg["max_length"]),
            "stride": int(cfg["stride"]),
            "merge_gap": int(cfg.get("merge_gap", 0)),
            "dropout": float(cfg.get("dropout", 0.1)),
            "head_weighting": str(cfg.get("head_weighting", "none")),
            "negative_head_weight": float(cfg.get("negative_head_weight", 1.0)),
            "aux_bio_ce_weight": float(cfg.get("aux_bio_ce_weight", 0.0)),
            "bio_positive_weight": float(cfg.get("bio_positive_weight", 1.0)),
            "deduplicate_train_windows": bool(cfg.get("deduplicate_train_windows", True)),
            "lazy_windows": bool(cfg.get("lazy_windows", False)),
            "lazy_window_cache_docs": int(cfg.get("lazy_window_cache_docs", 2)),
            "forbidden_score": float(cfg.get("forbidden_score", -10000.0)),
            "span_hidden_size": int(cfg.get("span_hidden_size", 256)),
            "span_mlp_layers": int(cfg.get("span_mlp_layers", 1)),
            "max_span_width": int(cfg.get("max_span_width_cap", 128)),
            "pos_weight_cap": float(cfg.get("pos_weight_cap", 20.0)),
            "candidate_chunk_size": int(cfg.get("candidate_chunk_size", 32768)),
            "threshold": float(cfg.get("threshold", 0.5)),
            "nms_iou": float(cfg.get("nms_iou", 0.5)),
            "max_proposals_per_document": int(cfg.get("max_proposals_per_document", 200)),
            "proposal_architecture": str(cfg.get("proposal_architecture", "")),
            "stage2_architecture": str(cfg.get("stage2_architecture", "")),
            "stage2_input_mode": str(cfg.get("stage2_input_mode", "")),
            "head_architecture": str(cfg.get("head_architecture", "shared")),
            "classifier_hidden_size": cfg.get("classifier_hidden_size"),
            "sentence_context_sentences": int(cfg.get("sentence_context_sentences", 0)),
            "positive_iou_threshold": float(cfg.get("positive_iou_threshold", 0.5)),
            "negative_iou_threshold": float(cfg.get("negative_iou_threshold", 0.1)),
            "positive_overlap_mode": str(cfg.get("positive_overlap_mode", "gold_coverage")),
            "negative_positive_ratio": int(cfg.get("negative_positive_ratio", cfg.get("positive_negative_ratio_cap", 5))),
            "max_negative_no_positive": int(cfg.get("max_negative_no_positive", 256)),
            "boundary_perturb_radius": int(cfg.get("boundary_perturb_radius", 0)),
        },
    )


def scheduler_for(optimizer, loader_len: int, cfg: dict[str, Any]):
    steps_per_epoch = max(1, math.ceil(loader_len / int(cfg["grad_accumulation"])))
    total_steps = steps_per_epoch * int(cfg["epochs"])
    if int(cfg.get("max_steps", -1)) > 0:
        total_steps = min(total_steps, int(cfg["max_steps"]))
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(cfg["warmup_ratio"])),
        num_training_steps=max(1, total_steps),
    )


def accumulation_denominator(batch_index: int, loader_len: int, grad_accumulation: int) -> int:
    group_start = (batch_index // grad_accumulation) * grad_accumulation
    return min(grad_accumulation, loader_len - group_start)


def head_weights_for_docs(train_docs: list[dict[str, Any]], labels: list[str], mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "inverse_sqrt":
        raise ValueError(f"Unsupported head_weighting: {mode}")
    counts = {label: 0 for label in labels}
    for doc in train_docs:
        for span in doc.get("spans", []):
            technique = span.get("technique")
            if technique in counts:
                counts[technique] += 1
    values = torch.tensor([1.0 / math.sqrt(max(1, counts[label])) for label in labels], dtype=torch.float32)
    return values / values.mean().clamp_min(1e-8)


def train_mh_bio_crf(args, cfg, ds_cfg, labels, tokenizer, device, fp16, out_dir, best_dir, log_path, run_name):
    processed_dir = Path(ds_cfg["processed_dir"])
    train_docs = read_jsonl(processed_dir / "train.jsonl")
    if args.max_train_docs:
        train_docs = train_docs[: args.max_train_docs]
    dev_docs = read_jsonl(processed_dir / "dev.jsonl")
    train_dataset = WindowDataset(
        train_docs,
        tokenizer,
        labels,
        int(cfg["max_length"]),
        int(cfg["stride"]),
        single_label=False,
        with_labels=True,
        deduplicate_training_tokens=bool(cfg.get("deduplicate_train_windows", True)),
        lazy=bool(cfg.get("lazy_windows", False)),
        cache_docs=int(cfg.get("lazy_window_cache_docs", 2)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    print(f"train={len(train_docs)} docs/{len(train_dataset)} windows  dev={len(dev_docs)} docs", flush=True)

    print("Loading model...", flush=True)
    model = XLMRMultiHeadBIOCRF.from_pretrained(
        model_name=cfg["model_name"],
        num_labels=len(labels),
        dropout=float(cfg.get("dropout", 0.1)),
        head_weighting=str(cfg.get("head_weighting", "none")),
        negative_head_weight=float(cfg.get("negative_head_weight", 1.0)),
        aux_bio_ce_weight=float(cfg.get("aux_bio_ce_weight", 0.0)),
        bio_positive_weight=float(cfg.get("bio_positive_weight", 1.0)),
        forbidden_score=float(cfg.get("forbidden_score", -10000.0)),
    ).to(device)
    memory_gb = memory_allocated_gb(device)
    if memory_gb is None:
        print("Model loaded", flush=True)
    else:
        print(f"Model loaded, accelerator memory: {memory_gb:.2f} GB", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(float(cfg["encoder_lr"]), float(cfg["head_lr"]), float(cfg["weight_decay"]))
    )
    scheduler = scheduler_for(optimizer, len(train_loader), cfg)
    scaler = make_grad_scaler(device, fp16)
    head_weights = head_weights_for_docs(train_docs, labels, str(cfg.get("head_weighting", "none")))
    print(
        f"CRF head weighting: mode={cfg.get('head_weighting', 'none')} "
        f"negative_head_weight={float(cfg.get('negative_head_weight', 1.0))} "
        f"aux_bio_ce_weight={float(cfg.get('aux_bio_ce_weight', 0.0))} "
        f"bio_positive_weight={float(cfg.get('bio_positive_weight', 1.0))}",
        flush=True,
    )
    if head_weights is not None:
        print(f"Using {cfg['head_weighting']} head weights: {[round(float(v), 4) for v in head_weights]}", flush=True)
    print(f"Optimizer ready, starting training loop", flush=True)

    best_score = -1.0
    patience_left = int(cfg["early_stopping_patience"])
    global_step = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        print(f"epoch {epoch} starting...", flush=True)
        model.train()
        total_loss = 0.0
        batches = 0
        optimizer.zero_grad()
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch}/{int(cfg['epochs'])}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch_index, batch in progress:
            labels_tensor = batch["labels"].transpose(1, 2).clamp_min(0).contiguous().to(device)
            with autocast_for(device, fp16):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=labels_tensor,
                    valid_mask=batch["valid_mask"].to(device),
                    head_weights=head_weights,
                )
                loss = output["loss"] / accumulation_denominator(
                    batch_index,
                    len(train_loader),
                    int(cfg["grad_accumulation"]),
                )
            scaler.scale(loss).backward()
            total_loss += float(output["loss"].item())
            batches += 1
            progress.set_postfix(loss=f"{total_loss / max(1, batches):.4f}", step=global_step)
            if batches % int(cfg["grad_accumulation"]) == 0 or (batch_index + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                optimizer_was_run = optimizer_step_was_run(scaler, optimizer)
                optimizer.zero_grad()
                if optimizer_was_run:
                    scheduler.step()
                    global_step += 1
                    if 0 < int(cfg.get("max_steps", -1)) <= global_step:
                        break

        eval_docs = sampled_dev_docs(dev_docs, cfg)
        print(f"epoch {epoch}: dev eval...", flush=True)
        predictions = predict_docs(
            model,
            tokenizer,
            eval_docs,
            labels,
            int(cfg["max_length"]),
            int(cfg["stride"]),
            int(cfg.get("merge_gap", 0)),
            int(cfg["batch_size"]),
            device,
            single_label=False,
        )
        metrics = partial_match_prf(
            {doc["id"]: doc["spans"] for doc in eval_docs},
            {doc["id"]: doc["spans"] for doc in predictions},
            use_class=True,
        )
        avg_loss = total_loss / max(1, batches)
        score = float(metrics["f1"])
        pred_counts = Counter(span["technique"] for doc in predictions for span in doc.get("spans", []))
        top_pred = ", ".join(f"{label}:{count}" for label, count in pred_counts.most_common(5)) or "none"
        pred_spans = sum(pred_counts.values())
        print(
            f"epoch {epoch}: loss={avg_loss:.4f} dev_micro_f1={score:.4f} "
            f"pred_spans={pred_spans} top_pred=[{top_pred}]"
        )
        log_jsonl(
            log_path,
            {
                "type": "epoch",
                "epoch": epoch,
                "loss": avg_loss,
                "dev_micro_f1": score,
                "pred_spans": pred_spans,
                "pred_label_counts": dict(pred_counts),
            },
        )
        if score > best_score:
            best_score = score
            patience_left = int(cfg["early_stopping_patience"])
            save_checkpoint(best_dir, model, tokenizer, labels, cfg, run_name, epoch, score)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        if 0 < int(cfg.get("max_steps", -1)) <= global_step:
            break
    print(f"training complete: best_score={best_score:.4f}, checkpoint={best_dir}")


def sampled_dev_docs(dev_docs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dev_sample = int(cfg.get("dev_sample", -1))
    if 0 < dev_sample < len(dev_docs):
        import random

        return random.Random(int(cfg["seed"])).sample(dev_docs, dev_sample)
    return dev_docs


def binary_span_docs(docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        doc["id"]: [
            {"start": span["start"], "end": span["end"], "technique": "PROPAGANDA"}
            for span in doc.get("spans", [])
        ]
        for doc in docs
    }


def proposal_span_docs(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["id"]: [
            {"start": proposal["start"], "end": proposal["end"], "technique": "PROPAGANDA"}
            for proposal in row.get("proposals", [])
        ]
        for row in rows
    }


def build_span_model(cfg: dict[str, Any], num_labels: int):
    if cfg["model_type"] == "ts_proposal":
        return SpanProposalDetector.from_pretrained(
            model_name=cfg["model_name"],
            span_hidden_size=int(cfg.get("span_hidden_size", 256)),
            span_mlp_layers=int(cfg.get("span_mlp_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    if cfg["model_type"] == "ts_classifier":
        return ZenSpanTechniqueClassifier.from_pretrained(
            model_name=cfg["model_name"],
            num_labels=num_labels,
            dropout=float(cfg.get("dropout", 0.1)),
            hidden_size=int(cfg["classifier_hidden_size"]) if cfg.get("classifier_hidden_size") else None,
            head_architecture=str(cfg.get("head_architecture", "independent")),
        )
    raise ValueError(f"Unsupported span model type: {cfg['model_type']}")


def bilou_class_weight_for_dataset(dataset: BILOUProposalWindowDataset, positive_weight: float) -> torch.Tensor | None:
    if positive_weight <= 0:
        return None
    counts = torch.zeros(5, dtype=torch.float64)
    for index in range(len(dataset)):
        labels = torch.tensor(dataset[index]["labels"], dtype=torch.long)
        valid = labels >= 0
        if valid.any():
            counts += torch.bincount(labels[valid], minlength=5).to(dtype=torch.float64)
    weights = torch.ones(5, dtype=torch.float32)
    for tag_id in range(1, 5):
        if counts[tag_id] > 0:
            weights[tag_id] = float(positive_weight)
    return weights


def multilabel_pos_weight_for_dataset(dataset: ZenStage2CandidateDataset, num_labels: int, cap: float) -> torch.Tensor | None:
    if cap <= 0:
        return None
    positives = torch.zeros(num_labels, dtype=torch.float64)
    totals = torch.zeros(num_labels, dtype=torch.float64)
    for record in dataset.records:
        labels = record.get("labels")
        if labels is None:
            continue
        labels = labels.to(dtype=torch.float64)
        positives += labels
        totals += 1
    negatives = totals - positives
    weights = torch.ones(num_labels, dtype=torch.float32)
    seen = positives > 0
    weights[seen] = (negatives[seen] / positives[seen].clamp_min(1.0)).to(dtype=torch.float32)
    return weights.clamp(min=1.0, max=float(cap))


def train_two_stage_proposal(args, cfg, ds_cfg, labels, tokenizer, device, fp16, out_dir, best_dir, log_path, run_name):
    del labels
    processed_dir = Path(ds_cfg["processed_dir"])
    train_docs = read_jsonl(processed_dir / "train.jsonl")
    if args.max_train_docs:
        train_docs = train_docs[: args.max_train_docs]
    dev_docs = read_jsonl(processed_dir / "dev.jsonl")
    train_dataset = BILOUProposalWindowDataset(
        train_docs,
        tokenizer,
        int(cfg["max_length"]),
        int(cfg["stride"]),
        with_labels=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=lambda batch: bilou_collate(batch, tokenizer.pad_token_id),
    )
    print(f"train={len(train_docs)} docs/{len(train_dataset)} BILOU windows  dev={len(dev_docs)} docs")
    class_weight = bilou_class_weight_for_dataset(train_dataset, float(cfg.get("proposal_positive_weight", 4.0)))
    if class_weight is not None:
        print(f"Using BILOU class weights: {[round(float(value), 4) for value in class_weight.tolist()]}", flush=True)
        class_weight = class_weight.to(device)
    model = build_span_model(cfg, 1).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, float(cfg["encoder_lr"]), float(cfg["task_lr"]), float(cfg["weight_decay"]))
    )
    scheduler = scheduler_for(optimizer, len(train_loader), cfg)
    scaler = make_grad_scaler(device, fp16)
    best_score = -1.0
    patience_left = int(cfg["early_stopping_patience"])
    global_step = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        optimizer.zero_grad()
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch}/{int(cfg['epochs'])}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch_index, batch in progress:
            with autocast_for(device, fp16):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    valid_mask=batch["valid_mask"].to(device),
                    class_weight=class_weight,
                )
                loss = output["loss"] / accumulation_denominator(
                    batch_index,
                    len(train_loader),
                    int(cfg["grad_accumulation"]),
                )
            scaler.scale(loss).backward()
            total_loss += float(output["loss"].item())
            batches += 1
            progress.set_postfix(loss=f"{total_loss / max(1, batches):.4f}", step=global_step)
            if batches % int(cfg["grad_accumulation"]) == 0 or (batch_index + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                optimizer_was_run = optimizer_step_was_run(scaler, optimizer)
                optimizer.zero_grad()
                if optimizer_was_run:
                    scheduler.step()
                    global_step += 1
                    if 0 < int(cfg.get("max_steps", -1)) <= global_step:
                        break

        eval_docs = sampled_dev_docs(dev_docs, cfg)
        proposal_rows = predict_proposal_docs(
            model,
            tokenizer,
            eval_docs,
            int(cfg["max_length"]),
            int(cfg["stride"]),
            int(cfg.get("max_span_width_cap", 128)),
            float(cfg.get("threshold", 0.5)),
            int(cfg["batch_size"]),
            device,
            candidate_chunk_size=int(cfg.get("candidate_chunk_size", 32768)),
            max_proposals_per_document=int(cfg.get("max_proposals_per_document", 200)),
        )
        metrics = partial_match_prf(
            merge_location_docs(binary_span_docs(eval_docs)),
            merge_location_docs(proposal_span_docs(proposal_rows)),
            use_class=False,
        )
        avg_loss = total_loss / max(1, batches)
        score = float(metrics["recall"])
        print(
            f"epoch {epoch}: loss={avg_loss:.4f} "
            f"dev_proposal_recall={metrics['recall']:.4f} dev_location_f1={metrics['f1']:.4f}"
        )
        log_jsonl(
            log_path,
            {
                "type": "epoch",
                "epoch": epoch,
                "loss": avg_loss,
                "dev_proposal_precision": metrics["precision"],
                "dev_proposal_recall": metrics["recall"],
                "dev_location_f1": metrics["f1"],
            },
        )
        if score > best_score:
            best_score = score
            patience_left = int(cfg["early_stopping_patience"])
            save_checkpoint(best_dir, model, tokenizer, ["PROPAGANDA"], cfg, run_name, epoch, score)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        if 0 < int(cfg.get("max_steps", -1)) <= global_step:
            break
    print(f"training complete: checkpoint={best_dir}")


def train_two_stage_classifier(args, cfg, ds_cfg, labels, tokenizer, device, fp16, out_dir, best_dir, log_path, run_name):
    processed_dir = Path(ds_cfg["processed_dir"])
    train_docs = read_jsonl(processed_dir / "train.jsonl")
    if args.max_train_docs:
        train_docs = train_docs[: args.max_train_docs]
    dev_docs = read_jsonl(processed_dir / "dev.jsonl")
    proposals_by_doc = read_proposal_file(args.proposal_file or cfg.get("proposal_file"))
    if not proposals_by_doc:
        print("Warning: no proposal_file provided; Stage 2 will train on gold selected spans only.", flush=True)
    model = build_span_model(cfg, len(labels)).to(device)
    sampling = Stage2SamplingConfig(
        max_span_width=int(cfg.get("max_span_width_cap", 128)),
        positive_iou_threshold=float(cfg.get("positive_iou_threshold", 0.5)),
        negative_iou_threshold=float(cfg.get("negative_iou_threshold", 0.1)),
        positive_overlap_mode=str(cfg.get("positive_overlap_mode", "gold_coverage")),
        negative_positive_ratio=int(cfg.get("negative_positive_ratio", cfg.get("positive_negative_ratio_cap", 5))),
        max_negative_no_positive=int(cfg.get("max_negative_no_positive", 256)),
        boundary_perturb_radius=int(cfg.get("boundary_perturb_radius", 0)),
        include_gold_candidates=bool(cfg.get("include_gold_candidates", True)),
        sentence_context_sentences=int(cfg.get("sentence_context_sentences", 0)),
        seed=int(cfg["seed"]),
    )
    train_dataset = ZenStage2CandidateDataset(
        train_docs,
        tokenizer,
        labels,
        int(cfg["max_length"]),
        sampling,
        proposals_by_doc=proposals_by_doc,
        with_labels=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=lambda batch: zen_stage2_collate(batch, tokenizer.pad_token_id),
    )
    print(f"train={len(train_docs)} docs/{len(train_dataset)} Zen stage2 candidates  dev={len(dev_docs)} docs")
    pos_weight = multilabel_pos_weight_for_dataset(
        train_dataset,
        len(labels),
        float(cfg.get("pos_weight_cap", 20.0)),
    )
    if pos_weight is not None:
        print(f"Using Stage2 pos_weight: {[round(float(value), 4) for value in pos_weight.tolist()]}", flush=True)
        pos_weight = pos_weight.to(device)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, float(cfg["encoder_lr"]), float(cfg["classifier_lr"]), float(cfg["weight_decay"]))
    )
    scheduler = scheduler_for(optimizer, len(train_loader), cfg)
    scaler = make_grad_scaler(device, fp16)
    best_score = -1.0
    best_loss = float("inf")
    patience_left = int(cfg["early_stopping_patience"])
    global_step = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        optimizer.zero_grad()
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch}/{int(cfg['epochs'])}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch_index, batch in progress:
            with autocast_for(device, fp16):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    span_token_mask=batch["span_token_mask"].to(device),
                    labels=batch["labels"].to(device),
                    pos_weight=pos_weight,
                )
                loss = output["loss"] / accumulation_denominator(
                    batch_index,
                    len(train_loader),
                    int(cfg["grad_accumulation"]),
                )
            scaler.scale(loss).backward()
            total_loss += float(output["loss"].item())
            batches += 1
            progress.set_postfix(loss=f"{total_loss / max(1, batches):.4f}", step=global_step)
            if (batch_index + 1) % int(cfg["grad_accumulation"]) == 0 or (batch_index + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                optimizer_was_run = optimizer_step_was_run(scaler, optimizer)
                optimizer.zero_grad()
                if optimizer_was_run:
                    scheduler.step()
                    global_step += 1
                    if 0 < int(cfg.get("max_steps", -1)) <= global_step:
                        break
        avg_loss = total_loss / max(1, batches)
        dev_proposals = {
            str(doc["id"]): proposals_by_doc[str(doc["id"])]
            for doc in dev_docs
            if str(doc["id"]) in proposals_by_doc
        }
        if dev_proposals:
            eval_docs = sampled_dev_docs(dev_docs, cfg)
            eval_proposals = {str(doc["id"]): dev_proposals.get(str(doc["id"]), []) for doc in eval_docs}
            print(f"epoch {epoch}: dev classifier eval...", flush=True)
            predictions = classify_proposals(
                model,
                tokenizer,
                eval_docs,
                eval_proposals,
                labels,
                int(cfg["max_length"]),
                float(cfg.get("threshold", 0.5)),
                int(cfg["batch_size"]),
                device,
                nms_iou=float(cfg.get("nms_iou", 0.5)),
                sentence_context_sentences=int(cfg.get("sentence_context_sentences", 0)),
                candidate_chunk_size=int(cfg.get("candidate_chunk_size", 32768)),
            )
            metrics = partial_match_prf(
                {doc["id"]: doc["spans"] for doc in eval_docs},
                {doc["id"]: doc["spans"] for doc in predictions},
                use_class=True,
            )
            score = float(metrics["f1"])
            improved = score > best_score
            print(f"epoch {epoch}: loss={avg_loss:.4f} dev_micro_f1={score:.4f}")
            log_jsonl(log_path, {"type": "epoch", "epoch": epoch, "loss": avg_loss, "dev_micro_f1": score})
        else:
            score = -avg_loss
            improved = avg_loss < best_loss
            print(f"epoch {epoch}: loss={avg_loss:.4f} selection=-loss (no dev proposals)")
            log_jsonl(log_path, {"type": "epoch", "epoch": epoch, "loss": avg_loss, "selection": "train_loss"})
        if improved:
            best_score = max(best_score, score)
            best_loss = avg_loss
            patience_left = int(cfg["early_stopping_patience"])
            save_checkpoint(best_dir, model, tokenizer, labels, cfg, run_name, epoch, score)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        if 0 < int(cfg.get("max_steps", -1)) <= global_step:
            break
    print(f"training complete: checkpoint={best_dir}")


def main() -> None:
    start_time = time.time()
    args = parse_args()
    ds_cfg = load_yaml(args.dataset_config)
    cfg = merge_config(args, ds_cfg)
    set_seed(int(cfg["seed"]))
    if cfg["model_type"] not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={cfg['model_type']}; expected one of {MODEL_TYPES}")

    labels = get_labels(ds_cfg)
    run_name = args.run_name or make_neural_run_name(ds_cfg["name"], cfg)
    run_root = dataset_output_root(args.out_root, ds_cfg["name"]) / run_name
    out_dir = run_root / "checkpoints"
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "trainer_log.jsonl"
    device = detect_device(args.device)
    fp16 = bool(cfg.get("fp16")) and device.type in {"cuda", "npu"}
    print(f"run_name={run_name}  model_type={cfg['model_type']}  device={device}  fp16={fp16}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(f"Tokenizer for {cfg['model_name']} is not fast; offset mapping is required.")
    write_json(
        out_dir / "resolved_config.json",
        {
            "run_name": run_name,
            "dataset_config": args.dataset_config,
            "model_config": args.model_config,
            "dataset": ds_cfg["name"],
            "labels": labels,
            **cfg,
        },
    )

    if cfg["model_type"] == "mhbio_crf":
        train_mh_bio_crf(args, cfg, ds_cfg, labels, tokenizer, device, fp16, out_dir, best_dir, log_path, run_name)
    elif cfg["model_type"] == "ts_proposal":
        train_two_stage_proposal(args, cfg, ds_cfg, labels, tokenizer, device, fp16, out_dir, best_dir, log_path, run_name)
    elif cfg["model_type"] == "ts_classifier":
        train_two_stage_classifier(args, cfg, ds_cfg, labels, tokenizer, device, fp16, out_dir, best_dir, log_path, run_name)
    else:
        raise ValueError(f"Unsupported model_type={cfg['model_type']}; expected one of {MODEL_TYPES}")
    write_json(
        out_dir / "runtime.json",
        runtime_snapshot(
            run_name=run_name,
            cfg=cfg,
            ds_cfg=ds_cfg,
            start_time=start_time,
            checkpoint=best_dir,
            device=device,
        ),
    )


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
