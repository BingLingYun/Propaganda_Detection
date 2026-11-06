"""Configuration resolution shared by training and run-name tools."""

from __future__ import annotations

import argparse
from typing import Any

from src.common.experiment import select_model_name
from src.common.utils import load_yaml


def merge_config(args: argparse.Namespace, ds_cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = load_yaml(args.train_config)
    model_cfg = load_yaml(args.model_config)
    cfg: dict[str, Any] = {**train_cfg, **model_cfg}
    if cfg.get("model_type") == "ts_pipeline":
        stage1 = cfg.get("stage1") or {}
        cfg = {**cfg, **stage1, "model_type": "ts_proposal"}
    if args.model_type == "ts_classifier" and "stage2" in cfg:
        cfg = {**cfg, **(cfg.get("stage2") or {}), "model_type": "ts_classifier"}
    rename = {
        "lr_encoder": "encoder_lr",
        "lr_head": "head_lr",
        "grad_accum": "grad_accumulation",
        "max_grad_norm": "grad_clip",
        "patience": "early_stopping_patience",
    }
    for old, new in rename.items():
        if old in cfg and new not in cfg:
            cfg[new] = cfg[old]
    for key, value in vars(args).items():
        if key in {"dataset_config", "model_config", "train_config", "out_root", "device"}:
            continue
        if value is not None:
            cfg[rename.get(key, key)] = value
    cfg.setdefault("model_type", "mhbio_crf")
    cfg.setdefault("seed", 42)
    cfg.setdefault("max_length", 512)
    cfg.setdefault("stride", 128)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("encoder_lr", 2.0e-5)
    cfg.setdefault("head_lr", cfg.get("task_lr", 1.0e-3))
    cfg.setdefault("task_lr", cfg.get("head_lr", 1.0e-3))
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("warmup_ratio", 0.1)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("grad_accumulation", 2)
    cfg.setdefault("epochs", 10)
    cfg.setdefault("early_stopping_patience", 3)
    cfg.setdefault("merge_gap", 0)
    cfg.setdefault("fp16", False)
    cfg.setdefault("head_weighting", "none")
    cfg.setdefault("negative_head_weight", 1.0)
    cfg.setdefault("aux_bio_ce_weight", 0.0)
    cfg.setdefault("bio_positive_weight", 1.0)
    cfg.setdefault("max_steps", -1)
    cfg.setdefault("dev_sample", -1)
    cfg.setdefault("num_workers", 0)
    cfg.setdefault("deduplicate_train_windows", True)
    cfg.setdefault("lazy_windows", False)
    cfg.setdefault("lazy_window_cache_docs", 2)
    cfg.setdefault("max_span_width_cap", cfg.get("max_span_width", 128))
    cfg.setdefault("negative_positive_ratio", cfg.get("positive_negative_ratio_cap", 5))
    cfg.setdefault("max_negative_no_positive", 256)
    cfg.setdefault("boundary_perturb_radius", 0)
    cfg.setdefault("pos_weight_cap", 20.0)
    cfg.setdefault("candidate_chunk_size", 32768)
    cfg.setdefault("nms_iou", 0.5)
    cfg.setdefault("threshold", 0.5)
    cfg.setdefault("classifier_lr", cfg.get("task_lr", 1.0e-4))
    cfg.setdefault("proposal_architecture", "bilou_token_tagger")
    cfg.setdefault("proposal_positive_weight", 4.0)
    cfg.setdefault("stage2_architecture", "zen_selected_span")
    cfg.setdefault("stage2_input_mode", "selected_span_context")
    cfg.setdefault("head_architecture", "independent")
    cfg.setdefault("sentence_context_sentences", 0)
    cfg.setdefault("positive_iou_threshold", 0.5)
    cfg.setdefault("negative_iou_threshold", 0.1)
    cfg.setdefault("positive_overlap_mode", "gold_coverage")
    cfg.setdefault("positive_negative_ratio_cap", cfg.get("negative_positive_ratio", 5))
    cfg["model_name"] = select_model_name(cfg, ds_cfg, explicit_model_name=args.model_name)
    cfg["dataset"] = ds_cfg["name"]
    return cfg
