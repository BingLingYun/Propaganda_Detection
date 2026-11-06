"""Shared loading of neural baseline checkpoints for prediction and tuning."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.common.utils import read_json
from src.models.mh_bio_crf import XLMRMultiHeadBIOCRF
from src.models.two_stage.classifier import ZenSpanTechniqueClassifier
from src.models.two_stage.proposal import SpanProposalDetector


def load_model(checkpoint: str | Path, device: torch.device, *, expected_model_type: str | None = None):
    """Restore a model, tokenizer and metadata; validate type before loading weights."""
    checkpoint = Path(checkpoint)
    meta = read_json(checkpoint / "meta.json")
    model_type = meta.get("model_type", "mhbio_crf")
    if expected_model_type is not None and model_type != expected_model_type:
        raise ValueError(f"Expected {expected_model_type} checkpoint, got {model_type}")
    if model_type not in {"mhbio_crf", "ts_proposal", "ts_classifier"}:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", use_fast=True)
    encoder = AutoModel.from_pretrained(checkpoint / "encoder")
    common = {"encoder": encoder, "dropout": float(meta.get("dropout", 0.1))}
    if model_type == "mhbio_crf":
        model = XLMRMultiHeadBIOCRF(
            **common,
            num_labels=len(meta["labels"]),
            head_weighting=str(meta.get("head_weighting", "none")),
            negative_head_weight=float(meta.get("negative_head_weight", 1.0)),
            aux_bio_ce_weight=float(meta.get("aux_bio_ce_weight", 0.0)),
            bio_positive_weight=float(meta.get("bio_positive_weight", 1.0)),
            forbidden_score=float(meta.get("forbidden_score", -10000.0)),
        )
    elif model_type == "ts_proposal":
        model = SpanProposalDetector(**common)
    else:
        model = ZenSpanTechniqueClassifier(
            **common,
            num_labels=len(meta["labels"]),
            hidden_size=int(meta["classifier_hidden_size"]) if meta.get("classifier_hidden_size") else None,
            head_architecture=str(meta.get("head_architecture", "independent")),
        )
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer, meta


def load_proposal(checkpoint: str | Path, device: torch.device):
    return load_model(checkpoint, device, expected_model_type="ts_proposal")


def load_classifier(checkpoint: str | Path, device: torch.device):
    return load_model(checkpoint, device, expected_model_type="ts_classifier")
