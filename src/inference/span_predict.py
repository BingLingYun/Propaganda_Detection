"""Prediction helpers for two-stage proposal baselines."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.training.stage2_data import BILOUProposalWindowDataset, bilou_collate, bilou_tags_to_spans


def _iter_bilou_window_chunks(
    docs: list[dict[str, Any]],
    tokenizer,
    max_length: int,
    stride: int,
    max_windows: int,
):
    max_windows = max(1, int(max_windows))
    chunk_docs: list[dict[str, Any]] = []
    chunk_windows = 0
    for doc in docs:
        doc_windows = len(
            tokenizer(
                doc["text"],
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
            )["input_ids"]
        )
        if chunk_docs and chunk_windows + doc_windows > max_windows:
            yield chunk_docs
            chunk_docs = []
            chunk_windows = 0
        chunk_docs.append(doc)
        chunk_windows += doc_windows
    if chunk_docs:
        yield chunk_docs


def _merge_overlapping_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompose binary proposal fragments that overlap across sliding windows."""
    if not proposals:
        return []
    merged: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda item: (int(item["start"]), int(item["end"]), -float(item.get("score", 0.0)))):
        item = {
            "start": int(proposal["start"]),
            "end": int(proposal["end"]),
            "score": float(proposal.get("score", 0.0)),
        }
        if not merged or item["start"] >= merged[-1]["end"]:
            merged.append(item)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["score"] = max(float(merged[-1].get("score", 0.0)), item["score"])
    return merged


@torch.no_grad()
def predict_proposal_docs(
    proposal_model,
    tokenizer,
    docs: list[dict[str, Any]],
    max_length: int,
    stride: int,
    max_span_width: int,
    threshold: float,
    batch_size: int,
    device,
    candidate_chunk_size: int = 32768,
    max_proposals_per_document: int = 200,
) -> list[dict[str, Any]]:
    """Generate Stage 1 span proposals without technique labels using BILOU tags."""
    proposal_model.eval()
    doc_props: dict[str, list[dict[str, Any]]] = {str(doc["id"]): [] for doc in docs}
    doc_chunks = list(_iter_bilou_window_chunks(docs, tokenizer, max_length, stride, candidate_chunk_size))
    for doc_chunk in tqdm(doc_chunks, total=len(doc_chunks), desc="proposal chunks", dynamic_ncols=True, leave=False):
        dataset = BILOUProposalWindowDataset(doc_chunk, tokenizer, max_length, stride, with_labels=False)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: bilou_collate(batch, tokenizer.pad_token_id),
        )
        for batch in tqdm(loader, total=len(loader), desc="proposal predict", dynamic_ncols=True, leave=False):
            output = proposal_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                valid_mask=batch["valid_mask"].to(device),
            )
            probs = torch.softmax(output["logits"], dim=-1).detach().cpu()
            tags = probs.argmax(dim=-1)
            positive_scores = probs[..., 1:].max(dim=-1).values
            for row_index, meta in enumerate(batch["meta"]):
                positions = [pos for pos, valid in enumerate(meta["valid"]) if valid]
                row_tags = [int(tags[row_index, pos].item()) for pos in positions]
                row_scores = [float(positive_scores[row_index, pos].item()) for pos in positions]
                for span in bilou_tags_to_spans(
                    row_tags,
                    row_scores,
                    positions,
                    meta["offsets"],
                    max_span_width,
                    threshold,
                ):
                    doc_props[meta["doc_id"]].append(span)

    outputs = []
    for doc in docs:
        proposals = sorted(
            _merge_overlapping_proposals(doc_props[str(doc["id"])]),
            key=lambda item: item["score"],
            reverse=True,
        )
        unique: dict[tuple[int, int], dict[str, Any]] = {}
        for proposal in proposals:
            key = (proposal["start"], proposal["end"])
            if key not in unique:
                unique[key] = proposal
        kept = sorted(unique.values(), key=lambda item: item["score"], reverse=True)[:max_proposals_per_document]
        kept = sorted(kept, key=lambda item: (item["start"], item["end"]))
        outputs.append({"id": doc["id"], "proposals": kept})
    return outputs
