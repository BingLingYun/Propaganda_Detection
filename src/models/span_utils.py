"""Span helpers shared by span and two-stage model baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch


def enumerate_token_spans(valid_mask: torch.Tensor, max_span_width: int) -> torch.Tensor:
    """Return candidate token spans as ``[batch, start, end]`` with inclusive ends."""
    if valid_mask.dim() != 2:
        raise ValueError("valid_mask must have shape [batch, seq_len]")
    if max_span_width <= 0:
        raise ValueError("max_span_width must be positive")
    candidates: list[tuple[int, int, int]] = []
    for batch_index in range(valid_mask.shape[0]):
        valid_positions = valid_mask[batch_index].nonzero(as_tuple=False).flatten().tolist()
        valid_set = set(valid_positions)
        for start in valid_positions:
            max_end = min(start + max_span_width - 1, valid_mask.shape[1] - 1)
            for end in range(start, max_end + 1):
                if end not in valid_set:
                    break
                candidates.append((batch_index, start, end))
    if not candidates:
        return torch.empty((0, 3), dtype=torch.long, device=valid_mask.device)
    return torch.tensor(candidates, dtype=torch.long, device=valid_mask.device)


def chunk_candidates(candidates: torch.Tensor, chunk_size: int) -> Iterable[torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, candidates.shape[0], chunk_size):
        yield candidates[start : start + chunk_size]


def char_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return 0.0 if union <= 0 else inter / union


def nms_same_label(spans: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Class-wise NMS. Different labels never suppress each other."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        grouped[str(span["technique"])].append(span)

    kept: list[dict] = []
    for technique, items in grouped.items():
        remaining = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        selected: list[dict] = []
        while remaining:
            current = remaining.pop(0)
            selected.append(current)
            remaining = [
                candidate
                for candidate in remaining
                if char_iou(current["start"], current["end"], candidate["start"], candidate["end"]) <= iou_threshold
            ]
        kept.extend(selected)
    return sorted(kept, key=lambda x: (x["start"], x["end"], str(x["technique"])))


def merge_overlapping_same_label(spans: list[dict]) -> list[dict]:
    """Merge transitive same-label character overlaps; keep cross-label overlaps."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        grouped[str(span["technique"])].append(span)

    merged: list[dict] = []
    for technique, items in grouped.items():
        items = sorted(items, key=lambda x: (x["start"], x["end"]))
        if not items:
            continue
        current = dict(items[0])
        current["technique"] = technique
        current["score"] = float(current.get("score", 0.0))
        for item in items[1:]:
            if max(current["start"], item["start"]) < min(current["end"], item["end"]):
                current["end"] = max(current["end"], item["end"])
                current["score"] = max(float(current.get("score", 0.0)), float(item.get("score", 0.0)))
            else:
                merged.append(current)
                current = dict(item)
                current["technique"] = technique
                current["score"] = float(current.get("score", 0.0))
        merged.append(current)
    return sorted(merged, key=lambda x: (x["start"], x["end"], str(x["technique"])))


def candidate_offsets(
    candidates: torch.Tensor,
    offsets: torch.Tensor,
) -> list[tuple[int, int, int, int]]:
    """Map token candidates to ``(candidate_index, batch, char_start, char_end)``."""
    mapped: list[tuple[int, int, int, int]] = []
    for candidate_index, (batch_index, token_start, token_end) in enumerate(candidates.detach().cpu().tolist()):
        char_start = int(offsets[batch_index, token_start, 0].item())
        char_end = int(offsets[batch_index, token_end, 1].item())
        if char_start < char_end:
            mapped.append((candidate_index, batch_index, char_start, char_end))
    return mapped
