"""Data utilities for the HALT-PROP stage 1 and ZenPropaganda stage 2 baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.models.span_utils import char_iou
from src.models.two_stage.proposal import B_TAG, I_TAG, IGNORE, L_TAG, O_TAG, U_TAG


SENTENCE_BOUNDARY_CHARS = set(".!?\u3002\uff01\uff1f\n\r")


@dataclass(frozen=True)
class Stage2SamplingConfig:
    max_span_width: int = 128
    positive_iou_threshold: float = 0.5
    negative_iou_threshold: float = 0.1
    positive_overlap_mode: str = "gold_coverage"
    negative_positive_ratio: int = 5
    max_negative_no_positive: int = 128
    boundary_perturb_radius: int = 0
    include_gold_candidates: bool = True
    sentence_context_sentences: int = 0
    seed: int = 42


def read_proposal_file(path: str | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    from src.common.utils import read_jsonl

    proposals = {}
    for row in read_jsonl(path):
        proposals[str(row["id"])] = list(row.get("proposals") or [])
    return proposals


class BILOUProposalWindowDataset(torch.utils.data.Dataset):
    """Sliding windows for binary propaganda span identification with BILOU tags."""

    def __init__(
        self,
        docs: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        stride: int,
        with_labels: bool = True,
    ) -> None:
        self.windows: list[dict[str, Any]] = []
        desc = "encoding BILOU proposal windows" if with_labels else "encoding BILOU proposal predict windows"
        for doc in tqdm(docs, desc=desc, dynamic_ncols=True, leave=False):
            self.windows.extend(encode_bilou_doc(tokenizer, doc, max_length, stride, with_labels=with_labels))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.windows[index]


def encode_bilou_doc(tokenizer, doc: dict[str, Any], max_length: int, stride: int, with_labels: bool) -> list[dict[str, Any]]:
    text = doc["text"]
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    windows = []
    for window_index in range(len(encoded["input_ids"])):
        offsets = [tuple(offset) for offset in encoded["offset_mapping"][window_index]]
        seq_ids = encoded.sequence_ids(window_index)
        valid = [sid == 0 and end > start for sid, (start, end) in zip(seq_ids, offsets)]
        item = {
            "doc_id": str(doc["id"]),
            "input_ids": encoded["input_ids"][window_index],
            "attention_mask": encoded["attention_mask"][window_index],
            "offsets": offsets,
            "valid": valid,
        }
        if with_labels:
            item["labels"] = make_bilou_labels(doc.get("spans", []), offsets, valid)
        windows.append(item)
    return windows


def has_different_boundary_overlap(spans: list[dict[str, Any]]) -> bool:
    intervals = [
        (int(span["start"]), int(span["end"]))
        for span in spans
        if int(span.get("start", 0)) < int(span.get("end", 0))
    ]
    intervals.sort()
    for index, (start, end) in enumerate(intervals):
        for other_start, other_end in intervals[index + 1 :]:
            if other_start >= end:
                break
            if (start, end) != (other_start, other_end):
                return True
    return False


def merge_location_spans(spans: list[dict[str, Any]]) -> list[tuple[int, int]]:
    intervals = sorted(
        (int(span["start"]), int(span["end"]))
        for span in spans
        if int(span.get("start", 0)) < int(span.get("end", 0))
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start >= merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def make_bilou_labels(spans: list[dict[str, Any]], offsets: list[tuple[int, int]], valid: list[bool]) -> list[int]:
    labels = [O_TAG if is_valid else IGNORE for is_valid in valid]
    for start, end in merge_location_spans(spans):
        overlapping = []
        for index, (tok_start, tok_end) in enumerate(offsets):
            if valid[index] and max(tok_start, start) < min(tok_end, end):
                overlapping.append(index)
        if not overlapping:
            continue
        desired: list[tuple[int, int]] = []
        if len(overlapping) == 1:
            desired.append((overlapping[0], U_TAG))
        else:
            for pos, token_index in enumerate(overlapping):
                tok_start, tok_end = offsets[token_index]
                touches_start = tok_start <= start < tok_end
                touches_end = tok_start < end <= tok_end
                if pos == 0 and touches_start:
                    desired.append((token_index, B_TAG))
                elif pos == len(overlapping) - 1 and touches_end:
                    desired.append((token_index, L_TAG))
                else:
                    desired.append((token_index, I_TAG))
        for token_index, tag in desired:
            if labels[token_index] != IGNORE:
                labels[token_index] = tag
    return labels


def bilou_collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad(values: list[int], value: int) -> list[int]:
        return list(values) + [value] * (max_len - len(values))

    output = {
        "input_ids": torch.tensor([pad(item["input_ids"], pad_token_id) for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in batch], dtype=torch.long),
        "valid_mask": torch.tensor(
            [pad([1 if value else 0 for value in item["valid"]], 0) for item in batch],
            dtype=torch.bool,
        ),
        "meta": [
            {"doc_id": item["doc_id"], "offsets": item["offsets"], "valid": item["valid"]}
            for item in batch
        ],
    }
    if "labels" in batch[0]:
        output["labels"] = torch.tensor([pad(item["labels"], IGNORE) for item in batch], dtype=torch.long)
    return output


def bilou_tags_to_spans(
    tags: list[int],
    scores: list[float],
    positions: list[int],
    offsets: list[tuple[int, int]],
    max_span_width: int,
    threshold: float,
) -> list[dict[str, Any]]:
    spans = []
    cur_start = None
    cur_end = None
    cur_scores: list[float] = []
    cur_width = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_scores, cur_width
        if cur_start is not None and cur_end is not None and cur_start < cur_end:
            score = sum(cur_scores) / max(1, len(cur_scores))
            if score >= threshold:
                spans.append({"start": cur_start, "end": cur_end, "score": score})
        cur_start = None
        cur_end = None
        cur_scores = []
        cur_width = 0

    for tag, score, pos in zip(tags, scores, positions):
        start, end = offsets[pos]
        if tag == O_TAG:
            flush()
        elif tag == U_TAG:
            flush()
            if score >= threshold:
                spans.append({"start": int(start), "end": int(end), "score": float(score)})
        elif tag == B_TAG:
            flush()
            cur_start = int(start)
            cur_end = int(end)
            cur_scores = [float(score)]
            cur_width = 1
        elif tag in {I_TAG, L_TAG}:
            if cur_start is None:
                cur_start = int(start)
                cur_scores = []
                cur_width = 0
            cur_end = int(end)
            cur_scores.append(float(score))
            cur_width += 1
            if tag == L_TAG or cur_width >= max_span_width:
                flush()
    flush()
    return spans


class ZenStage2CandidateDataset(torch.utils.data.Dataset):
    """ZenPropaganda-style selected span/context candidates with multi-hot labels."""

    def __init__(
        self,
        docs: list[dict[str, Any]],
        tokenizer,
        label_list: list[str],
        max_length: int,
        sampling: Stage2SamplingConfig,
        proposals_by_doc: dict[str, list[dict[str, Any]]] | None = None,
        with_labels: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_to_idx = {label: idx for idx, label in enumerate(label_list)}
        self.max_length = max_length
        self.sampling = sampling
        self.with_labels = with_labels
        self.records: list[dict[str, Any]] = []
        proposals_by_doc = proposals_by_doc or {}
        desc = "building Zen stage2 train items" if with_labels else "building Zen stage2 predict items"
        for doc_index, doc in tqdm(
            enumerate(docs),
            total=len(docs),
            desc=desc,
            dynamic_ncols=True,
            leave=False,
        ):
            rng = random.Random(sampling.seed + doc_index)
            self.records.extend(
                build_zen_stage2_records(
                    doc,
                    self.label_to_idx,
                    sampling,
                    proposals_by_doc.get(str(doc["id"]), []),
                    rng,
                    with_labels,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded = encode_zen_selected_span(
            self.tokenizer,
            record["text"],
            record["char_start"],
            record["char_end"],
            self.max_length,
            sentence_context_sentences=self.sampling.sentence_context_sentences,
        )
        if encoded is None:
            raise ValueError(
                f"Stage 2 candidate doc={record['doc_id']} "
                f"[{record['char_start']}, {record['char_end']}) could not be tokenized"
            )
        item = {
            "doc_id": record["doc_id"],
            "char_start": record["char_start"],
            "char_end": record["char_end"],
            **encoded,
        }
        if "labels" in record:
            item["labels"] = record["labels"]
        return item


def build_zen_stage2_records(
    doc: dict[str, Any],
    label_to_idx: dict[str, int],
    sampling: Stage2SamplingConfig,
    proposals: list[dict[str, Any]],
    rng: random.Random,
    with_labels: bool,
) -> list[dict[str, Any]]:
    gold = normalize_gold_spans(doc.get("spans", []), label_to_idx) if with_labels else {}
    candidates = build_candidate_spans(doc["text"], gold, proposals, sampling, with_labels)
    records = []
    positives = []
    negatives = []
    for start, end in candidates:
        if with_labels:
            labels, ambiguous = label_for_candidate(start, end, gold, len(label_to_idx), sampling)
            if ambiguous:
                continue
        else:
            labels = None
        if not doc["text"][start:end].strip():
            continue
        record = {"doc_id": str(doc["id"]), "text": doc["text"], "char_start": start, "char_end": end}
        if labels is not None:
            record["labels"] = labels
            if labels.sum().item() > 0:
                positives.append(record)
            else:
                negatives.append(record)
        else:
            records.append(record)
    if not with_labels:
        return records
    max_negatives = sampling.max_negative_no_positive
    if positives:
        max_negatives = max(0, len(positives) * sampling.negative_positive_ratio)
    if max_negatives <= 0:
        negatives = []
    elif len(negatives) > max_negatives:
        negatives = rng.sample(negatives, max_negatives)
    records = positives + negatives
    rng.shuffle(records)
    return records


def normalize_gold_spans(spans: list[dict[str, Any]], label_to_idx: dict[str, int]) -> dict[tuple[int, int], set[int]]:
    gold: dict[tuple[int, int], set[int]] = {}
    for span in spans:
        label_idx = label_to_idx.get(span.get("technique"))
        if label_idx is None:
            continue
        start = int(span["start"])
        end = int(span["end"])
        if start < end:
            gold.setdefault((start, end), set()).add(label_idx)
    return gold


def build_candidate_spans(
    text: str,
    gold: dict[tuple[int, int], set[int]],
    proposals: list[dict[str, Any]],
    sampling: Stage2SamplingConfig,
    with_labels: bool,
) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for proposal in proposals:
        start = int(proposal.get("start", -1))
        end = int(proposal.get("end", -1))
        if 0 <= start < end <= len(text):
            candidates.add((start, end))
    if with_labels and sampling.include_gold_candidates:
        candidates.update(gold)
    if with_labels and sampling.boundary_perturb_radius > 0:
        radius = int(sampling.boundary_perturb_radius)
        for gold_start, gold_end in gold:
            for start_delta in range(-radius, radius + 1):
                for end_delta in range(-radius, radius + 1):
                    start = gold_start + start_delta
                    end = gold_end + end_delta
                    if 0 <= start < end <= len(text):
                        candidates.add((start, end))
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def label_for_candidate(
    start: int,
    end: int,
    gold: dict[tuple[int, int], set[int]],
    num_labels: int,
    sampling: Stage2SamplingConfig,
) -> tuple[torch.Tensor, bool]:
    labels = torch.zeros(num_labels, dtype=torch.float32)
    best_overlap = 0.0
    for (gold_start, gold_end), label_ids in gold.items():
        overlap = candidate_gold_overlap_score(
            start,
            end,
            gold_start,
            gold_end,
            sampling.positive_overlap_mode,
        )
        best_overlap = max(best_overlap, overlap)
        if overlap >= sampling.positive_iou_threshold:
            for label_idx in label_ids:
                labels[label_idx] = 1.0
    if labels.sum().item() > 0:
        return labels, False
    if best_overlap < sampling.negative_iou_threshold:
        return labels, False
    return labels, True


def candidate_gold_overlap_score(
    start: int,
    end: int,
    gold_start: int,
    gold_end: int,
    mode: str,
) -> float:
    inter = max(0, min(end, gold_end) - max(start, gold_start))
    if inter <= 0:
        return 0.0
    if mode == "iou":
        return char_iou(start, end, gold_start, gold_end)
    if mode == "gold_coverage":
        return inter / max(1, gold_end - gold_start)
    if mode == "candidate_coverage":
        return inter / max(1, end - start)
    if mode == "either_coverage":
        return max(inter / max(1, gold_end - gold_start), inter / max(1, end - start))
    raise ValueError(f"Unsupported positive_overlap_mode={mode!r}")


def encode_zen_selected_span(
    tokenizer,
    text: str,
    start: int,
    end: int,
    max_length: int,
    sentence_context_sentences: int = 0,
) -> dict[str, Any] | None:
    span_text = text[start:end]
    if not span_text.strip():
        return None
    window_start, window_end = sentence_bounds_for_span(text, start, end, sentence_context_sentences)
    for char_budget in (None, max_length * 5, max_length * 3, max(end - start, 1)):
        ctx_start, ctx_end = window_start, window_end
        if char_budget is not None:
            extra = max(0, char_budget - (end - start))
            left_budget = extra // 2
            right_budget = extra - left_budget
            ctx_start = max(0, start - left_budget)
            ctx_end = min(len(text), end + right_budget)
        encoded = tokenizer(
            span_text,
            text[ctx_start:ctx_end],
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        seq_ids = encoded.sequence_ids()
        offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
        span_token_mask = [
            bool(sid == 0 and tok_end > tok_start)
            for sid, (tok_start, tok_end) in zip(seq_ids, offsets)
        ]
        if not any(span_token_mask):
            continue
        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "span_token_mask": span_token_mask,
        }
    return None


def sentence_bounds_for_span(text: str, start: int, end: int, context_sentences: int = 0) -> tuple[int, int]:
    left = start
    while left > 0 and text[left - 1] not in SENTENCE_BOUNDARY_CHARS:
        left -= 1
    right = end
    while right < len(text) and text[right] not in SENTENCE_BOUNDARY_CHARS:
        right += 1
    if right < len(text):
        right += 1
    for _ in range(max(0, context_sentences)):
        while left > 0 and text[left - 1] in SENTENCE_BOUNDARY_CHARS:
            left -= 1
        while left > 0 and text[left - 1] not in SENTENCE_BOUNDARY_CHARS:
            left -= 1
        while right < len(text) and text[right] in SENTENCE_BOUNDARY_CHARS:
            right += 1
        while right < len(text) and text[right] not in SENTENCE_BOUNDARY_CHARS:
            right += 1
        if right < len(text):
            right += 1
    return left, right


def zen_stage2_collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad(values: list[int], value: int) -> list[int]:
        return list(values) + [value] * (max_len - len(values))

    output = {
        "input_ids": torch.tensor([pad(item["input_ids"], pad_token_id) for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in batch], dtype=torch.long),
        "span_token_mask": torch.tensor(
            [pad([1 if value else 0 for value in item["span_token_mask"]], 0) for item in batch],
            dtype=torch.bool,
        ),
        "meta": [
            {"doc_id": item["doc_id"], "start": item["char_start"], "end": item["char_end"]}
            for item in batch
        ],
    }
    if "labels" in batch[0]:
        output["labels"] = torch.stack([item["labels"] for item in batch], dim=0)
    return output


Stage2CandidateDataset = ZenStage2CandidateDataset
stage2_collate = zen_stage2_collate
