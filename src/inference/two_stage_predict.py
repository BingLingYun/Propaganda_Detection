"""End-to-end two-stage prediction helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.common.utils import sort_spans, validate_spans
from src.inference.span_predict import predict_proposal_docs
from src.models.span_utils import merge_overlapping_same_label, nms_same_label
from src.training.stage2_data import Stage2SamplingConfig, ZenStage2CandidateDataset, zen_stage2_collate


def _iter_candidate_doc_chunks(
    docs: list[dict[str, Any]],
    proposals_by_doc: dict[str, list[dict[str, Any]]],
    max_candidates: int,
):
    max_candidates = max(1, int(max_candidates))
    chunk: list[dict[str, Any]] = []
    count = 0
    for doc in docs:
        n_doc = max(1, len(proposals_by_doc.get(str(doc["id"]), [])))
        if chunk and count + n_doc > max_candidates:
            yield chunk
            chunk = []
            count = 0
        chunk.append(doc)
        count += n_doc
    if chunk:
        yield chunk


@torch.no_grad()
def classify_proposals(
    classifier,
    tokenizer,
    docs: list[dict[str, Any]],
    proposals_by_doc: dict[str, list[dict[str, Any]]],
    label_list: list[str],
    max_length: int,
    threshold: float,
    batch_size: int,
    device,
    nms_iou: float = 0.5,
    sentence_context_sentences: int = 0,
    candidate_chunk_size: int = 32768,
) -> list[dict[str, Any]]:
    classifier.eval()
    doc_spans: dict[str, list[dict[str, Any]]] = {str(doc["id"]): [] for doc in docs}
    doc_chunks = list(_iter_candidate_doc_chunks(docs, proposals_by_doc, candidate_chunk_size))
    for doc_chunk in tqdm(doc_chunks, total=len(doc_chunks), desc="stage2 chunks", dynamic_ncols=True, leave=False):
        dataset = ZenStage2CandidateDataset(
            doc_chunk,
            tokenizer,
            label_list,
            max_length,
            Stage2SamplingConfig(sentence_context_sentences=sentence_context_sentences),
            proposals_by_doc=proposals_by_doc,
            with_labels=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: zen_stage2_collate(batch, tokenizer.pad_token_id),
        )
        for batch in tqdm(loader, total=len(loader), desc="stage2 classify", dynamic_ncols=True, leave=False):
            output = classifier(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                span_token_mask=batch["span_token_mask"].to(device),
            )
            probs = torch.sigmoid(output["logits"]).detach().cpu()
            for row_index, meta in enumerate(batch["meta"]):
                for label_idx, technique in enumerate(label_list):
                    score = float(probs[row_index, label_idx].item())
                    if score >= threshold:
                        doc_spans[meta["doc_id"]].append(
                            {
                                "start": meta["start"],
                                "end": meta["end"],
                                "technique": technique,
                                "score": score,
                            }
                        )

    predictions = []
    for doc in docs:
        spans = nms_same_label(doc_spans[str(doc["id"])], iou_threshold=nms_iou)
        spans = merge_overlapping_same_label(spans)
        clean = [
            {"start": span["start"], "end": span["end"], "technique": span["technique"]}
            for span in sort_spans(spans)
        ]
        validate_spans(clean, len(doc["text"]), str(doc["id"]))
        predictions.append({"id": doc["id"], "spans": clean})
    return predictions


@torch.no_grad()
def predict_two_stage_pipeline(
    proposal_model,
    classifier,
    proposal_tokenizer,
    classifier_tokenizer,
    docs: list[dict[str, Any]],
    label_list: list[str],
    max_length: int,
    stride: int,
    max_span_width: int,
    proposal_threshold: float,
    classifier_threshold: float,
    batch_size: int,
    device,
    candidate_chunk_size: int = 32768,
    max_proposals_per_document: int = 200,
    nms_iou: float = 0.5,
    sentence_context_sentences: int = 0,
) -> list[dict[str, Any]]:
    proposal_rows = predict_proposal_docs(
        proposal_model,
        proposal_tokenizer,
        docs,
        max_length,
        stride,
        max_span_width,
        proposal_threshold,
        batch_size,
        device,
        candidate_chunk_size=candidate_chunk_size,
        max_proposals_per_document=max_proposals_per_document,
    )
    proposals_by_doc = {str(row["id"]): row.get("proposals", []) for row in proposal_rows}
    return classify_proposals(
        classifier,
        classifier_tokenizer,
        docs,
        proposals_by_doc,
        label_list,
        max_length,
        classifier_threshold,
        batch_size,
        device,
        nms_iou=nms_iou,
        sentence_context_sentences=sentence_context_sentences,
        candidate_chunk_size=candidate_chunk_size,
    )
