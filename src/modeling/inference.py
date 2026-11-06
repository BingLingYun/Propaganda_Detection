"""推理核心:文档 -> 窗口批量解码 -> 片段合并回原文坐标。

被 train.py(epoch 间 Dev 评测)与 predict.py(正式推理)共用。
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.common.utils import sort_spans, validate_spans
from src.modeling.data import WindowDataset, collate, tags_to_spans, merge_window_spans


@torch.no_grad()
def predict_docs(model, tokenizer, docs: list[dict], class_list: list[str],
                 max_length: int, stride: int, merge_gap: int,
                 batch_size: int, device, single_label: bool) -> list[dict]:
    """对文档列表推理,返回标准预测记录 [{id, text?, spans}]。

    步骤:滑动窗口编码 -> 各窗口独立解码(窗口内即原文字符坐标,
    见 data.py 说明)-> 加 window_char_offset 映射回原文 -> 同类别相交/相邻合并
    -> 排序与合法性断言。
    """
    model.eval()
    ds = WindowDataset(docs, tokenizer, class_list, max_length, stride,
                       single_label, with_labels=False, lazy=True, cache_docs=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    doc_spans: dict[str, list] = {d["id"]: [] for d in docs}
    for batch in tqdm(loader, total=len(loader), desc="BIO-CRF predict", dynamic_ncols=True, leave=False):
        tags = model.decode(batch["input_ids"].to(device),
                            batch["attention_mask"].to(device),
                            batch["valid_mask"].to(device))
        for b, meta in enumerate(batch["meta"]):
            positions = [j for j, v in enumerate(meta["valid"]) if v]
            for h in range(model.num_heads):
                for s in tags_to_spans(tags[b][h], positions, meta["offsets"],
                                       len(class_list), single_label, h):
                    # 映射回原文字符坐标(HF 溢出窗口 offset 已相对原文,偏移为 0)
                    s["start"] += meta["window_char_offset"]
                    s["end"] += meta["window_char_offset"]
                    doc_spans[meta["doc_id"]].append(s)

    preds = []
    for d in docs:
        spans = merge_window_spans(doc_spans[d["id"]], class_list, gap=merge_gap)
        spans = sort_spans(spans)
        validate_spans(spans, len(d["text"]), d["id"])   # 0 <= start < end <= len(text)
        preds.append({"id": d["id"], "spans": spans})
    return preds
