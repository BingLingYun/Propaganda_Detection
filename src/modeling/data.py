"""模块二:数据编码(BIO 标签生成 + 滑动窗口)与解码(BIO -> 字符片段)。

关键约定:
    - 标签 id:O=0, B=1, I=2(multi-head 每头独立);
      single_label 模式:O=0, B_c=1+2c, I_c=2+2c(c 为类别下标)
    - 使用 tokenizer 的 return_offsets_mapping,offset 始终相对传入的原始文本,
      因此滑动窗口(return_overflowing_tokens)各窗口 offset 已是原文字符坐标,
      window_char_offset 恒为 0(代码中仍保留该字段以明确语义)。
    - 空 offset (tok_start == tok_end) 的 token(特殊 token 或词表外字符)一律跳过,
      标签置 -100。
    - 禁止任何改变字符数的文本预处理。
"""
from __future__ import annotations

import torch
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    def tqdm(iterable, *args, **kwargs):
        return iterable

O_TAG, B_TAG, I_TAG = 0, 1, 2
IGNORE = -100


# ============================================================ 编码

def encode_doc(tokenizer, doc: dict, class_list: list[str], max_length: int, stride: int,
               single_label: bool, with_labels: bool = True,
               deduplicate_training_tokens: bool = True) -> list[dict]:
    """把一个文档编码为若干窗口。

    返回每窗口 dict:
        input_ids, attention_mask: list[int]
        labels: (H, L) 列表(with_labels 时);multi-head H=类别数,single H=1
        doc_id, window_char_offset, offsets: [(s,e)], valid: [bool]
    """
    text = doc["text"]
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    cls2idx = {c: i for i, c in enumerate(class_list)}
    windows = []
    seen_training_offsets: set[tuple[int, int]] = set()
    for w in range(len(enc["input_ids"])):
        offsets = enc["offset_mapping"][w]
        seq_ids = enc.sequence_ids(w)
        # 有效 token:属于正文(sequence_id==0)且 offset 非空
        token_valid = [sid == 0 and off[1] > off[0] for sid, off in zip(seq_ids, offsets)]
        if with_labels and deduplicate_training_tokens:
            valid = []
            for is_valid, off in zip(token_valid, offsets):
                offset = tuple(off)
                keep = bool(is_valid and offset not in seen_training_offsets)
                valid.append(keep)
            for is_valid, off in zip(token_valid, offsets):
                if is_valid:
                    seen_training_offsets.add(tuple(off))
        else:
            valid = token_valid
        win = {
            "doc_id": doc["id"],
            "input_ids": enc["input_ids"][w],
            "attention_mask": enc["attention_mask"][w],
            "offsets": [tuple(o) for o in offsets],
            "valid": valid,
            # HF fast tokenizer 溢出窗口的 offset 已相对原文,故偏移为 0(见模块 docstring)
            "window_char_offset": 0,
        }
        if with_labels:
            win["labels"] = _make_bio_labels(
                doc.get("spans", []), offsets, valid, cls2idx, single_label)
        windows.append(win)
    return windows


def count_doc_windows(tokenizer, text: str, max_length: int, stride: int) -> int:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
    )
    return len(enc["input_ids"])


def _make_bio_labels(spans, offsets, valid, cls2idx, single_label):
    """按 4.2.1 规则生成 BIO 标签。

    对每个类别头:token 与该类别任一 gold 片段字符区间相交则标 I;
    其中「片段起点字符所在的 token」标 B。片段起点在窗口外(跨窗口片段)时
    窗口内全为 I,由解码端的孤儿 I 规则兜底。
    """
    L = len(offsets)
    n_heads = 1 if single_label else len(cls2idx)
    labels = [[IGNORE] * L for _ in range(n_heads)]
    for h in range(n_heads):
        for j in range(L):
            if valid[j]:
                labels[h][j] = O_TAG

    for span in spans:
        c = cls2idx[span["technique"]]
        h = 0 if single_label else c
        b_val = 1 + 2 * c if single_label else B_TAG
        i_val = 2 + 2 * c if single_label else I_TAG
        first_overlap = True
        for j in range(L):
            if not valid[j]:
                continue
            ts, te = offsets[j]
            if max(ts, span["start"]) < min(te, span["end"]):     # 相交
                # Keep each window-local BIO2 sequence legal under the constrained CRF.
                if first_overlap or ts <= span["start"] < te:
                    labels[h][j] = b_val
                    first_overlap = False
                else:
                    labels[h][j] = i_val
    return labels


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """padding 到 batch 内最大长度;labels 以 -100 填充。"""
    L = max(len(x["input_ids"]) for x in batch)
    H = len(batch[0]["labels"]) if "labels" in batch[0] else 0

    def pad(seq, value, length=L):
        return list(seq) + [value] * (length - len(seq))

    out = {
        "input_ids": torch.tensor([pad(x["input_ids"], pad_token_id) for x in batch]),
        "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
        "valid_mask": torch.tensor([pad([1 if v else 0 for v in x["valid"]], 0) for x in batch],
                                   dtype=torch.bool),
    }
    if H:
        out["labels"] = torch.tensor(
            [[pad(head, IGNORE) for head in x["labels"]] for x in batch])   # (B, H, L)
    out["meta"] = [{"doc_id": x["doc_id"], "offsets": x["offsets"], "valid": x["valid"],
                    "window_char_offset": x["window_char_offset"]} for x in batch]
    return out


class WindowDataset(torch.utils.data.Dataset):
    """Document windows, either eagerly encoded or lazily materialized per document."""

    def __init__(self, docs, tokenizer, class_list, max_length, stride,
                 single_label, with_labels=True, deduplicate_training_tokens=True,
                 lazy=False, cache_docs=2):
        self.docs = list(docs)
        self.tokenizer = tokenizer
        self.class_list = class_list
        self.max_length = max_length
        self.stride = stride
        self.single_label = single_label
        self.with_labels = with_labels
        self.deduplicate_training_tokens = deduplicate_training_tokens
        self.lazy = bool(lazy)
        self.cache_docs = max(0, int(cache_docs))
        self._cache: dict[int, list[dict]] = {}
        self.windows = []
        self.window_index: list[tuple[int, int]] = []
        if self.lazy:
            desc = "indexing labeled windows" if with_labels else "indexing prediction windows"
            for doc_index, doc in tqdm(enumerate(self.docs), total=len(self.docs), desc=desc, dynamic_ncols=True, leave=False):
                n_windows = count_doc_windows(tokenizer, doc["text"], max_length, stride)
                self.window_index.extend((doc_index, window_index) for window_index in range(n_windows))
        else:
            desc = "encoding labeled windows" if with_labels else "encoding prediction windows"
            for doc in tqdm(self.docs, desc=desc, dynamic_ncols=True, leave=False):
                self.windows += encode_doc(tokenizer, doc, class_list, max_length, stride,
                                           single_label, with_labels, deduplicate_training_tokens)

    def __len__(self):
        return len(self.window_index) if self.lazy else len(self.windows)

    def __getitem__(self, i):
        if not self.lazy:
            return self.windows[i]
        doc_index, window_index = self.window_index[i]
        windows = self._windows_for_doc(doc_index)
        return windows[window_index]

    def _windows_for_doc(self, doc_index: int) -> list[dict]:
        cached = self._cache.get(doc_index)
        if cached is not None:
            return cached
        windows = encode_doc(
            self.tokenizer,
            self.docs[doc_index],
            self.class_list,
            self.max_length,
            self.stride,
            self.single_label,
            self.with_labels,
            self.deduplicate_training_tokens,
        )
        if self.cache_docs > 0:
            self._cache[doc_index] = windows
            while len(self._cache) > self.cache_docs:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
        return windows


# ============================================================ 解码

def tags_to_spans(tags: list[int], positions: list[int], offsets: list[tuple],
                  n_classes: int, single_label: bool, head_idx: int) -> list[dict]:
    """一个头在有效 token 位置上的标签序列 -> 窗口内字符片段(类别下标形式)。

    tags 与 positions 等长且一一对应(仅有效 token)。
    孤儿 I(前面无 B)按新片段开始处理。
    """
    spans = []
    cur = None          # [class_idx, char_start, char_end]

    def tag_kind(t):
        """返回 (kind, class_idx);kind ∈ {'O','B','I'}"""
        if single_label:
            if t == 0:
                return "O", -1
            c = (t - 1) // 2
            return ("B" if (t - 1) % 2 == 0 else "I"), c
        return ("O", -1) if t == O_TAG else (("B", head_idx) if t == B_TAG else ("I", head_idx))

    for t, pos in zip(tags, positions):
        kind, c = tag_kind(t)
        s_char, e_char = offsets[pos]
        if kind == "B":
            if cur:
                spans.append(cur)
            cur = [c, s_char, e_char]
        elif kind == "I":
            if cur and cur[0] == c:
                cur[2] = e_char
            else:                    # 孤儿 I / 类别切换:视为新片段
                if cur:
                    spans.append(cur)
                cur = [c, s_char, e_char]
        else:
            if cur:
                spans.append(cur)
                cur = None
    if cur:
        spans.append(cur)
    return [{"class_idx": c, "start": s, "end": e} for c, s, e in spans]


def merge_window_spans(all_spans: list[dict], class_list: list[str], gap: int = 0) -> list[dict]:
    """合并一个文档所有窗口解码出的片段(已映射回原文坐标):
    同类别片段相交或相邻(间隔 <= gap)则合并。返回标准 span 列表,按 start 排序。"""
    from src.common.utils import merge_same_technique_spans, sort_spans
    named = [{"start": s["start"], "end": s["end"], "technique": class_list[s["class_idx"]]}
             for s in all_spans]
    if not named:
        return []
    merged, _ = merge_same_technique_spans(named, gap=gap)
    return sort_spans(merged)
