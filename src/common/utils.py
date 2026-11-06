"""全项目公共工具:JSONL 读写、种子、片段操作、配置加载。

约定:全项目内 span 均为字典 {"start": int, "end": int, "technique": str},
字符级、0 起、左闭右开 [start, end)。
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------- I/O

def read_jsonl(path: str | Path) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip("\n")
            if not line.strip():
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} 不是合法 JSON: {e}") from e
    return docs


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False)
            line = line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            f.write(line + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_scalar(value: str):
    value = value.strip()
    if value in ("", "null", "None"):
        return None
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(part.strip()) for part in body.split(",")]
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1].strip()
        out = {}
        if body:
            for part in body.split(","):
                k, v = part.split(":", 1)
                out[k.strip()] = _parse_scalar(v.strip())
        return out
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _strip_yaml_comment(line: str) -> str:
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _load_simple_yaml(path: str | Path) -> dict:
    """Small fallback for the simple project configs when PyYAML is unavailable."""
    root: dict = {}
    stack: list[tuple[int, dict | list]] = [(-1, root)]
    pending: list[tuple[int, dict, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _strip_yaml_comment(raw).rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            text = line.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            while pending and indent <= pending[-1][0]:
                pending.pop()

            parent = stack[-1][1]
            if text.startswith("- "):
                value = _parse_scalar(text[2:].strip())
                if not isinstance(parent, list):
                    if not pending:
                        raise ValueError(f"{path}: list item without list parent: {raw}")
                    p_indent, p_dict, p_key = pending[-1]
                    new_list: list = []
                    p_dict[p_key] = new_list
                    stack.append((p_indent, new_list))
                    parent = new_list
                parent.append(value)
                continue

            if ":" not in text:
                raise ValueError(f"{path}: invalid yaml line: {raw}")
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not isinstance(parent, dict):
                raise ValueError(f"{path}: key/value under list is not supported: {raw}")
            if value:
                parent[key] = _parse_scalar(value)
            else:
                child: dict = {}
                parent[key] = child
                pending.append((indent, parent, key))
                stack.append((indent, child))
    return root


def get_labels(cfg: dict) -> list[str]:
    """Return dataset labels from inline config or a generated label file."""
    labels = cfg.get("labels")
    if labels:
        return list(labels)
    label_file = cfg.get("label_file")
    if label_file:
        data = read_json(label_file)
        if isinstance(data, dict):
            data = data.get("labels")
        if data:
            return list(data)
    raise KeyError("dataset config must define either labels or label_file")


# ---------------------------------------------------------------- 随机种子

def set_seed(seed: int, torch_too: bool = True) -> None:
    """固定 python / numpy / torch / cudnn 随机源。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if torch_too:
        try:
            import torch
            from src.common.device import manual_seed_all

            torch.manual_seed(seed)
            manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass


# ---------------------------------------------------------------- 片段操作

def span_len(s: dict) -> int:
    return s["end"] - s["start"]


def spans_overlap(a: dict, b: dict) -> bool:
    """区间是否相交(左闭右开,共享端点不算相交)。"""
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def overlap_len(a: dict, b: dict) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def sort_spans(spans: list[dict]) -> list[dict]:
    return sorted(spans, key=lambda s: (s["start"], s["end"], s.get("technique", "")))


def merge_same_technique_spans(spans: list[dict], gap: int = 0) -> tuple[list[dict], int]:
    """将同类别、相交或间隔 <= gap 的片段合并。

    返回 (合并后的片段列表, 被合并掉的片段数)。跨类别片段不受影响。
    """
    by_tech: dict[str, list[dict]] = {}
    for s in spans:
        by_tech.setdefault(s["technique"], []).append(s)
    merged_all: list[dict] = []
    n_merged = 0
    for tech, group in by_tech.items():
        group = sorted(group, key=lambda s: (s["start"], s["end"]))
        cur = dict(group[0])
        for s in group[1:]:
            if s["start"] - cur["end"] <= gap:  # 相交或相邻
                cur["end"] = max(cur["end"], s["end"])
                n_merged += 1
            else:
                merged_all.append(cur)
                cur = dict(s)
        merged_all.append(cur)
    return sort_spans(merged_all), n_merged


def validate_spans(spans: list[dict], text_len: int, doc_id: str = "?") -> None:
    """断言所有片段满足 0 <= start < end <= len(text),否则抛 AssertionError。"""
    for s in spans:
        assert 0 <= s["start"] < s["end"] <= text_len, (
            f"文档 {doc_id} 片段越界: [{s['start']}, {s['end']}), 文本长度 {text_len}"
        )


# ---------------------------------------------------------------- 其它

def quantiles(values: list[float], qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)) -> dict:
    """纯 python 分位数(线性插值),避免对 numpy 的硬依赖。"""
    if not values:
        return {str(q): None for q in qs}
    xs = sorted(values)
    out = {}
    for q in qs:
        pos = q * (len(xs) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
        frac = pos - lo
        out[str(q)] = xs[lo] * (1 - frac) + xs[hi] * frac
    return out


def make_run_name(dataset: str, model_name: str, use_crf: bool, single_label: bool, seed: int) -> str:
    model_short = model_name.rstrip("/").split("/")[-1]
    parts = [dataset, model_short, "crf" if use_crf else "nocrf"]
    if single_label:
        parts.append("single")
    parts.append(f"seed{seed}")
    return "_".join(parts)
