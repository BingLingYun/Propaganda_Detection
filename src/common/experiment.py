"""Shared experiment helpers for neural baseline scripts."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

from src.common.utils import read_json, write_json


def select_model_name(cfg: dict[str, Any], ds_cfg: dict[str, Any], explicit_model_name: str | None = None) -> str:
    """Resolve the encoder name, including optional language-specific overrides."""
    if explicit_model_name:
        return explicit_model_name
    language = ds_cfg.get("language")
    overrides = cfg.get("language_model_overrides") or {}
    if language and language in overrides:
        return str(overrides[language])
    return str(cfg["model_name"])


def make_neural_run_name(dataset_name: str, cfg: dict[str, Any]) -> str:
    method_names = {
        "mhbio_crf": "mhbio_crf",
        "ts_pipeline": "bilou_select",
        "ts_proposal": "bilou_select_proposal",
        "ts_classifier": "bilou_select_classifier",
    }
    model_short = str(cfg["model_name"]).rstrip("/").split("/")[-1]
    method_name = method_names.get(str(cfg["model_type"]), str(cfg["model_type"]))
    return f"{dataset_name}_{method_name}_{model_short}_seed{cfg['seed']}"


def checkpoint_threshold(checkpoint: Path, default: float = 0.5, key: str = "threshold") -> float:
    """Read tuned threshold from ``thresholds.json`` if present, otherwise meta/default."""
    thresholds_path = checkpoint / "thresholds.json"
    if thresholds_path.exists():
        data = read_json(thresholds_path)
        if key in data:
            return float(data[key])
        if "threshold" in data:
            return float(data["threshold"])
    meta_path = checkpoint / "meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        return float(meta.get(key, meta.get("threshold", default)))
    return float(default)


def update_checkpoint_threshold(checkpoint: Path, threshold: float, payload: dict[str, Any]) -> None:
    """Persist tuned threshold and mirror it into ``meta.json`` for older scripts."""
    checkpoint.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint / "thresholds.json", {"threshold": float(threshold), **payload})
    meta_path = checkpoint / "meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        meta["threshold"] = float(threshold)
        meta["threshold_source"] = "dev_tuning"
        write_json(meta_path, meta)


def parameter_count_from_state_dict(checkpoint: Path) -> int | None:
    model_path = checkpoint / "model.pt"
    if not model_path.exists():
        return None
    try:
        import torch

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        return int(sum(t.numel() for t in state.values() if hasattr(t, "numel")))
    except Exception:
        return None


def runtime_snapshot(
    *,
    run_name: str,
    cfg: dict[str, Any],
    ds_cfg: dict[str, Any],
    start_time: float,
    checkpoint: Path,
    device: Any = None,
    status: str = "complete",
) -> dict[str, Any]:
    elapsed = time.time() - start_time
    device_type = "cpu"
    accelerator = None
    peak_accelerator_memory_mb = None
    peak_cuda_memory_mb = None
    try:
        from src.common.device import accelerator_name, detect_device, max_memory_allocated_mb

        runtime_device = device if device is not None else detect_device("auto")
        if not hasattr(runtime_device, "type"):
            import torch

            runtime_device = torch.device(str(runtime_device))
        runtime_type = getattr(runtime_device, "type", str(runtime_device).split(":")[0])
        device_type = runtime_type
        accelerator = accelerator_name(runtime_device)
        peak_accelerator_memory_mb = max_memory_allocated_mb(runtime_device)
        if runtime_type == "cuda":
            peak_cuda_memory_mb = peak_accelerator_memory_mb
    except Exception:
        pass
    return {
        "run_name": run_name,
        "status": status,
        "dataset": ds_cfg.get("name"),
        "language": ds_cfg.get("language"),
        "model_type": cfg.get("model_type"),
        "model_name": cfg.get("model_name"),
        "seed": cfg.get("seed"),
        "device": device_type,
        "accelerator_name": accelerator,
        "gpu_name": accelerator if device_type == "cuda" else None,
        "npu_name": accelerator if device_type == "npu" else None,
        "peak_accelerator_memory_mb": peak_accelerator_memory_mb,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "parameter_count": parameter_count_from_state_dict(checkpoint),
        "elapsed_seconds": round(elapsed, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
