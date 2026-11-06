"""Device helpers for CPU, CUDA, and Huawei Ascend NPU backends."""

from __future__ import annotations

import contextlib
import warnings
from typing import Iterator

import torch


ACCELERATOR_TYPES = {"cuda", "npu"}


class DummyGradScaler:
    """Small no-op GradScaler used when AMP scaling is unavailable."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def optimizer_step_was_run(scaler, optimizer) -> bool:
    """Run a scaled optimizer step and report whether it was actually applied.

    AMP scalers skip ``optimizer.step()`` on overflow. Scheduler/global step
    accounting should only advance when parameters were updated.
    """
    get_scale = getattr(scaler, "get_scale", None)
    old_scale = None
    if callable(get_scale):
        try:
            old_scale = float(get_scale())
        except (TypeError, ValueError):
            old_scale = None

    scaler.step(optimizer)
    scaler.update()

    if old_scale is None or not callable(get_scale):
        return True
    try:
        new_scale = float(get_scale())
    except (TypeError, ValueError):
        return True
    return new_scale >= old_scale


def import_torch_npu() -> bool:
    """Import torch_npu when installed so ``torch.npu`` is registered."""
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError:
        return False


def npu_is_available() -> bool:
    import_torch_npu()
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def detect_device(prefer: str = "auto") -> torch.device:
    """Resolve the requested runtime device.

    ``auto`` prefers NPU first, then CUDA, then CPU. This mirrors the server
    target for this copy of the project while keeping local smoke tests usable.
    """
    prefer = str(prefer or "auto").lower()
    if prefer == "auto":
        if npu_is_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if prefer == "npu":
        if not npu_is_available():
            raise RuntimeError("Requested --device npu, but torch_npu/Ascend NPU is not available.")
        return torch.device("npu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, npu, cuda, cpu")


def accelerator_name(device: torch.device) -> str | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device.index or 0)
    if device.type == "npu" and npu_is_available():
        try:
            return torch.npu.get_device_name(device.index or 0)
        except Exception:
            return "Ascend NPU"
    return None


def memory_allocated_gb(device: torch.device) -> float | None:
    try:
        if device.type == "cuda":
            return torch.cuda.memory_allocated(device.index or 0) / 1024**3
        if device.type == "npu" and npu_is_available():
            return torch.npu.memory_allocated(device.index or 0) / 1024**3
    except Exception:
        return None
    return None


def max_memory_allocated_mb(device: torch.device) -> float | None:
    try:
        if device.type == "cuda":
            return round(torch.cuda.max_memory_allocated(device.index or 0) / 1024**2, 2)
        if device.type == "npu" and npu_is_available():
            return round(torch.npu.max_memory_allocated(device.index or 0) / 1024**2, 2)
    except Exception:
        return None
    return None


def manual_seed_all(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if npu_is_available():
        torch.npu.manual_seed_all(seed)


def make_grad_scaler(device: torch.device, enabled: bool):
    enabled = bool(enabled) and device.type in ACCELERATOR_TYPES
    if not enabled:
        return DummyGradScaler()
    if device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=True)
    if device.type == "npu":
        import_torch_npu()
        try:
            return torch.npu.amp.GradScaler(enabled=True)
        except Exception:
            warnings.warn("NPU GradScaler is unavailable; continuing with no-op loss scaling.", RuntimeWarning)
            return DummyGradScaler()
    return DummyGradScaler()


def autocast_for(device: torch.device, enabled: bool) -> Iterator[None]:
    enabled = bool(enabled) and device.type in ACCELERATOR_TYPES
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "npu":
        import_torch_npu()
        try:
            return torch.npu.amp.autocast(enabled=True)
        except Exception:
            pass
    try:
        return torch.autocast(device_type=device.type, enabled=True)
    except (AttributeError, TypeError):
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=True)
        if device.type == "npu":
            warnings.warn("NPU autocast is unavailable; running this step without autocast.", RuntimeWarning)
            return contextlib.nullcontext()
    return contextlib.nullcontext()
