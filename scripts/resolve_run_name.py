"""Resolve neural run names exactly like ``scripts/train.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.experiment import make_neural_run_name
from src.common.utils import load_yaml
from src.training.config import merge_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_config", default="configs/train_default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds_cfg = load_yaml(args.dataset_config)
    cfg = merge_config(args, ds_cfg)
    print(args.run_name or make_neural_run_name(ds_cfg["name"], cfg))


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
