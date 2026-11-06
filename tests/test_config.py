"""Configuration precedence and two-stage selection without model dependencies."""

import argparse
import unittest
from pathlib import Path

from src.common.experiment import make_neural_run_name
from src.common.utils import load_yaml
from src.training.config import merge_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def resolve(self, model="xlm_mhbio_crf", **overrides):
        values = dict(
            train_config=ROOT / "configs/train_default.yaml",
            model_config=ROOT / f"configs/models/{model}.yaml",
            model_type=None, model_name=None, seed=None, run_name=None,
        )
        values.update(overrides)
        dataset = load_yaml(ROOT / "configs/examples/custom_dataset.yaml")
        return merge_config(argparse.Namespace(**values), dataset)

    def test_command_line_overrides_defaults_and_aliases(self):
        cfg = self.resolve(seed=7, model_name="local/encoder", grad_accumulation=3, patience=9, fp16=False)
        self.assertEqual(cfg["grad_accumulation"], 3)
        self.assertEqual(cfg["early_stopping_patience"], 9)
        self.assertFalse(cfg["fp16"])
        self.assertEqual(make_neural_run_name("custom", cfg), "custom_mhbio_crf_encoder_seed7")

    def test_pipeline_selects_stage_specific_settings(self):
        proposal = self.resolve("xlm_bilou_select")
        classifier = self.resolve("xlm_bilou_select", model_type="ts_classifier")
        self.assertEqual(proposal["model_type"], "ts_proposal")
        self.assertEqual(classifier["model_type"], "ts_classifier")
        self.assertEqual(classifier["positive_overlap_mode"], "gold_coverage")
        self.assertEqual(proposal["dataset"], classifier["dataset"])
        self.assertIn("bilou_select_classifier", make_neural_run_name("custom", classifier))


if __name__ == "__main__":
    unittest.main()
