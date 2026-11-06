"""Regression checks for local span scoring and evaluation reports."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation import evaluate as local
from src.evaluation import span_metrics

ROOT = Path(__file__).resolve().parents[1]


def span(start, end, label="A"):
    return {"start": start, "end": end, "technique": label}


def reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def strict_read(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


class CoverageTests(unittest.TestCase):

    def test_overlapping_predictions_do_not_inflate_recall(self):
        result = local.partial_match_prf({'d': [span(0, 10)]}, {'d': [span(0, 10), span(0, 9)]})
        self.assertEqual(result, {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'n_pred': 2, 'n_gold': 1})

    def test_overlapping_gold_does_not_inflate_precision(self):
        result = local.partial_match_prf({'d': [span(0, 10), span(0, 9)]}, {'d': [span(0, 10)]})
        self.assertEqual((result['precision'], result['recall'], result['f1']), (1, 1, 1))

    def test_union_keeps_gaps_and_partial_credit(self):
        for preds, expected_recall in [([span(0, 6), span(4, 8)], 0.8), ([span(0, 2), span(8, 10)], 0.4)]:
            result = local.partial_match_prf({'d': [span(0, 10)]}, {'d': preds})
            self.assertEqual(result['precision'], 1)
            self.assertAlmostEqual(result['recall'], expected_recall)
            self.assertAlmostEqual(result['f1'], 2 * expected_recall / (1 + expected_recall))

    def test_labels_and_documents_stay_independent(self):
        self.assertEqual(local.partial_match_prf({'a': [span(0, 4)]}, {'b': [span(0, 4)]})['f1'], 0)
        gold = {'a': [span(0, 4, 'A'), span(0, 4, 'B')]}
        pred = {'a': [span(0, 4, 'B')]}
        result = local.partial_match_prf(gold, pred)
        self.assertEqual(result['precision'], 1)
        self.assertEqual(result['recall'], 0.5)
        self.assertEqual(local.partial_match_prf(gold, pred, use_class=False)['f1'], 1)

    def test_normalization_preserves_multilabel_and_removes_exact_duplicates(self):
        gold = [{'id': 'd', 'text': 'abcd', 'spans': [span(0, 4, 'A'), span(0, 4, 'B')]}]
        pred = [{'id': 'd', 'spans': [span(0, 4, 'A'), span(0, 4, 'A'), span(0, 4, 'B')]}]
        report = local.evaluate(gold, pred, labels=['A', 'B'])
        self.assertEqual(report['micro']['f1'], 1)
        self.assertEqual(report['counts']['n_pred_spans'], 2)
        self.assertEqual(report['scoring']['version'], 'span-coverage-v2')

    def test_random_intervals_against_character_sets(self):
        rng = random.Random(42)
        for _ in range(200):
            gold, pred = ([], [])
            for spans in (gold, pred):
                for _ in range(rng.randrange(8)):
                    start = rng.randrange(20)
                    spans.append(span(start, rng.randrange(start + 1, 22), rng.choice('AB')))
            for use_class in (True, False):

                def average_coverage(rows, refs):
                    credits = []
                    for row in rows:
                        chars = set(range(row['start'], row['end']))
                        covered = set()
                        for ref in refs:
                            if not use_class or ref['technique'] == row['technique']:
                                covered.update(range(ref['start'], ref['end']))
                        credits.append(len(chars & covered) / len(chars))
                    return sum(credits) / len(rows) if rows else 0
                expected_p = average_coverage(pred, gold)
                expected_r = average_coverage(gold, pred)
                result = local.partial_match_prf({'d': gold}, {'d': pred}, use_class)
                self.assertAlmostEqual(result['precision'], expected_p)
                self.assertAlmostEqual(result['recall'], expected_r)
                for key in ('precision', 'recall', 'f1'):
                    self.assertTrue(0 <= result[key] <= 1, result)


class GRRTests(unittest.TestCase):

    def test_valid_zero_and_greater_than_one(self):
        result = local.compute_grr({'a': {'f1': 0.8}, 'b': {'f1': 0.4}}, 0.2)
        self.assertAlmostEqual(result['GRR'], 3)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(local.compute_grr({'a': {'f1': 0}}, 0.5)['GRR'], 0)

    def test_undefined_reasons_never_use_nan_or_zero(self):
        cases = [(None, {'a': {'f1': 1}}, 'missing_id_eval'), (0, {'a': {'f1': 1}}, 'zero_id_score'), (-0.1, {'a': {'f1': 1}}, 'invalid_id_score'), (1.1, {'a': {'f1': 1}}, 'invalid_id_score'), (float('nan'), {'a': {'f1': 1}}, 'invalid_id_score'), (float('inf'), {'a': {'f1': 1}}, 'invalid_id_score'), (True, {'a': {'f1': 1}}, 'invalid_id_score'), (10 ** 400, {'a': {'f1': 1}}, 'invalid_id_score'), (0.5, {}, 'missing_groups'), (0.5, {'a': {'f1': float('inf')}}, 'invalid_group_score'), (0.5, {'a': {'f1': 1.2}}, 'invalid_group_score'), (1e-320, {'a': {'f1': 1}}, 'non_finite_ratio')]
        for id_f1, groups, status in cases:
            with self.subTest(evaluator=local.__name__, status=status):
                result = local.compute_grr(groups, id_f1)
                self.assertIsNone(result['GRR'])
                self.assertEqual(result['status'], status)
                self.assertTrue(result['warning'])
                json.dumps(result, allow_nan=False)


class EvaluationCLITests(unittest.TestCase):

    def test_csv_group_arrays_preserve_separate_target_groups(self):
        groups = {'d': '["target_a", "target_b"]'}
        gold = [{'id': 'd', 'text': 'abcd', 'spans': [span(0, 4)]}]
        pred = [{'id': 'd', 'spans': [span(0, 4)]}]
        report = local.evaluate(gold, pred, groups=groups, id_eval_micro_f1=0.5, labels=['A'])
        self.assertEqual(set(report['groups']['per_group']), {'target_a', 'target_b'})
        self.assertEqual(report['GRR']['GRR'], 2)

    def test_local_cli_serializes_invalid_id_as_null(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            gold = base / 'gold.jsonl'
            pred = base / 'pred.jsonl'
            report = base / 'report.json'
            gold.write_text(json.dumps({'id': 'd', 'text': 'abcd', 'spans': [span(0, 4)]}) + '\n', encoding='utf-8')
            pred.write_text(json.dumps({'id': 'd', 'spans': [span(0, 4)]}) + '\n', encoding='utf-8')
            run = subprocess.run([sys.executable, '-m', 'src.evaluation.evaluate', '--gold', str(gold), '--pred', str(pred), '--id_eval_micro_f1', 'nan', '--out_json', str(report)], cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = strict_read(report)
            self.assertIsNone(payload['id_eval_micro_f1'])
            self.assertEqual(payload['GRR']['status'], 'invalid_id_score')


class ReportTests(unittest.TestCase):
    def test_report_serializes_undefined_grr_as_null(self):
        gold = [{"id": "d", "text": "abcd", "spans": [span(0, 4)]}]
        pred = [{"id": "d", "spans": [span(0, 4), span(0, 3)]}]
        for id_f1 in (None, 0, float("nan"), float("inf"), 0.25):
            with self.subTest(id_f1=id_f1):
                report = local.evaluate(gold, pred, groups={"d": ["a", "b"]}, id_eval_micro_f1=id_f1, labels=["A", "B"])
                json.dumps(report, allow_nan=False)
                if id_f1 == 0.25:
                    self.assertEqual(report["GRR"]["GRR"], 4)
                else:
                    self.assertIsNone(report["GRR"]["GRR"])

    def test_strict_writer_rejects_nonfinite_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            with self.assertRaises(ValueError):
                span_metrics.write_json(output, {"GRR": float("nan")})
            self.assertFalse(output.exists())
            span_metrics.write_json(output, {"GRR": None, "status": "zero_id_score"})
            self.assertIsNone(strict_read(output)["GRR"])


if __name__ == "__main__":
    unittest.main()
