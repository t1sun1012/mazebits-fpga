"""Validate fair grouping, paired outcomes, raw-action traces, and test reservation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from compare_policies import (compare, diagnostic_trace, distance_bucket,
                              paired_outcomes, rollout_metrics_grouped,
                              state_metrics_grouped)
from mazebits.dataset import generate_dataset
from mazebits.evaluation import evaluate_rollouts
from mazebits.model import ModelConfig, PolicyCNN
from mazebits.model_io import save_model
from mazebits.training import state_metrics


class TablePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Identity()
        self.register_buffer("table", torch.tensor([
            [0.1, 0.4, 0.4, 0.1], [0.5, 0.3, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.4]], dtype=torch.float64).log())

    def forward(self, inputs):
        return self.table[inputs[:, 1].flatten(1).argmax(1)]


class ConstantPolicy(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        scores = torch.zeros((len(inputs), 4))
        scores[:, self.action] = 1
        return scores


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_grouped_states_match_existing_metrics_and_preserve_modes(self):
        inputs = torch.zeros((3, 3, 8, 8))
        for i in range(3):
            inputs[i, 1, 0, i] = 1
        inputs[:, 2, 7, 7] = 1
        targets = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        model = TablePolicy()
        model.child.eval()
        expected = state_metrics(model, inputs, targets)
        model.child.eval()
        grouped = state_metrics_grouped(model, inputs, targets, np.array([0, 0, 1]),
            np.array([9, 10, 26]), {"layouts": [{"family": "bernoulli"}, {"family": "barrier"}]},
            batch_size=2)
        for key in expected:
            self.assertAlmostEqual(grouped[key], expected[key])
        self.assertEqual(grouped["by_family"]["bernoulli"]["optimal_action_correct"], 1)
        self.assertEqual(grouped["by_family"]["barrier"]["optimal_action_accuracy"], 1)
        self.assertEqual({k: v["states"] for k, v in grouped["by_bfs_distance"].items()},
                         {"0-9": 1, "10-19": 1, "20+": 1})
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

    def test_paired_comparison_reports_gains_and_losses_and_rejects_different_episodes(self):
        base = [{"layout_index": i, "goal_index": 0, "start": [0, 0], "goal": [7, 7],
                 "optimal_steps": 14, "outcome": outcome}
                for i, outcome in enumerate(["success", "success", "loop", "blocked"])]
        candidate = deepcopy(base)
        for row, outcome in zip(candidate, ["success", "loop", "success", "loop"]):
            row["outcome"] = outcome
        result = paired_outcomes(base, candidate)
        self.assertEqual(result, {"both_success": 1, "both_failed": 1,
            "candidate_gained_success": 1, "candidate_lost_success": 1,
            "episodes": 4, "net_success_count_change": 0})
        candidate[0]["start"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "identical episode"):
            paired_outcomes(base, candidate)
        with self.assertRaises(ValueError):
            paired_outcomes(base, [])

    def test_grouped_rollouts_keep_failure_and_path_efficiency_distinct(self):
        metadata = {"layouts": [
            {"walls": "0"*64, "family": "random", "goals": [[2, 0]]},
            {"walls": "0"*64, "family": "barrier", "goals": [[2, 0]]}],
            "episodes": [
                {"layout_index": 0, "goal_index": 0, "start": [0, 0], "goal": [2, 0], "optimal_steps": 2},
                {"layout_index": 1, "goal_index": 0, "start": [0, 1], "goal": [2, 0], "optimal_steps": 3}]}
        result = rollout_metrics_grouped(evaluate_rollouts(ConstantPolicy(1), metadata), metadata)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["by_family"]["random"]["successful_mean_path_ratio"], 1)
        self.assertIsNone(result["by_family"]["barrier"]["successful_mean_path_ratio"])
        self.assertEqual(result["by_family"]["barrier"]["outcomes"]["blocked"], 1)
        self.assertEqual(result["by_bfs_distance"]["0-9"]["episodes"], 2)

    def test_diagnostic_records_illegal_action_without_repair_and_calls_bfs_afterward(self):
        model = ConstantPolicy(1)
        text = "S#G.....\n........\n........\n........\n........\n........\n........\n........"
        def report_only(_maze):
            self.assertEqual(model.calls, 1)
            return ((0, 0), (0, 1), (1, 1), (2, 1), (2, 0))
        with patch("compare_policies.shortest_path", side_effect=report_only):
            trace = diagnostic_trace(model, text)
        self.assertEqual((trace["outcome"], trace["attempts"]), ("blocked", 1))
        self.assertEqual(trace["path"], [[0, 0], [0, 0]])
        self.assertEqual(trace["events"][0]["action"], "RIGHT")
        self.assertEqual(trace["optimal_steps"], 4)

    def test_nonfinite_scores_rejected_and_modes_restored(self):
        model = TablePolicy()
        model.table[0, 0] = torch.nan
        model.child.eval()
        inputs = torch.zeros((1, 3, 8, 8))
        inputs[:, 1, 0, 0] = inputs[:, 2, 7, 7] = 1
        with self.assertRaisesRegex(ValueError, "finite"):
            state_metrics_grouped(model, inputs, torch.tensor([[1, 0, 0, 0]]),
                                  np.array([0]), np.array([1]), {"layouts": [{}]})
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        for distance, expected in ((0, "0-9"), (9, "0-9"), (10, "10-19"), (19, "10-19"), (20, "20+")):
            self.assertEqual(distance_bucket(distance), expected)
        with self.assertRaises(ValueError):
            distance_bucket(-1)

    def test_full_comparison_needs_no_test_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, model_dir, out = root/"data", root/"model", root/"comparison"
            generate_dataset(data, train_layouts=1, val_layouts=1, test_layouts=1,
                             goals_per_layout=1, episodes_per_goal=2)
            # The manifest still lists these reserved files; comparison must not open them.
            (data/"test.npz").unlink()
            (data/"test.json").unlink()
            save_model(PolicyCNN(ModelConfig(channels=2, depth=1)), model_dir, provenance={})
            result = compare(model_dir, model_dir, [data], out, batch_size=32)
            self.assertEqual(result["datasets"][0]["paired_outcomes"]["net_success_count_change"], 0)
            self.assertTrue(result["inputs_unchanged_during_evaluation"])
            self.assertEqual(result["datasets"][0]["models"]["baseline"]["rollouts"],
                             result["datasets"][0]["models"]["candidate"]["rollouts"])
            self.assertEqual(json.loads((out/"comparison.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                compare(model_dir, model_dir, [data], out)


if __name__ == "__main__":
    unittest.main()
