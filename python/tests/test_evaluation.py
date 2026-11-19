"""Rollout checks use fixed policies; no shortest-path solver supplies actions."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.evaluation import evaluate_rollouts


def metadata(start=(0, 0), goal=(2, 0), *, walls=(), optimal_steps=2, density=0.1):
    return {
        "split": "validation",
        "layouts": [{
            "id": "fixture",
            "walls": "".join("1" if (x, y) in walls else "0"
                             for y in range(8) for x in range(8)),
            "requested_density": density,
            "goals": [list(goal)],
        }],
        "episodes": [{
            "layout_index": 0, "goal_index": 0,
            "start": list(start), "goal": list(goal), "optimal_steps": optimal_steps,
        }],
    }


class ConstantPolicy(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        assert not self.training
        assert not torch.is_grad_enabled()
        assert inputs.shape[1:] == (3, 8, 8)
        assert torch.all(inputs[:, 1].sum((1, 2)) == 1)
        assert torch.all(inputs[:, 2].sum((1, 2)) == 1)
        logits = torch.zeros((len(inputs), 4))
        logits[:, self.action] = 1
        return logits


class EvaluationTests(unittest.TestCase):
    def test_success_with_exact_path_and_metrics(self):
        policy = ConstantPolicy(1)
        definition = metadata()
        original = deepcopy(definition)
        with patch("mazebits.reference.bfs_distances", side_effect=AssertionError("BFS used")):
            result = evaluate_rollouts(policy, definition)
        row, summary = result["episodes"][0], result["summary"]
        self.assertEqual(row["outcome"], "success")
        self.assertEqual(row["actions"], [1, 1])
        self.assertEqual(row["path"], [[0, 0], [1, 0], [2, 0]])
        self.assertEqual(summary["success_rate"], 1)
        self.assertEqual(summary["successful_mean_steps"], 2)
        self.assertEqual(summary["successful_mean_path_ratio"], 1)
        self.assertEqual(summary["successful_optimal_rate"], 1)
        self.assertEqual(summary["attempted_actions"], 2)
        self.assertTrue(policy.training)
        self.assertEqual(definition, original)

    def test_wall_collision_does_not_repair_action(self):
        result = evaluate_rollouts(
            ConstantPolicy(1), metadata(walls={(1, 0)}, optimal_steps=4)
        )
        row, summary = result["episodes"][0], result["summary"]
        self.assertEqual(row["outcome"], "blocked")
        self.assertEqual(row["steps"], 1)
        self.assertEqual(row["path"], [[0, 0], [0, 0]])
        self.assertEqual(summary["blocked_proposal_rate"], 1)
        self.assertEqual(summary["outcomes"], {
            "success": 0, "blocked": 1, "loop": 0, "timeout": 0,
        })
        self.assertIsNone(summary["successful_mean_path_ratio"])

    def test_unmasked_tie_selects_lowest_action_even_if_blocked(self):
        class TiedPolicy(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros((len(inputs), 4))
        row = evaluate_rollouts(TiedPolicy(), metadata())["episodes"][0]
        self.assertEqual(row["actions"], [0])
        self.assertEqual(row["outcome"], "blocked")

    def test_repeated_position_is_a_loop(self):
        class BackAndForth(torch.nn.Module):
            def forward(self, inputs):
                x = inputs[:, 1].flatten(1).argmax(1) % 8
                actions = torch.where(x == 0, 1, 3)
                return torch.nn.functional.one_hot(actions, 4).float()
        row = evaluate_rollouts(BackAndForth(), metadata())["episodes"][0]
        self.assertEqual(row["outcome"], "loop")
        self.assertEqual(row["actions"], [1, 3])
        self.assertEqual(row["path"], [[0, 0], [1, 0], [0, 0]])

    def test_step_budget_and_success_on_last_allowed_action(self):
        row = evaluate_rollouts(ConstantPolicy(1), metadata(), max_steps=1)["episodes"][0]
        self.assertEqual((row["outcome"], row["steps"]), ("timeout", 1))
        row = evaluate_rollouts(ConstantPolicy(1), metadata(), max_steps=2)["episodes"][0]
        self.assertEqual((row["outcome"], row["steps"]), ("success", 2))

    def test_initial_goal_never_calls_model_and_has_no_path_ratio(self):
        policy = ConstantPolicy(1)
        result = evaluate_rollouts(policy, metadata(goal=(0, 0), optimal_steps=0))
        self.assertEqual(policy.calls, 0)
        self.assertEqual(result["episodes"][0]["path"], [[0, 0]])
        self.assertEqual(result["summary"]["success_rate"], 1)
        self.assertEqual(result["summary"]["successful_optimal_rate"], 1)
        self.assertIsNone(result["summary"]["successful_mean_path_ratio"])
        self.assertIsNone(result["summary"]["blocked_proposal_rate"])

    def test_batch_sizes_and_episode_order_do_not_change_results(self):
        definition = metadata()
        definition["episodes"] += [
            dict(definition["episodes"][0], start=[1, 0], optimal_steps=1),
            dict(definition["episodes"][0], start=[0, 1], optimal_steps=3),
        ]
        single = evaluate_rollouts(ConstantPolicy(1), definition, batch_size=1)
        batched = evaluate_rollouts(ConstantPolicy(1), definition, batch_size=32)
        self.assertEqual(single, batched)
        reversed_definition = dict(definition, episodes=list(reversed(definition["episodes"])))
        reversed_result = evaluate_rollouts(ConstantPolicy(1), reversed_definition)
        self.assertEqual(single["episodes"], list(reversed(reversed_result["episodes"])))
        self.assertEqual(single["summary"], reversed_result["summary"])

    def test_empty_evaluation_has_explicit_null_rates(self):
        definition = metadata()
        definition["episodes"] = []
        result = evaluate_rollouts(ConstantPolicy(1), definition)
        self.assertEqual(result["episodes"], [])
        self.assertEqual(result["summary"]["episodes"], 0)
        self.assertIsNone(result["summary"]["success_rate"])
        self.assertEqual(result["summary"]["by_requested_density"], {})

    def test_invalid_logits_restore_all_training_states(self):
        class BadPolicy(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.child = torch.nn.Identity()
                self.kind = kind

            def forward(self, inputs):
                if self.kind == "shape":
                    return torch.zeros((len(inputs), 5))
                if self.kind == "nan":
                    return torch.full((len(inputs), 4), float("nan"))
                raise RuntimeError("deliberate model failure")

        for kind, error_type in (("shape", ValueError), ("nan", ValueError),
                                 ("raise", RuntimeError)):
            with self.subTest(kind=kind):
                policy = BadPolicy(kind)
                policy.child.eval()
                with self.assertRaises(error_type):
                    evaluate_rollouts(policy, metadata())
                self.assertTrue(policy.training)
                self.assertFalse(policy.child.training)

    def test_invalid_limits_and_metadata_are_rejected(self):
        for kwargs in ({"max_steps": 0}, {"max_steps": True},
                       {"batch_size": 0}, {"batch_size": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_rollouts(ConstantPolicy(1), metadata(), **kwargs)
        definition = metadata()
        definition["layouts"][0]["walls"] = "0" * 63
        with self.assertRaises(ValueError):
            evaluate_rollouts(ConstantPolicy(1), definition)
        definition = metadata()
        definition["episodes"][0]["goal_index"] = 1
        with self.assertRaises(ValueError):
            evaluate_rollouts(ConstantPolicy(1), definition)


if __name__ == "__main__":
    unittest.main()
