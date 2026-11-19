"""CNN arithmetic/layout contracts and safe floating-point bundle round trips."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.maze import Maze
from mazebits.model import ModelConfig, PolicyCNN, augment_batch, optimal_set_loss
from mazebits.model_io import load_model, save_model, sha256
from mazebits.reference import optimal_actions


def input_batch(maze, positions):
    return torch.tensor([encode_state(maze, position) for position in positions],
                        dtype=torch.float32).reshape(-1, 3, 8, 8)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_parameter_and_dense_mac_budget(self):
        description = PolicyCNN().description()
        self.assertEqual(description["parameters"], 16_756)
        self.assertEqual(description["weights"], 16_624)
        self.assertEqual(description["biases"], 132)
        self.assertEqual(description["dense_macs_per_decision"], 1_063_936)
        self.assertEqual(description["receptive_field"], 17)
        self.assertEqual(description["action_order"], ["UP", "RIGHT", "DOWN", "LEFT"])

    def test_agent_cell_readout_matches_four_action_field_values(self):
        model = PolicyCNN(ModelConfig(channels=3, depth=2)).eval()
        maze = EXAMPLES["open"]
        positions = [(0, 0), (3, 4), (7, 7)]
        inputs = input_batch(maze, positions)
        with torch.no_grad():
            field = model.action_field(inputs)
            actual = model(inputs)
        self.assertEqual(tuple(field.shape), (3, 4, 8, 8))
        expected = torch.stack([field[index, :, y, x]
                                for index, (x, y) in enumerate(positions)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_first_layer_sees_walls_outside_grid_but_no_agent_or_goal(self):
        model = PolicyCNN(ModelConfig(channels=2, depth=1)).eval()
        inputs = input_batch(EXAMPLES["open"], [(3, 4)])
        captured = []
        hook = model.convs[0].register_forward_pre_hook(
            lambda module, arguments: captured.append(arguments[0].detach().clone()))
        try:
            with torch.no_grad():
                model(inputs)
        finally:
            hook.remove()
        padded = captured[0]
        self.assertEqual(tuple(padded.shape), (1, 3, 10, 10))
        torch.testing.assert_close(padded[:, :, 1:9, 1:9], inputs, rtol=0, atol=0)
        border = torch.ones((10, 10), dtype=torch.bool)
        border[1:9, 1:9] = False
        self.assertTrue(torch.all(padded[0, 0, border] == 1))
        self.assertTrue(torch.all(padded[0, 1:, border] == 0))

    def test_optimal_set_loss_uses_probability_mass_of_all_good_actions(self):
        probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                      [0.1, 0.1, 0.2, 0.6]], dtype=torch.float64)
        logits = probabilities.log().requires_grad_()
        targets = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]])
        loss = optimal_set_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), -math.log(0.4) / 2, places=12)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[1], torch.zeros(4, dtype=torch.float64),
                                   rtol=0, atol=1e-15)

    def test_invalid_model_shapes_targets_and_config_are_rejected(self):
        for config in (dict(channels=0), dict(depth=0), dict(channels=True),
                       dict(depth=1.5)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ModelConfig(**config)
        model = PolicyCNN(ModelConfig(channels=2, depth=1))
        for shape in ((1, 3, 7, 8), (3, 8, 8), (1, 2, 8, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))
        for targets in (torch.zeros(2, 4), torch.full((2, 4), 0.5), torch.ones(2, 3)):
            with self.subTest(targets=targets.tolist()), self.assertRaises(ValueError):
                optimal_set_loss(torch.zeros(2, 4), targets)
        with self.assertRaises(ValueError):
            augment_batch(torch.zeros(1, 3, 8, 8), torch.ones(1, 4), 4, False)

    def test_all_d4_augmentations_preserve_maze_state_and_bfs_action_labels(self):
        maze = EXAMPLES["detour"]
        positions = [(0, 0), (0, 7), (2, 7), (2, 0), (7, 1), (5, 4)]
        inputs = input_batch(maze, positions)
        targets = torch.zeros(len(positions), 4)
        for index, position in enumerate(positions):
            targets[index, list(map(int, optimal_actions(maze, position)))] = 1
        seen_layouts = set()
        for turns in range(4):
            for reflect in (False, True):
                augmented, labels = augment_batch(inputs, targets, turns, reflect)
                seen_layouts.add(tuple(augmented[0, 0].flatten().tolist()))
                for index in range(len(positions)):
                    with self.subTest(turns=turns, reflect=reflect, sample=index):
                        wall_cells = frozenset(
                            (int(x), int(y)) for y, x in augmented[index, 0].nonzero())
                        agents = augmented[index, 1].nonzero().tolist()
                        goals = augmented[index, 2].nonzero().tolist()
                        self.assertEqual(len(agents), 1)
                        self.assertEqual(len(goals), 1)
                        self.assertEqual(len(wall_cells), len(maze.walls))
                        agent, goal = tuple(reversed(agents[0])), tuple(reversed(goals[0]))
                        transformed = Maze(wall_cells, agent, goal)
                        expected = set(map(int, optimal_actions(transformed, agent)))
                        self.assertEqual(set(labels[index].nonzero().flatten().tolist()),
                                         expected)
                        self.assertEqual(augmented[index].flatten().tolist(),
                                         list(encode_state(transformed, agent)))
        self.assertEqual(len(seen_layouts), 8)
        # Augmentation must not modify the original state/label tensors.
        torch.testing.assert_close(inputs, input_batch(maze, positions), rtol=0, atol=0)


class ModelBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_model_round_trip_preserves_tensors_and_outputs_exactly(self):
        model = PolicyCNN(ModelConfig(channels=3, depth=2)).eval()
        inputs = input_batch(EXAMPLES["detour"], [(0, 0), (2, 7)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            saved = save_model(model, path, provenance={"purpose": "unit-test"})
            loaded, manifest = load_model(path)
            self.assertEqual(manifest, saved)
            self.assertFalse(loaded.training)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, loaded.state_dict()[name],
                                           rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), loaded(inputs), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                save_model(model, path, provenance={})

    def test_tampered_weight_bytes_fail_hash_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            save_model(PolicyCNN(ModelConfig(2, 1)), path, provenance={})
            with (path / "weights.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_model(path)

    def test_wrong_tensor_shape_dtype_and_nonfinite_values_are_rejected(self):
        for corruption in ("shape", "dtype", "nonfinite", "names"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model"
                save_model(PolicyCNN(ModelConfig(2, 1)), path, provenance={})
                with np.load(path / "weights.npz", allow_pickle=False) as archive:
                    tensors = {name: archive[name].copy() for name in archive.files}
                name = "convs.0.weight"
                if corruption == "shape":
                    tensors[name] = tensors[name][:1]
                elif corruption == "dtype":
                    tensors[name] = tensors[name].astype(np.float64)
                elif corruption == "nonfinite":
                    tensors[name].flat[0] = np.nan
                else:
                    tensors["unexpected"] = tensors.pop(name)
                np.savez(path / "weights.npz", **tensors)
                manifest = json.loads((path / "manifest.json").read_text())
                # Independently validate actual tensor data, beyond file integrity.
                manifest["weights_sha256"] = sha256(path / "weights.npz")
                (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_model(path)


if __name__ == "__main__":
    unittest.main()

