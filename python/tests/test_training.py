"""Guard test-set reservation, dataset integrity, and tie-aware state metrics."""

import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.dataset import generate_dataset
from mazebits.model_io import sha256
from mazebits.training import load_split, state_metrics


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.folder = cls.root / "source"
        generate_dataset(cls.folder, train_layouts=1, val_layouts=1, test_layouts=1,
                         goals_per_layout=1, episodes_per_goal=2, seed=311)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_final_test_split_is_rejected_before_any_file_is_opened(self):
        with patch.object(Path, "open", side_effect=AssertionError("must not read files")) as opened:
            with self.assertRaisesRegex(ValueError, "test is reserved"):
                load_split(self.root / "does-not-exist", "test")
            opened.assert_not_called()

    def test_train_and_validation_load_exact_float_inputs_and_binary_targets(self):
        for split in ("train", "val"):
            with self.subTest(split=split):
                inputs, targets, metadata = load_split(self.folder, split)
                self.assertEqual(inputs.dtype, torch.float32)
                self.assertEqual(targets.dtype, torch.uint8)
                self.assertEqual(tuple(inputs.shape[1:]), (3, 8, 8))
                self.assertEqual(metadata["split"], split)
                with np.load(self.folder / f"{split}.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(inputs.numpy(), archive["inputs"])
                    np.testing.assert_array_equal(targets.numpy(), archive["targets"])

    def test_corrupted_array_file_fails_integrity_check(self):
        folder = self.root / "corrupt-array"
        shutil.copytree(self.folder, folder)
        with (folder / "train.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "dataset hash mismatch: train.npz"):
            load_split(folder, "train")

    def test_inconsistent_metadata_split_is_rejected_even_with_matching_hash(self):
        folder = self.root / "wrong-split"
        shutil.copytree(self.folder, folder)
        metadata_path = folder / "val.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["split"] = "train"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["val.json"]["sha256"] = sha256(metadata_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata split mismatch"):
            load_split(folder, "val")


class TablePolicy(torch.nn.Module):
    """A fixed policy indexed by the agent's cell, independent of BFS."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scores", torch.tensor([
            [0.1, 0.4, 0.4, 0.1],
            [0.5, 0.3, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ], dtype=torch.float64).log())

    def forward(self, inputs):
        index = inputs[:, 1].flatten(1).argmax(dim=1)
        return self.scores[index]


class StateMetricTests(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.zeros(3, 3, 8, 8)
        self.inputs[:, 2, 7, 7] = 1
        for index in range(3):
            self.inputs[index, 1, 0, index] = 1
        self.targets = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def test_tied_optimal_actions_accuracy_loss_and_mode_restore(self):
        model = TablePolicy()
        expected_loss = -(math.log(0.8) + math.log(0.3) + math.log(0.4)) / 3
        for training in (True, False):
            for batch_size in (1, 2, 3):
                with self.subTest(training=training, batch_size=batch_size):
                    model.train(training)
                    result = state_metrics(model, self.inputs, self.targets,
                                           batch_size=batch_size)
                    self.assertEqual(result["states"], 3)
                    self.assertEqual(result["optimal_action_correct"], 2)
                    self.assertEqual(result["optimal_action_accuracy"], 2 / 3)
                    self.assertAlmostEqual(result["optimal_set_loss"], expected_loss, places=12)
                    self.assertEqual(model.training, training)

    def test_nonfinite_predictions_raise_and_restore_training_mode(self):
        model = TablePolicy().train()
        model.scores[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "non-finite logits"):
            state_metrics(model, self.inputs, self.targets)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()

