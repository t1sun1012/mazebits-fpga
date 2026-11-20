"""Verify model integrity and retention when fine-tuning does not improve."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mazebits.dataset import generate_dataset
from mazebits.model import ModelConfig, PolicyCNN
from mazebits.model_io import load_model, save_model
from mazebits.training import initialize_policy, train


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = ModelConfig(channels=2, depth=1)
        torch.manual_seed(312)
        self.original = PolicyCNN(self.config)
        self.bundle = self.root / "initial"
        save_model(self.original, self.bundle, provenance={"dataset_manifest_sha256": "previous-dataset"})

    def tearDown(self):
        self.temporary.cleanup()

    def test_initialization_loads_exact_weights_and_records_identity(self):
        loaded, identity = initialize_policy(self.config, self.bundle)
        for name, tensor in self.original.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
        self.assertEqual(identity["source_dataset_manifest_sha256"], "previous-dataset")
        self.assertEqual(len(identity["weights_sha256"]), 64)

    def test_initialization_rejects_architecture_mismatch_and_corrupt_weights(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            initialize_policy(ModelConfig(channels=3, depth=1), self.bundle)
        with (self.bundle / "weights.npz").open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            initialize_policy(self.config, self.bundle)

    def test_checkpoint_zero_is_retained_without_validation_improvement(self):
        dataset = self.root / "data"
        generate_dataset(dataset, train_layouts=1, val_layouts=1, test_layouts=1,
                         goals_per_layout=1, episodes_per_goal=2, seed=312)
        run_dir = self.root / "run"
        # Real training/evaluation forward and backward passes, but deliberately
        # suppress updates to verify an unchanged candidate cannot displace zero.
        with patch("torch.set_num_interop_threads"), patch.object(torch.optim.Adam, "step", return_value=None):
            metrics = train(dataset, run_dir, channels=2, depth=1, epochs=1,
                            initial_model=self.bundle, batch_size=128, threads=1)
        self.assertEqual(metrics["best_epoch"], 0)
        self.assertEqual(metrics["epochs_completed"], 1)
        selected, manifest = load_model(run_dir / "model")
        for name, tensor in self.original.state_dict().items():
            self.assertTrue(torch.equal(tensor, selected.state_dict()[name]), name)
        self.assertEqual(manifest["provenance"]["selected_epoch"], 0)
        self.assertTrue((run_dir / "initial_metrics.json").is_file())
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["training_config"]["initialization"]["source_dataset_manifest_sha256"], "previous-dataset")


if __name__ == "__main__":
    unittest.main()
