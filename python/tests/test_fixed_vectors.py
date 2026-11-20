"""Full trace ordering, independent replay, and invalid-vector rejection."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.fixed_point import QuantLayer, QuantizedPolicy
from mazebits.fixed_vectors import export_vectors, verify_vectors
from mazebits.model_io import sha256, write_json
from mazebits.quantized_io import save_quantized


def tiny_policy():
    return QuantizedPolicy(4, (
        QuantLayer("conv0", (np.arange(27) % 16 - 8).reshape(1, 3, 3, 3),
                   np.array([3]), 0, 1, 2),
        QuantLayer("head", np.array([-8, 7, -1, 1]).reshape(4, 1, 1, 1),
                   np.array([-9, 2, 0, 13]), 2, 3, None),
    ))


class FixedVectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.vectors = self.root / "vectors"
        self.training = self.root / "train.json"
        self.metadata = {
            "split": "train",
            "layouts": [{"walls": "0" * 64, "goals": [[7, 7]]}],
            "episodes": [{"layout_index": 0, "goal_index": 0,
                          "start": [0, 0], "goal": [7, 7], "optimal_steps": 14}],
        }
        write_json(self.training, self.metadata)
        save_quantized(tiny_policy(), self.model)
        self.manifest = export_vectors(self.model, self.vectors, self.training)

    def tearDown(self):
        self.temp.cleanup()

    def store_manifest(self, manifest):
        write_json(self.vectors / "manifest.json", manifest)

    def rewrite_memory(self, filename, change):
        path = self.vectors / filename
        rows = path.read_text().splitlines()
        change(rows)
        path.write_text("\n".join(rows) + "\n", encoding="ascii", newline="\n")
        manifest = copy.deepcopy(self.manifest)
        manifest["files"][filename]["sha256"] = sha256(path)
        manifest["files"][filename]["bytes"] = path.stat().st_size
        self.store_manifest(manifest)

    def test_complete_export_and_independent_replay(self):
        result = verify_vectors(self.model, self.vectors)
        self.assertEqual(result["cases"], 8)
        self.assertEqual(result["total_cases"], 8)
        self.assertEqual(result["compared_words"], 8 * (64 + 64 + 256 + 4 + 1))
        self.assertEqual(result["input_words"], 8 * 192)
        self.assertEqual(result["mismatches"], 0)
        self.assertTrue(result["all_file_hashes_verified"])
        self.assertEqual(self.manifest["verification"]["compared_words"],
                         result["compared_words"])
        self.assertEqual(self.manifest["files"]["conv0_acc.mem"]["shape"], [8, 1, 8, 8])
        self.assertEqual(self.manifest["files"]["head_acc.mem"]["shape"], [8, 4, 8, 8])
        self.assertEqual(self.manifest["files"]["inputs.mem"]["bits"], 1)
        self.assertFalse(self.manifest["files"]["inputs.mem"]["signed"])
        self.assertEqual(self.manifest["files"]["conv0_act.mem"]["bits"], 4)
        self.assertTrue(self.manifest["files"]["scores.mem"]["signed"])
        self.assertEqual(self.manifest["files"]["actions.mem"]["bits"], 2)
        inputs = (self.vectors / "inputs.mem").read_text().splitlines()
        self.assertEqual(inputs[64], "1")  # Open maze's initial agent (0, 0).
        self.assertEqual(inputs[191], "1")  # Its goal (7, 7).
        self.assertEqual(inputs[192 + 64 + 3 * 8 + 3], "1")  # Adjacent case starts (3, 3).

    def test_limited_replay_reports_selected_case_count(self):
        result = verify_vectors(self.model, self.vectors, case_limit=1)
        self.assertEqual(result["cases"], 1)
        self.assertEqual(result["total_cases"], 8)
        self.assertEqual(result["compared_words"], 389)
        self.assertEqual(result["input_words"], 192)

    def test_invalid_limits_are_rejected(self):
        for limit in (0, -1, 9, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                verify_vectors(self.model, self.vectors, case_limit=limit)

    def test_hashes_cover_all_files_even_when_replay_is_limited(self):
        path = self.vectors / "actions.mem"
        original = path.read_bytes()
        rows = path.read_text().splitlines()
        rows[-1] = str(int(rows[-1], 16) ^ 1)
        path.write_text("\n".join(rows) + "\n", encoding="ascii", newline="\n")
        self.assertNotEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_vectors(self.model, self.vectors, case_limit=1)

    def test_rehashed_numerically_wrong_scores_fail_reference_check(self):
        self.rewrite_memory(
            "scores.mem", lambda rows: rows.__setitem__(
                0, f"{int(rows[0], 16) ^ 1:08x}"))
        with self.assertRaisesRegex(ValueError, "trace mismatch"):
            verify_vectors(self.model, self.vectors)

    def test_rehashed_truncated_memory_is_rejected(self):
        self.rewrite_memory("scores.mem", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "word count"):
            verify_vectors(self.model, self.vectors)

    def test_upper_bits_and_word_counts_are_checked(self):
        self.rewrite_memory("inputs.mem", lambda rows: rows.__setitem__(0, "2"))
        with self.assertRaisesRegex(ValueError, "declared width"):
            verify_vectors(self.model, self.vectors)

    def test_shape_and_case_metadata_cannot_disagree_with_memory(self):
        mutations = [
            lambda m: m["files"]["scores.mem"].update(shape=[8, 5]),
            lambda m: m["files"]["inputs.mem"].update(words=1),
            lambda m: m["files"]["actions.mem"].update(signed=True),
            lambda m: m["cases"][0].update(maze_ascii=".S......\n........\n........\n........\n........\n........\n........\n.......G"),
            lambda m: m["cases"][0].update(name="unknown"),
            lambda m: m["verification"].update(compared_words=1),
            lambda m: m["verification"].update(mismatches=False),
            lambda m: m["fixture_sources"].update(training_episode_index=True),
        ]
        for mutation in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutation(manifest)
            self.store_manifest(manifest)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_vectors(self.model, self.vectors)
        self.store_manifest(self.manifest)

    def test_file_names_cannot_redirect_reads(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["files"]["../inputs.mem"] = manifest["files"].pop("inputs.mem")
        self.store_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "file set"):
            verify_vectors(self.model, self.vectors)

    def test_a_different_model_bundle_is_rejected(self):
        policy = tiny_policy()
        changed = QuantizedPolicy(4, (policy.layers[0], QuantLayer(
            "head", policy.layers[1].weight, policy.layers[1].bias + 1,
            2, 3, None,
        )))
        other = self.root / "other-model"
        save_quantized(changed, other)
        with self.assertRaisesRegex(ValueError, "different model"):
            verify_vectors(other, self.vectors)

    def test_only_training_metadata_is_used_and_existing_output_is_preserved(self):
        original = (self.vectors / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            export_vectors(self.model, self.vectors, self.training)
        self.assertEqual((self.vectors / "manifest.json").read_bytes(), original)
        self.metadata["split"] = "test"
        write_json(self.training, self.metadata)
        with self.assertRaisesRegex(ValueError, "only training"):
            export_vectors(self.model, self.root / "invalid-export", self.training)
        self.assertFalse((self.root / "invalid-export").exists())


if __name__ == "__main__":
    unittest.main()
