"""FPGA memory order, signed encoding, and quantized-bundle integrity."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.fixed_point import QuantLayer, QuantizedPolicy, infer
from mazebits.model_io import sha256, write_json
from mazebits.quantized_io import load_quantized, save_quantized


def small_policy(bits):
    lower, count = -(1 << (bits - 1)), 1 << bits
    weights = (np.arange(54, dtype=np.int64) % count + lower).reshape(2, 3, 3, 3)
    head = np.array([lower, (count // 2) - 1, -1, 0, 1, 2, -2, 3],
                    dtype=np.int64).reshape(4, 2, 1, 1)
    return QuantizedPolicy(bits, (
        QuantLayer("conv0", weights, np.array([-13, 17]), 0, 1, 2),
        QuantLayer("head", head, np.array([-123, 0, 456, -9]), 2, 3, None),
    ), {"purpose": "memory-contract test"})


class QuantizedBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "model"
        self.policy = small_policy(4)
        self.manifest = save_quantized(self.policy, self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def store_manifest(self, manifest):
        write_json(self.folder / "manifest.json", manifest)

    def update_file_hash(self, manifest, name):
        path = self.folder / name
        manifest["files"][name]["sha256"] = sha256(path)
        manifest["files"][name]["bytes"] = path.stat().st_size

    def test_both_widths_round_trip_exact_words_scales_and_inference(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                folder = Path(self.temp.name) / f"width-{bits}"
                original = small_policy(bits)
                manifest = save_quantized(original, folder)
                loaded = load_quantized(folder)
                self.assertEqual(loaded.bits, bits)
                self.assertEqual(loaded.provenance, original.provenance)
                for first, second in zip(original.layers, loaded.layers):
                    self.assertEqual(first.name, second.name)
                    self.assertEqual(
                        (first.input_frac, first.weight_frac, first.output_frac),
                        (second.input_frac, second.weight_frac, second.output_frac),
                    )
                    np.testing.assert_array_equal(first.weight, second.weight)
                    np.testing.assert_array_equal(first.bias, second.bias)
                flat_weights = np.concatenate([layer.weight.ravel() for layer in original.layers])
                raw_words = (folder / "weights.mem").read_text().splitlines()
                self.assertEqual(
                    [int(word, 16) for word in raw_words],
                    [int(value) & ((1 << bits) - 1) for value in flat_weights],
                )
                self.assertEqual(len(raw_words), 62)
                self.assertEqual((folder / "biases.mem").read_text().splitlines()[0], "fffffff3")
                self.assertEqual(manifest["layers"][1]["weight_offset"], 54)
                self.assertEqual(manifest["layers"][1]["bias_offset"], 2)
                self.assertEqual(manifest["logical_storage"]["total_bytes"], 55 if bits == 4 else 86)
                header = (folder / "quantization.vh").read_text()
                self.assertIn("localparam integer L0_SHIFT = -1;", header)
                self.assertIn("localparam integer L1_OUTPUT_FRAC = 5;", header)
                self.assertIn("localparam integer L1_HAS_RELU = 0;", header)
                inputs = np.asarray([
                    encode_state(EXAMPLES[name], EXAMPLES[name].start)
                    for name in ("open", "detour", "at-goal")
                ]).reshape(-1, 3, 8, 8)
                scores, trace = infer(original, inputs, trace=True)
                restored_scores, restored_trace = infer(loaded, inputs, trace=True)
                np.testing.assert_array_equal(scores, restored_scores)
                self.assertEqual(trace.keys(), restored_trace.keys())
                for name in trace:
                    np.testing.assert_array_equal(trace[name], restored_trace[name])

    def test_existing_bundle_is_not_overwritten(self):
        original = (self.folder / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            save_quantized(self.policy, self.folder)
        self.assertEqual((self.folder / "manifest.json").read_bytes(), original)

    def test_tampered_weight_bias_and_header_files_are_rejected(self):
        for name in ("weights.mem", "biases.mem", "quantization.vh"):
            with self.subTest(name=name):
                path = self.folder / name
                original = path.read_bytes()
                path.write_bytes(original + b"0\n")
                try:
                    with self.assertRaises(ValueError):
                        load_quantized(self.folder)
                finally:
                    path.write_bytes(original)

    def test_hashed_invalid_hex_and_wrong_word_counts_are_rejected(self):
        path = self.folder / "weights.mem"
        original = path.read_text()
        for modified, change_words in (
            ("10\n" + original.split("\n", 1)[1], False),
            ("g\n" + original.split("\n", 1)[1], False),
            (original + "0\n", False),
            (original + "0\n", True),
        ):
            with self.subTest(modified=modified[:4], change_words=change_words):
                path.write_text(modified, encoding="ascii", newline="\n")
                manifest = copy.deepcopy(self.manifest)
                self.update_file_hash(manifest, "weights.mem")
                if change_words:
                    manifest["files"]["weights.mem"]["words"] += 1
                self.store_manifest(manifest)
                with self.assertRaises(ValueError):
                    load_quantized(self.folder)
        path.write_text(original, encoding="ascii", newline="\n")
        self.store_manifest(self.manifest)

    def test_offsets_shapes_fractions_proofs_and_storage_are_checked(self):
        mutations = [
            lambda m: m["layers"][1].update(weight_offset=53),
            lambda m: m["layers"][0].update(weight_shape=[2, 3, 3, 2]),
            lambda m: m["layers"][0].update(input_frac=True),
            lambda m: m["layers"][0].update(shift=0),
            lambda m: m["layers"][1].update(input_frac=3, accumulator_frac=6),
            lambda m: m["accumulator_bounds"][0].update(max=123456),
            lambda m: m["logical_storage"].update(total_bytes=1),
            lambda m: m.update(layer_count=3),
            lambda m: m["contract"].update(bias_bits=16),
        ]
        for mutation in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutation(manifest)
            self.store_manifest(manifest)
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                load_quantized(self.folder)
        self.store_manifest(self.manifest)

    def test_header_with_a_matching_hash_must_still_match_manifest(self):
        path = self.folder / "quantization.vh"
        text = path.read_text().replace("ACC_BITS = 32", "ACC_BITS = 31")
        path.write_text(text, encoding="ascii", newline="\n")
        manifest = copy.deepcopy(self.manifest)
        self.update_file_hash(manifest, "quantization.vh")
        self.store_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "header disagrees"):
            load_quantized(self.folder)

    def test_manifest_cannot_redirect_reads_to_other_filenames(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["files"]["../weights.mem"] = manifest["files"].pop("weights.mem")
        self.store_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "exactly the fixed"):
            load_quantized(self.folder)


if __name__ == "__main__":
    unittest.main()
