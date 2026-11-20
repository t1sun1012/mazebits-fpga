"""Training-only deterministic scale selection and calibration arithmetic."""

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.calibration import _choose_fraction, _round_away, calibrate
from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.model import ModelConfig, PolicyCNN


def example_inputs():
    maze = EXAMPLES["open"]
    return np.array([
        encode_state(maze, (0, 0)),
        encode_state(maze, (3, 4)),
        encode_state(maze, (7, 0)),
        encode_state(maze, (0, 7)),
    ], dtype=np.uint8).reshape(-1, 3, 8, 8)


class ScaleSelectionTests(unittest.TestCase):
    def test_halfway_rounding_is_away_from_zero(self):
        values = np.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5])
        np.testing.assert_array_equal(_round_away(values), [-3, -2, -1, 0, 1, 2, 3])

    def test_selected_fraction_minimizes_mse_and_ties_prefer_max_fit(self):
        result = _choose_fraction(np.array([-1.0, 0.0, 1.0]), bits=4, signed=True)
        self.assertEqual(result["max_fit_frac"], 2)
        self.assertEqual(result["selected_frac"], 2)
        self.assertEqual(result["selected_mse"], 0)
        candidate = next(row for row in result["candidates"] if row["frac"] == 3)
        self.assertEqual(candidate["clipped_count"], 1)
        self.assertAlmostEqual(candidate["mse"], (0.125 ** 2) / 3)
        self.assertEqual(result["selected_mse"],
                         min(row["mse"] for row in result["candidates"]))

    def test_unsigned_codes_and_fraction_bounds_are_explicit(self):
        result = _choose_fraction(np.array([0.0, 1.0, 2.0, 3.0]), bits=4, signed=False)
        self.assertEqual((result["code_min"], result["code_max"]), (0, 15))
        self.assertEqual(result["selected_frac"], 2)
        for values, expected in ((np.zeros(8), 0),
                                 (np.array([2.0 ** -40]), 24),
                                 (np.array([1.0e20]), -16)):
            result = _choose_fraction(values, bits=8, signed=False)
            self.assertEqual(result["selected_frac"], expected)
            self.assertTrue(all(-16 <= row["frac"] <= 24 for row in result["candidates"]))

    def test_invalid_analysis_values_are_rejected(self):
        for values, bits, signed in (
            (np.array([]), 8, True),
            (np.array([float("nan")]), 8, True),
            (np.array([float("inf")]), 4, False),
            (np.array([-1.0]), 4, False),
            (np.array([1.0]), 3, True),
        ):
            with self.subTest(values=values, bits=bits), self.assertRaises(ValueError):
                _choose_fraction(values, bits=bits, signed=signed)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self):
        torch.manual_seed(311)
        return PolicyCNN(ModelConfig(channels=2, depth=1))

    def test_seeded_indices_reports_and_model_modes_are_reproducible(self):
        inputs = example_inputs()
        model = self.make_model()
        model.train()
        model.head.eval()
        before = [module.training for module in model.modules()]
        first = calibrate(model, inputs, sample_count=3, seed=311)
        second = calibrate(model, inputs, sample_count=3, seed=311)
        self.assertEqual(first, second)
        self.assertEqual([module.training for module in model.modules()], before)
        expected = np.sort(np.random.default_rng(311).choice(4, 3, replace=False)).astype("<i8")
        self.assertEqual(first["sampling"]["indices"], expected.tolist())
        self.assertEqual(first["sampling"]["indices_sha256"],
                         hashlib.sha256(expected.tobytes()).hexdigest())
        for bits in ("8", "4"):
            scheme = first["schemes"][bits]
            self.assertEqual(len(scheme["weight_fracs"]), 2)
            self.assertEqual(len(scheme["activation_fracs"]), 1)
            self.assertEqual([row["name"] for row in scheme["layers"]], ["convs.0", "head"])
            self.assertIsNone(scheme["layers"][-1]["activation"])
            self.assertEqual(scheme["layers"][0]["activation"]["values"], 3 * 2 * 8 * 8)

    def test_small_input_caps_sampling_and_zero_model_uses_zero_exponents(self):
        model = self.make_model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        result = calibrate(model, example_inputs(), sample_count=4096)
        self.assertEqual(result["sampling"]["actual_samples"], 4)
        for scheme in result["schemes"].values():
            self.assertEqual(scheme["weight_fracs"], [0, 0])
            self.assertEqual(scheme["activation_fracs"], [0])

    def test_input_configuration_and_nonfinite_model_validation(self):
        model = self.make_model()
        valid = example_inputs()
        invalid = [
            valid.tolist(), np.empty((0, 3, 8, 8), dtype=np.uint8),
            np.zeros((2, 3, 7, 8)), np.zeros_like(valid),
        ]
        nonbinary = valid.astype(np.float32)
        nonbinary[0, 0, 0, 1] = 0.5
        invalid.append(nonbinary)
        overlapping = valid.copy()
        overlapping[0, 0, 0, 0] = 1
        invalid.append(overlapping)
        for values in invalid:
            with self.subTest(kind=type(values)), self.assertRaises(ValueError):
                calibrate(model, values)
        for kwargs in ({"sample_count": 0}, {"sample_count": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calibrate(model, valid, **kwargs)
        with torch.no_grad():
            model.head.weight[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite model"):
            calibrate(model, valid)


if __name__ == "__main__":
    unittest.main()
