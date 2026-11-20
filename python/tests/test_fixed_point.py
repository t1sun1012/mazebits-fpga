"""Arithmetic edge cases and independent scalar/vectorized CNN agreement."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.fixed_point import (
    INT32_MAX, INT32_MIN, IntegerPolicy, QuantLayer, QuantizedPolicy,
    accumulator_bounds, infer, quantize_array, requantize, round_away, round_shift,
)
from mazebits.maze import Maze
from mazebits.scalar_fixed import infer_scalar


def state(maze):
    return np.asarray(encode_state(maze, maze.start), dtype=np.int64).reshape(1, 3, 8, 8)


def one_layer(bits=8, *, bias=0, first_weights=None, head_weights=None):
    weights = np.zeros((1, 3, 3, 3), dtype=np.int64)
    if first_weights is not None:
        weights = np.asarray(first_weights, dtype=np.int64)
    head = np.zeros((4, 1, 1, 1), dtype=np.int64)
    if head_weights is not None:
        head[:, 0, 0, 0] = head_weights
    return QuantizedPolicy(bits, (
        QuantLayer("conv0", weights, np.array([bias], dtype=np.int64), 0, 0, 0),
        QuantLayer("head", head, np.zeros(4, dtype=np.int64), 0, 0, None),
    ))


class FixedArithmeticTests(unittest.TestCase):
    def test_real_quantization_rounds_half_ties_away_and_saturates(self):
        values = [-2.5, -1.5, -0.5, -0.49, 0, 0.49, 0.5, 1.5, 2.5]
        np.testing.assert_array_equal(round_away(values), [-3, -2, -1, 0, 0, 0, 1, 2, 3])
        np.testing.assert_array_equal(
            quantize_array([-9.0, -8.0, -7.5, 6.5, 7.5, 10.0], 0, 4),
            [-8, -8, -8, 7, 7, 7],
        )
        np.testing.assert_array_equal(
            quantize_array([-1.0, 0.25, 0.75, 8.0], 1, 4, signed=False),
            [0, 1, 2, 15],
        )
        for invalid in ([float("nan")], [float("inf")], [float(1 << 62)]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                round_away(invalid)

    def test_signed_shifts_handle_half_ties_and_signed_minimum(self):
        np.testing.assert_array_equal(
            round_shift(np.array([-7, -5, -3, -1, 0, 1, 3, 5, 7]), 1),
            [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        )
        extremes = np.array([INT32_MIN, INT32_MAX], dtype=np.int64)
        np.testing.assert_array_equal(round_shift(extremes, 0), extremes)
        np.testing.assert_array_equal(round_shift(extremes, 32), [-1, 0])
        np.testing.assert_array_equal(
            round_shift(extremes, -31), [-(1 << 62), ((1 << 31) - 1) * (1 << 31)]
        )
        for values, shift in (([INT32_MAX + 1], 0), ([1.0], 1), ([1], -32), ([1], 33)):
            with self.subTest(values=values, shift=shift), self.assertRaises(ValueError):
                round_shift(np.asarray(values), shift)

    def test_relu_rounding_and_left_shift_saturation(self):
        np.testing.assert_array_equal(
            requantize(np.array([-5, 0, 1, 3, 29, 30, 31, INT32_MAX]), 1, 4),
            [0, 0, 1, 2, 15, 15, 15, 15],
        )
        np.testing.assert_array_equal(
            requantize(np.array([INT32_MIN, -1, 0, 1, 7, 8, INT32_MAX]), -1, 4),
            [0, 0, 0, 2, 14, 15, 15],
        )

    def test_unsigned_255_activation_times_negative_weight_is_negative(self):
        policy = one_layer(bias=255, head_weights=[-128, -1, 0, 127])
        scores, traces = infer(policy, state(EXAMPLES["open"]), trace=True)
        np.testing.assert_array_equal(traces["conv0.act"], np.full((1, 1, 8, 8), 255))
        np.testing.assert_array_equal(scores, [[-32640, -255, 0, 32385]])
        self.assertEqual(int(traces["actions"][0]), 3)
        bounds = accumulator_bounds(policy)[1]
        self.assertEqual(bounds["lower"][0], -32640)
        self.assertEqual(bounds["upper"][3], 32385)

    def test_zero_head_scores_choose_lowest_action(self):
        scores, traces = infer(one_layer(bits=4), state(EXAMPLES["at-goal"]), trace=True)
        np.testing.assert_array_equal(scores, np.zeros((1, 4), dtype=np.int64))
        np.testing.assert_array_equal(traces["actions"], [0])

    def test_overflow_proof_rejects_bias_and_canceling_partial_sums(self):
        for bias in (INT32_MIN - 1, INT32_MAX + 1):
            with self.subTest(bias=bias), self.assertRaises(ValueError):
                one_layer(bias=bias)
        weights = np.zeros((1, 3, 3, 3), dtype=np.int64)
        weights[0, 0, 0, 0] = 7
        weights[0, 0, 0, 1] = -7
        # The two coefficients cancel if both inputs are one, but the positive
        # partial sum can still overflow. Final-sum-only reasoning is unsafe.
        with self.assertRaisesRegex(ValueError, "possible signed32 accumulator overflow"):
            one_layer(bits=4, bias=INT32_MAX - 3, first_weights=weights)
        with self.assertRaisesRegex(ValueError, "possible signed32 accumulator overflow"):
            one_layer(bits=4, bias=INT32_MIN + 3, first_weights=weights)

    def test_parameter_arrays_are_copied_and_read_only(self):
        source = np.zeros((1, 3, 3, 3), dtype=np.int64)
        layer = QuantLayer("conv0", source, np.array([0]), 0, 0, 0)
        source[0, 0, 0, 0] = 7
        self.assertEqual(int(layer.weight[0, 0, 0, 0]), 0)
        with self.assertRaises(ValueError):
            layer.weight[0, 0, 0, 0] = 1
        with self.assertRaises(FrozenInstanceError):
            layer.weight_frac = 2

    def test_input_contract_rejects_invalid_states(self):
        policy = one_layer()
        valid = state(EXAMPLES["open"])
        invalid = [valid[:, :, :, :-1], np.zeros((1, 3, 8, 8), dtype=np.int64)]
        overlap = valid.copy()
        overlap[0, 0, 0, 0] = 1
        invalid.append(overlap)
        fractional = valid.astype(np.float64)
        fractional[0, 0, 2, 2] = 0.5
        invalid.append(fractional)
        for inputs in invalid:
            with self.subTest(shape=inputs.shape), self.assertRaises(ValueError):
                infer(policy, inputs)


class ScalarAgreementTests(unittest.TestCase):
    def test_all_intermediates_match_for_both_widths_and_shift_directions(self):
        rng = np.random.default_rng(311)
        walls = frozenset((x, y) for y in range(8) for x in range(8)
                          if (x, y) not in ((0, 0), (7, 7)) and rng.random() < 0.25)
        mazes = [EXAMPLES["open"], EXAMPLES["detour"], Maze(walls, (0, 0), (7, 7))]
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            policy = QuantizedPolicy(bits, (
                QuantLayer("conv0", rng.integers(-limit, limit, (2, 3, 3, 3)),
                           np.array([-3, 5]), 0, 0, 1),  # Left shift by one.
                QuantLayer("conv1", rng.integers(-limit, limit, (2, 2, 3, 3)),
                           np.array([7, -9]), 1, 1, 0),  # Right shift by two.
                QuantLayer("head", rng.integers(-limit, limit, (4, 2, 1, 1)),
                           np.array([-1, 0, 1, 2]), 0, 0, None),
            ))
            for name, maze in enumerate(mazes):
                with self.subTest(bits=bits, maze=name):
                    inputs = state(maze)
                    vector_scores, vector_trace = infer(policy, inputs, trace=True)
                    scalar_scores, scalar_trace = infer_scalar(policy, inputs[0])
                    np.testing.assert_array_equal(vector_scores, scalar_scores)
                    self.assertEqual(set(vector_trace), set(scalar_trace))
                    for key in vector_trace:
                        np.testing.assert_array_equal(vector_trace[key], scalar_trace[key],
                                                      err_msg=key)

    def test_scalar_explicitly_checks_each_partial_sum(self):
        policy = one_layer(bits=4)
        # Simulate corruption after contract validation so this independently
        # tests the oracle's runtime check rather than the constructor proof.
        weights = np.zeros((1, 3, 3, 3), dtype=np.int64)
        weights[0, 0, 0, 0] = 7
        weights[0, 0, 0, 1] = -7
        object.__setattr__(policy.layers[0], "weight", weights)
        object.__setattr__(policy.layers[0], "bias", np.array([INT32_MAX - 3]))
        with self.assertRaisesRegex(OverflowError, "partial sum"):
            infer_scalar(policy, state(EXAMPLES["open"]))

    def test_scalar_requires_one_state_and_returns_same_batch_shape(self):
        policy = one_layer()
        inputs = state(EXAMPLES["open"])
        scores, traces = infer_scalar(policy, inputs)
        self.assertEqual(scores.shape, (1, 4))
        self.assertEqual(traces["head.acc"].shape, (1, 4, 8, 8))
        with self.assertRaises(ValueError):
            infer_scalar(policy, np.concatenate([inputs, inputs]))


if __name__ == "__main__":
    unittest.main()
