"""Independent Python-integer CNN oracle for fixed-point export verification."""

import numpy as np

from .fixed_point import QuantizedPolicy, input_codes


def infer_scalar(policy: QuantizedPolicy, inputs):
    """Evaluate one state with explicit loops and check every signed32 partial sum.

    The arithmetic intentionally does not call the vector convolution, rounded
    shift, or requantization helpers. Python integers provide the widened
    temporary, including negative shifts and the signed32 minimum value.
    """
    original = np.asarray(inputs)
    if original.shape == (3, 8, 8):
        original = original[None, ...]
    if original.shape != (1, 3, 8, 8):
        raise ValueError("scalar inference accepts exactly one 3x8x8 state")
    original = input_codes(original)
    current = original[0].tolist()
    traces = {}
    lower, upper = -(1 << 31), (1 << 31) - 1
    for layer_index, layer in enumerate(policy.layers):
        weights, biases = layer.weight.tolist(), layer.bias.tolist()
        outputs, inputs_count, kernel, _ = layer.weight.shape
        radius = kernel // 2
        accumulators = [[[0 for _ in range(8)] for _ in range(8)]
                        for _ in range(outputs)]
        next_values = [[[0 for _ in range(8)] for _ in range(8)]
                       for _ in range(outputs)]
        for output_channel in range(outputs):
            for y in range(8):
                for x in range(8):
                    total = int(biases[output_channel])
                    if not lower <= total <= upper:
                        raise OverflowError(f"{layer.name}: bias exceeds signed32")
                    for input_channel in range(inputs_count):
                        for ky in range(kernel):
                            for kx in range(kernel):
                                iy, ix = y + ky - radius, x + kx - radius
                                if 0 <= iy < 8 and 0 <= ix < 8:
                                    value = current[input_channel][iy][ix]
                                else:
                                    value = int(layer_index == 0 and input_channel == 0)
                                coefficient = weights[output_channel][input_channel][ky][kx]
                                total += value * coefficient
                                if not lower <= total <= upper:
                                    raise OverflowError(
                                        f"{layer.name}: partial sum exceeds signed32"
                                    )
                    accumulators[output_channel][y][x] = total
                    if layer.output_frac is None:
                        result = total
                    else:
                        shift = layer.input_frac + layer.weight_frac - layer.output_frac
                        if shift > 0:
                            magnitude = (abs(total) + (1 << (shift - 1))) // (1 << shift)
                            shifted = -magnitude if total < 0 else magnitude
                        else:
                            shifted = total * (1 << -shift)
                        result = min((1 << policy.bits) - 1, max(0, shifted))
                    next_values[output_channel][y][x] = result
        traces[f"{layer.name}.acc"] = np.asarray([accumulators], dtype=np.int64)
        if layer.output_frac is not None:
            traces[f"{layer.name}.act"] = np.asarray([next_values], dtype=np.int64)
        current = next_values
    position = original[0, 1].reshape(-1).tolist().index(1)
    y, x = divmod(position, 8)
    scores_list = [current[action][y][x] for action in range(4)]
    scores = np.asarray([scores_list], dtype=np.int64)
    # Python's max keeps the first item when scores tie.
    action = max(range(4), key=lambda index: scores_list[index])
    traces["scores"] = scores.copy()
    traces["actions"] = np.asarray([action], dtype=np.int64)
    return scores, traces
