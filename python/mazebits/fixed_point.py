"""Exact integer reference for the MazeBits FPGA arithmetic contract."""

from dataclasses import dataclass, field
import numpy as np
import torch

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def round_away(values) -> np.ndarray:
    """Round finite real values to integers, with half ties away from zero."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (np.abs(values) >= (1 << 62)).any():
        raise ValueError("non-finite or excessively large value for integer conversion")
    return (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(np.int64)


def quantize_array(values, frac: int, bits: int, *, signed: bool = True) -> np.ndarray:
    if type(frac) is not int or not -32 <= frac <= 48:
        raise ValueError("invalid fractional-bit exponent")
    if type(bits) is not int or not 1 <= bits <= 32:
        raise ValueError("invalid bit width")
    codes = round_away(np.ldexp(np.asarray(values, dtype=np.float64), frac))
    lower = -(1 << (bits - 1)) if signed else 0
    upper = (1 << (bits - (1 if signed else 0))) - 1
    return np.clip(codes, lower, upper)


def round_shift(values, shift: int) -> np.ndarray:
    """Signed32 -> signed64 shift, ties away from zero; negative shift is left."""
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or (values < INT32_MIN).any() or (values > INT32_MAX).any():
        raise ValueError("shift input must contain signed32-range integers")
    if type(shift) is not int or not -31 <= shift <= 32:
        raise ValueError("shift must be between -31 and 32")
    values = values.astype(np.int64)
    if shift <= 0:
        return values * (1 << -shift)
    magnitude = (np.abs(values) + (1 << (shift - 1))) >> shift
    return np.where(values < 0, -magnitude, magnitude)


def requantize(accumulator, shift: int, bits: int) -> np.ndarray:
    """ReLU, rounded shift, and unsigned saturation, in widened arithmetic."""
    if bits not in (4, 8):
        raise ValueError("activation width must be 4 or 8")
    shifted = round_shift(accumulator, shift)
    return np.clip(shifted, 0, (1 << bits) - 1)


@dataclass(frozen=True)
class QuantLayer:
    name: str
    weight: np.ndarray
    bias: np.ndarray
    input_frac: int
    weight_frac: int
    output_frac: int | None

    def __post_init__(self):
        for name in ("weight", "bias"):
            array = np.asarray(getattr(self, name))
            if array.dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integers")
            if array.dtype.kind == "u" and (array > INT32_MAX).any():
                raise ValueError(f"{name} contains an out-of-range unsigned value")
            array = array.astype(np.int64, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for value in (self.input_frac, self.weight_frac):
            if type(value) is not int or not -16 <= value <= 24:
                raise ValueError("fraction exponents must be integers in [-16,24]")
        if self.output_frac is not None and (
            type(self.output_frac) is not int or not -16 <= self.output_frac <= 24
        ):
            raise ValueError("invalid output exponent")

    @property
    def shift(self) -> int | None:
        return None if self.output_frac is None else (
            self.input_frac + self.weight_frac - self.output_frac
        )


@dataclass(frozen=True)
class QuantizedPolicy:
    bits: int
    layers: tuple[QuantLayer, ...]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.bits) is not int or self.bits not in (4, 8):
            raise ValueError("weights and hidden activations must use 4 or 8 bits")
        object.__setattr__(self, "layers", tuple(self.layers))
        if len(self.layers) < 2:
            raise ValueError("at least one hidden convolution and one head are required")
        previous_channels, previous_frac = 3, 0
        channels = self.layers[0].weight.shape[0] if self.layers[0].weight.ndim == 4 else 0
        if channels < 1:
            raise ValueError("invalid channel count")
        for index, layer in enumerate(self.layers):
            head = index == len(self.layers) - 1
            expected_name = "head" if head else f"conv{index}"
            expected_shape = (4 if head else channels, previous_channels,
                              1 if head else 3, 1 if head else 3)
            if layer.name != expected_name or layer.weight.shape != expected_shape:
                raise ValueError(f"invalid layer name or OIHW shape: {layer.name}")
            if layer.bias.shape != (expected_shape[0],):
                raise ValueError("bias count does not match output channels")
            if layer.input_frac != previous_frac:
                raise ValueError("inconsistent activation scale between layers")
            if head != (layer.output_frac is None):
                raise ValueError("only the head omits activation requantization")
            if layer.shift is not None and not -31 <= layer.shift <= 32:
                raise ValueError("requantization shift exceeds widened-reference contract")
            if (layer.weight < -(1 << (self.bits - 1))).any() or (
                layer.weight > (1 << (self.bits - 1)) - 1
            ).any():
                raise ValueError("weight exceeds declared signed width")
            if (layer.bias < INT32_MIN).any() or (layer.bias > INT32_MAX).any():
                raise ValueError("bias exceeds signed32 range")
            previous_channels, previous_frac = expected_shape[0], layer.output_frac
        for bound in accumulator_bounds(self):
            if bound["min"] < INT32_MIN or bound["max"] > INT32_MAX:
                raise ValueError(f"possible signed32 accumulator overflow: {bound['name']}")


def accumulator_bounds(policy: QuantizedPolicy) -> list[dict]:
    """Order-independent bounds for every bias-first partial sum, not just finals."""
    bounds = []
    for index, layer in enumerate(policy.layers):
        maximum = 1 if index == 0 else (1 << policy.bits) - 1
        lower, upper = [], []
        for channel, bias in zip(layer.weight, layer.bias):
            values = [int(value) for value in channel.flat]
            lower.append(int(bias) + maximum * sum(min(value, 0) for value in values))
            upper.append(int(bias) + maximum * sum(max(value, 0) for value in values))
        bounds.append({"name": layer.name, "lower": lower, "upper": upper,
                       "min": min(lower), "max": max(upper), "shift": layer.shift})
    return bounds


def from_float(model, bits: int, calibration: dict, *, provenance: dict | None = None) -> QuantizedPolicy:
    scheme = calibration["schemes"][str(bits)]
    weights = scheme["weight_fracs"]
    activations = scheme["activation_fracs"]
    modules = [*model.convs, model.head]
    if len(weights) != len(modules) or len(activations) != len(model.convs):
        raise ValueError("calibration layer count mismatch")
    layers, input_frac = [], 0
    for index, module in enumerate(modules):
        head = index == len(modules) - 1
        weight_frac = int(weights[index])
        bias = round_away(np.ldexp(module.bias.detach().cpu().numpy().astype(np.float64),
                                  input_frac + weight_frac))
        if (bias < INT32_MIN).any() or (bias > INT32_MAX).any():
            raise ValueError("bias quantization overflow")
        output_frac = None if head else int(activations[index])
        layers.append(QuantLayer(
            "head" if head else f"conv{index}",
            quantize_array(module.weight.detach().cpu().numpy(), weight_frac, bits),
            bias, input_frac, weight_frac, output_frac,
        ))
        input_frac = output_frac
    return QuantizedPolicy(bits, tuple(layers), provenance or {})


def input_codes(inputs) -> np.ndarray:
    inputs = np.asarray(inputs)
    if inputs.ndim != 4 or inputs.shape[1:] != (3, 8, 8) or len(inputs) < 1:
        raise ValueError("inputs must have shape (N,3,8,8) with nonempty batch")
    if inputs.dtype.kind not in "buif" or not np.isfinite(inputs).all() or not (
        (inputs == 0) | (inputs == 1)
    ).all():
        raise ValueError("maze inputs must be exact binary values")
    if not (inputs[:, 1:].sum(axis=(2, 3)) == 1).all():
        raise ValueError("agent and goal planes must be one-hot")
    if (inputs[:, 0:1] * inputs[:, 1:]).any():
        raise ValueError("agent and goal must be on free cells")
    return inputs.astype(np.int64)


def _convolution(inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, first=False):
    kernel = weight.shape[2]
    if kernel == 3:
        padded = np.pad(inputs, ((0, 0), (0, 0), (1, 1), (1, 1)))
        if first:
            padded[:, 0, 0, :] = padded[:, 0, -1, :] = 1
            padded[:, 0, :, 0] = padded[:, 0, :, -1] = 1
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
        columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(len(inputs), 64, -1)
    else:
        columns = inputs.transpose(0, 2, 3, 1).reshape(len(inputs), 64, -1)
    sums = columns @ weight.reshape(weight.shape[0], -1).T + bias
    return sums.transpose(0, 2, 1).reshape(len(inputs), weight.shape[0], 8, 8)


def infer(policy: QuantizedPolicy, inputs, *, trace=False, stats: dict | None = None):
    original = input_codes(inputs)
    current, traces = original, {}
    for index, layer in enumerate(policy.layers):
        accumulator = _convolution(current, layer.weight, layer.bias, first=index == 0)
        minimum, maximum = int(accumulator.min()), int(accumulator.max())
        if minimum < INT32_MIN or maximum > INT32_MAX:
            raise OverflowError("integer accumulator exceeded signed32")
        if trace:
            traces[f"{layer.name}.acc"] = accumulator.copy()
        clipped = 0
        if layer.output_frac is not None:
            shifted = round_shift(accumulator, layer.shift)
            clipped = int((shifted > (1 << policy.bits) - 1).sum())
            current = np.clip(shifted, 0, (1 << policy.bits) - 1)
            if trace:
                traces[f"{layer.name}.act"] = current.copy()
        else:
            current = accumulator
        if stats is not None:
            entry = stats.setdefault(layer.name, {"values": 0, "high_saturations": 0,
                                                  "acc_min": minimum, "acc_max": maximum})
            entry["values"] += int(accumulator.size)
            entry["high_saturations"] += clipped
            entry["acc_min"] = min(entry["acc_min"], minimum)
            entry["acc_max"] = max(entry["acc_max"], maximum)
    indices = original[:, 1].reshape(len(original), 64).argmax(axis=1)
    field = current.reshape(len(original), 4, 64)
    scores = field[np.arange(len(original))[:, None], np.arange(4)[None, :], indices[:, None]]
    if trace:
        traces["scores"] = scores.copy()
        traces["actions"] = scores.argmax(axis=1)
        return scores, traces
    return scores


class IntegerPolicy(torch.nn.Module):
    """CPU evaluator adapter; all network arithmetic remains NumPy int64."""
    def __init__(self, policy: QuantizedPolicy):
        super().__init__()
        self.policy = policy
        self.statistics: dict = {}

    def forward(self, inputs: torch.Tensor):
        if inputs.device.type != "cpu":
            raise ValueError("integer reference is CPU-only")
        scores = infer(self.policy, inputs.detach().numpy(), stats=self.statistics)
        return torch.from_numpy(scores)

    def reset_statistics(self):
        self.statistics.clear()
