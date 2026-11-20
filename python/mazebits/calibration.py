"""Training-only calibration of power-of-two weight and activation scales."""

import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

from .model import PolicyCNN

MIN_FRAC = -16
MAX_FRAC = 24


def _round_away(values: np.ndarray) -> np.ndarray:
    """Round nearest with exact halfway cases away from zero."""
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def _choose_fraction(values: np.ndarray, *, bits: int, signed: bool) -> dict:
    """Minimize dequantized MSE over five nearby power-of-two scales.

    Ties prefer fewer saturated codes, then the max-fitting exponent, then
    the smaller exponent. Clipping counts refer to rounded codes outside the
    storage interval, rather than merely values beyond its endpoint.
    """
    if bits not in (4, 8):
        raise ValueError("calibration supports only 4-bit or 8-bit storage")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("calibration values must be nonempty and finite")
    if not signed and (array < 0).any():
        raise ValueError("unsigned activation calibration requires nonnegative values")
    lower = -(1 << (bits - 1)) if signed else 0
    upper = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    absolute = np.abs(array)
    maximum = float(absolute.max())
    raw_fit = 0 if maximum == 0 else math.floor(math.log2(upper) - math.log2(maximum))
    max_fit = min(MAX_FRAC, max(MIN_FRAC, raw_fit))
    fractions = sorted({
        min(MAX_FRAC, max(MIN_FRAC, value))
        for value in range(max_fit - 1, max_fit + 4)
    })
    candidates = []
    try:
        with np.errstate(over="raise", invalid="raise"):
            for fraction in fractions:
                rounded = _round_away(np.ldexp(array, fraction))
                clipped_count = int(np.count_nonzero((rounded < lower) | (rounded > upper)))
                quantized = np.clip(rounded, lower, upper)
                restored = np.ldexp(quantized, -fraction)
                mse = float(np.mean(np.square(restored - array), dtype=np.float64))
                if not math.isfinite(mse):
                    raise ValueError("calibration produced non-finite MSE")
                candidates.append({
                    "frac": fraction,
                    "mse": mse,
                    "clipped_count": clipped_count,
                    "clipped_fraction": clipped_count / array.size,
                })
    except FloatingPointError as error:
        raise ValueError("calibration values exceed finite error-analysis range") from error
    selected = min(candidates, key=lambda candidate: (
        candidate["mse"], candidate["clipped_count"],
        abs(candidate["frac"] - max_fit), candidate["frac"],
    ))
    return {
        "bits": bits,
        "signed": signed,
        "code_min": lower,
        "code_max": upper,
        "values": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "max_abs": maximum,
        "p99_9": float(np.percentile(absolute, 99.9)),
        "percentile_basis": "absolute values" if signed else "nonnegative activations",
        "raw_max_fit_frac": raw_fit,
        "max_fit_frac": max_fit,
        "selected_frac": selected["frac"],
        "selected_mse": selected["mse"],
        "selected_clipped_count": selected["clipped_count"],
        "selected_clipped_fraction": selected["clipped_fraction"],
        "candidates": candidates,
    }


def _validate_inputs(training_inputs: np.ndarray) -> None:
    if (
        not isinstance(training_inputs, np.ndarray)
        or training_inputs.ndim != 4
        or tuple(training_inputs.shape[1:]) != (3, 8, 8)
        or len(training_inputs) == 0
        or training_inputs.dtype.kind not in "biuf"
    ):
        raise ValueError("training inputs must be a nonempty numeric array of shape (N, 3, 8, 8)")
    if not np.isfinite(training_inputs).all() or not np.isin(training_inputs, (0, 1)).all():
        raise ValueError("training inputs must contain only finite binary values")
    if (
        not (training_inputs[:, 1].sum(axis=(1, 2)) == 1).all()
        or not (training_inputs[:, 2].sum(axis=(1, 2)) == 1).all()
        or (training_inputs[:, 0] * training_inputs[:, 1]).any()
        or (training_inputs[:, 0] * training_inputs[:, 2]).any()
    ):
        raise ValueError("agent and goal must each be one-hot on free cells")


def calibrate(
    model: PolicyCNN,
    training_inputs: np.ndarray,
    *,
    sample_count: int = 4096,
    seed: int = 311,
) -> dict:
    """Select scales using a deterministic sample of training states only.

    The caller must supply the training partition; no dataset files or validation
    examples are opened here. Activations are collected layer by layer from the
    original FP32 model. This MSE heuristic does not simulate propagated integer
    quantization error, which must be evaluated separately after export.
    """
    _validate_inputs(training_inputs)
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(model, PolicyCNN):
        raise ValueError("calibration requires a PolicyCNN")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("calibration requires an FP32 model on the CPU")
        if not torch.isfinite(parameter).all():
            raise ValueError("calibration requires finite model parameters")
    count = min(sample_count, len(training_inputs))
    indices = np.sort(np.random.default_rng(seed).choice(
        len(training_inputs), size=count, replace=False,
    )).astype("<i8", copy=False)
    sampled = torch.from_numpy(np.ascontiguousarray(training_inputs[indices], dtype=np.float32))
    result = {
        "format": "mazebits-power2-calibration-v1",
        "sampling": {
            "seed": seed,
            "requested_samples": sample_count,
            "actual_samples": count,
            "available_training_states": len(training_inputs),
            "algorithm": "sorted numpy.default_rng(seed).choice without replacement",
            "indices": indices.tolist(),
            "indices_sha256": hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
            "index_byte_format": "little-endian signed int64",
        },
        "method": {
            "rounding": "nearest, halfway away from zero",
            "fraction_bounds": [MIN_FRAC, MAX_FRAC],
            "candidates": "clipped max-fit exponent minus 1 through plus 3, deduplicated",
            "criterion": "minimum dequantized mean-squared error",
            "tie_break": "fewer clipped codes, closest to max-fit, smaller exponent",
            "activations": "FP32 ReLU outputs from original model; no integer error propagation",
            "test_and_validation_inputs": "not supplied or opened by this function",
        },
        "schemes": {
            str(bits): {"weight_fracs": [], "activation_fracs": [], "layers": []}
            for bits in (8, 4)
        },
    }
    training_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            hidden = torch.cat((
                F.pad(sampled[:, :1], (1, 1, 1, 1), value=1.0),
                F.pad(sampled[:, 1:], (1, 1, 1, 1), value=0.0),
            ), dim=1)
            for index, conv in enumerate(model.convs):
                hidden = F.relu(conv(hidden))
                if not torch.isfinite(hidden).all():
                    raise ValueError(f"non-finite calibration activations at convs.{index}")
                weights = conv.weight.detach().numpy()
                activations = hidden.numpy()
                for bits in (8, 4):
                    weight = _choose_fraction(weights, bits=bits, signed=True)
                    activation = _choose_fraction(activations, bits=bits, signed=False)
                    scheme = result["schemes"][str(bits)]
                    scheme["weight_fracs"].append(weight["selected_frac"])
                    scheme["activation_fracs"].append(activation["selected_frac"])
                    scheme["layers"].append({
                        "name": f"convs.{index}",
                        "weight": weight,
                        "activation": activation,
                    })
            for bits in (8, 4):
                weight = _choose_fraction(model.head.weight.detach().numpy(), bits=bits, signed=True)
                scheme = result["schemes"][str(bits)]
                scheme["weight_fracs"].append(weight["selected_frac"])
                scheme["layers"].append({"name": "head", "weight": weight, "activation": None})
    finally:
        for module, training in training_states:
            module.training = training
    return result
