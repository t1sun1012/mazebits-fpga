"""Hash-verified FPGA memory bundles for the integer CNN reference."""

from math import prod
from copy import deepcopy
from pathlib import Path
import json
import re

import numpy as np

from .fixed_point import QuantLayer, QuantizedPolicy, accumulator_bounds
from .model_io import sha256, write_json

_FORMAT = "mazebits-fixed-v1"
_FILES = ("weights.mem", "biases.mem", "quantization.vh")
_CONTRACT = {
    "input": {"bits": 1, "frac": 0, "shape": [3, 8, 8],
              "channels": ["walls", "agent", "goal"], "layout": "CHW",
              "first_padding": "walls=1; agent=0; goal=0"},
    "weight_layout": "OIHW; layers concatenated in manifest order",
    "activation_signed": False,
    "hidden_padding": 0,
    "bias_bits": 32,
    "accumulator_bits": 32,
    "requant_temporary_bits": 64,
    "rounding": "nearest; ties away from zero",
    "hidden_output": "ReLU followed by unsigned saturation",
    "head_output": "signed raw accumulator; no requantization or ReLU",
    "argmax_ties": "lowest action index",
    "action_order": ["UP", "RIGHT", "DOWN", "LEFT"],
}


def _int(value, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be an integer"
                         + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _storage(weight_words: int, bias_words: int, bits: int) -> dict:
    weight_bits = weight_words * bits
    bias_bits = bias_words * 32
    return {
        "weight_bits": weight_bits, "bias_bits": bias_bits,
        "weight_bytes": (weight_bits + 7) // 8,
        "bias_bytes": bias_words * 4,
        "total_bits": weight_bits + bias_bits,
        "total_bytes": (weight_bits + bias_bits + 7) // 8,
        "meaning": "packed parameter payload; excludes metadata and FPGA allocation overhead",
    }


def _layer_records(policy: QuantizedPolicy) -> list[dict]:
    layers, weight_offset, bias_offset = [], 0, 0
    for layer in policy.layers:
        weight_count = int(layer.weight.size)
        bias_count = int(layer.bias.size)
        acc_frac = int(layer.input_frac + layer.weight_frac)
        shift = None if layer.output_frac is None else acc_frac - int(layer.output_frac)
        layers.append({
            "name": layer.name,
            "weight_shape": list(layer.weight.shape),
            "bias_shape": list(layer.bias.shape),
            "weight_offset": weight_offset, "weight_count": weight_count,
            "bias_offset": bias_offset, "bias_count": bias_count,
            "input_frac": int(layer.input_frac),
            "weight_frac": int(layer.weight_frac),
            "output_frac": None if layer.output_frac is None else int(layer.output_frac),
            "accumulator_frac": acc_frac, "shift": shift,
        })
        weight_offset += weight_count
        bias_offset += bias_count
    return layers


def _header(manifest: dict) -> str:
    rows = [
        "// Generated MazeBits integer-policy constants. Include inside a module.",
        "// Activations are unsigned; zero-extend before signed multiplication.",
        "// Real value = integer * 2**(-FRAC). Negative SHIFT means left shift.",
        "// Head OUTPUT_FRAC is its raw accumulator fraction; HAS_RELU is zero.",
    ]
    constants = {
        "W_BITS": manifest["bits"], "A_BITS": manifest["bits"],
        "INPUT_BITS": 1, "INPUT_FRAC": 0, "BIAS_BITS": 32, "ACC_BITS": 32,
        "REQUANT_BITS": 64, "N_LAYERS": len(manifest["layers"]), "GRID_SIZE": 8,
        "WEIGHT_WORDS": sum(layer["weight_count"] for layer in manifest["layers"]),
        "BIAS_WORDS": sum(layer["bias_count"] for layer in manifest["layers"]),
    }
    rows.extend(f"localparam integer {name} = {value};" for name, value in constants.items())
    for index, layer in enumerate(manifest["layers"]):
        shape = layer["weight_shape"]
        values = {
            "WEIGHT_OFFSET": layer["weight_offset"], "WEIGHT_COUNT": layer["weight_count"],
            "BIAS_OFFSET": layer["bias_offset"], "BIAS_COUNT": layer["bias_count"],
            "C_IN": shape[1], "C_OUT": shape[0], "KERNEL": shape[2],
            "INPUT_FRAC": layer["input_frac"], "WEIGHT_FRAC": layer["weight_frac"],
            "ACC_FRAC": layer["accumulator_frac"],
            "OUTPUT_FRAC": (layer["accumulator_frac"] if layer["output_frac"] is None
                            else layer["output_frac"]),
            "SHIFT": 0 if layer["shift"] is None else layer["shift"],
            "HAS_RELU": int(layer["output_frac"] is not None),
        }
        rows.append(f"// Layer {index}: {layer['name']}")
        rows.extend(f"localparam integer L{index}_{name} = {value};"
                    for name, value in values.items())
    return "\n".join(rows) + "\n"


def _write_mem(path: Path, values: np.ndarray, bits: int) -> None:
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    text = "".join(f"{int(value) & mask:0{digits}x}\n" for value in values.reshape(-1))
    path.write_text(text, encoding="ascii", newline="\n")


def _read_mem(path: Path, bits: int, expected_words: int) -> np.ndarray:
    try:
        rows = path.read_bytes().decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"non-ASCII memory file: {path.name}") from error
    if len(rows) != expected_words:
        raise ValueError(f"incorrect word count: {path.name}")
    digits = (bits + 3) // 4
    pattern = re.compile(rf"[0-9a-fA-F]{{{digits}}}")
    values = []
    for index, row in enumerate(rows):
        if pattern.fullmatch(row) is None:
            raise ValueError(f"invalid hexadecimal word in {path.name} at {index}")
        value = int(row, 16)
        if value >= (1 << bits):
            raise ValueError(f"word exceeds its declared width: {path.name}")
        values.append(value - (1 << bits) if value & (1 << (bits - 1)) else value)
    return np.asarray(values, dtype=np.int64)


def save_quantized(policy: QuantizedPolicy, directory: Path) -> dict:
    """Export canonical readmemh words and the arithmetic/shape contract."""
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"quantized output must be new or empty: {directory}")
    # Constructing the policy validates ranges and accumulator safety. Recheck
    # here so a caller cannot export arrays that were mutated after construction.
    checked = QuantizedPolicy(bits=policy.bits, layers=policy.layers,
                              provenance=policy.provenance)
    records = _layer_records(checked)
    weights = np.concatenate([layer.weight.reshape(-1) for layer in checked.layers])
    biases = np.concatenate([layer.bias.reshape(-1) for layer in checked.layers])
    manifest = {
        "format": _FORMAT, "bits": int(checked.bits), "layer_count": len(records),
        "contract": deepcopy(_CONTRACT), "layers": records,
        "accumulator_bounds": accumulator_bounds(checked),
        "logical_storage": _storage(len(weights), len(biases), checked.bits),
        "provenance": checked.provenance,
    }
    # Reject unserializable/non-finite provenance before writing a partial bundle.
    json.dumps(manifest, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=True)
    _write_mem(directory / "weights.mem", weights, checked.bits)
    _write_mem(directory / "biases.mem", biases, 32)
    (directory / "quantization.vh").write_text(_header(manifest), encoding="ascii", newline="\n")
    manifest["files"] = {
        name: {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
        for name in _FILES
    }
    manifest["files"]["weights.mem"].update(
        {"words": len(weights), "word_bits": checked.bits, "signed": True})
    manifest["files"]["biases.mem"].update(
        {"words": len(biases), "word_bits": 32, "signed": True})
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_quantized(directory: Path) -> QuantizedPolicy:
    """Load the exported memory words as the authoritative integer model."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != _FORMAT:
        raise ValueError("unsupported quantized bundle format")
    bits = _int(manifest.get("bits"), "bits")
    if bits not in (4, 8):
        raise ValueError("weight and activation precision must be 4 or 8")
    if manifest.get("contract") != _CONTRACT:
        raise ValueError("quantized arithmetic contract does not match this implementation")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_FILES):
        raise ValueError("bundle must name exactly the fixed memory/header files")
    for name in _FILES:
        entry = files[name]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid file metadata: {name}")
        path = directory / name  # No manifest-supplied path is used.
        if _int(entry.get("bytes"), "file bytes", minimum=0) != path.stat().st_size:
            raise ValueError(f"file length mismatch: {name}")
        if sha256(path) != entry.get("sha256"):
            raise ValueError(f"file hash mismatch: {name}")
    for name, width in (("weights.mem", bits), ("biases.mem", 32)):
        if files[name].get("word_bits") != width or files[name].get("signed") is not True:
            raise ValueError(f"invalid memory word format: {name}")
        _int(files[name].get("words"), "memory words", minimum=1)
    records = manifest.get("layers")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("bundle needs convolution layers followed by a head")
    if _int(manifest.get("layer_count"), "layer_count", minimum=2) != len(records):
        raise ValueError("layer count mismatch")
    weights = _read_mem(directory / "weights.mem", bits, files["weights.mem"]["words"])
    biases = _read_mem(directory / "biases.mem", 32, files["biases.mem"]["words"])
    layers, weight_offset, bias_offset = [], 0, 0
    for index, entry in enumerate(records):
        if not isinstance(entry, dict):
            raise ValueError("invalid layer record")
        expected_name = "head" if index == len(records) - 1 else f"conv{index}"
        if entry.get("name") != expected_name:
            raise ValueError("layer names/order do not match the architecture")
        shape = entry.get("weight_shape")
        if not isinstance(shape, list) or len(shape) != 4:
            raise ValueError("weights need an OIHW shape")
        shape = [_int(value, "weight dimension", minimum=1) for value in shape]
        bias_shape = entry.get("bias_shape")
        if (not isinstance(bias_shape, list) or len(bias_shape) != 1
                or _int(bias_shape[0], "bias dimension", minimum=1) != shape[0]):
            raise ValueError("bias shape does not match output channels")
        wc, bc = prod(shape), shape[0]
        expected_fields = {"weight_offset": weight_offset, "weight_count": wc,
                           "bias_offset": bias_offset, "bias_count": bc}
        for name, expected in expected_fields.items():
            if _int(entry.get(name), name, minimum=0) != expected:
                raise ValueError(f"noncontiguous or inconsistent {name}")
        if weight_offset + wc > len(weights) or bias_offset + bc > len(biases):
            raise ValueError("layer extends beyond the memory file")
        input_frac = _int(entry.get("input_frac"), "input_frac")
        weight_frac = _int(entry.get("weight_frac"), "weight_frac")
        output_frac = entry.get("output_frac")
        if output_frac is not None:
            output_frac = _int(output_frac, "output_frac")
        acc_frac = input_frac + weight_frac
        shift = None if output_frac is None else acc_frac - output_frac
        if _int(entry.get("accumulator_frac"), "accumulator_frac") != acc_frac:
            raise ValueError("incorrect accumulator fraction")
        if entry.get("shift") != shift or (shift is not None and type(entry["shift"]) is not int):
            raise ValueError("incorrect requantization shift")
        layers.append(QuantLayer(
            name=expected_name,
            weight=weights[weight_offset:weight_offset + wc].reshape(shape),
            bias=biases[bias_offset:bias_offset + bc],
            input_frac=input_frac, weight_frac=weight_frac, output_frac=output_frac,
        ))
        weight_offset += wc
        bias_offset += bc
    if weight_offset != len(weights) or bias_offset != len(biases):
        raise ValueError("unreferenced words at the end of a memory file")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    policy = QuantizedPolicy(bits=bits, layers=tuple(layers), provenance=provenance)
    if manifest.get("accumulator_bounds") != accumulator_bounds(policy):
        raise ValueError("recorded accumulator proof does not match the memory words")
    if manifest.get("logical_storage") != _storage(len(weights), len(biases), bits):
        raise ValueError("logical storage metadata does not match the memory words")
    if (directory / "quantization.vh").read_bytes() != _header(manifest).encode("ascii"):
        raise ValueError("generated header disagrees with the manifest")
    return policy
