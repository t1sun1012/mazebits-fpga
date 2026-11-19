"""Portable, hash-verified floating-point model bundles; no pickle is required."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import ModelConfig, PolicyCNN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def save_model(model: PolicyCNN, directory: Path, *, provenance: dict) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"model output must be empty: {directory}")
    weights = {name: value.detach().cpu().numpy()
               for name, value in model.state_dict().items()}
    np.savez(directory / "weights.npz", **weights)
    manifest = {
        "format": "mazebits-fp32-v1",
        "config": asdict(model.config),
        "model": model.description(),
        "weights_sha256": sha256(directory / "weights.npz"),
        "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in weights.items()},
        "provenance": provenance,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_model(directory: Path) -> tuple[PolicyCNN, dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "mazebits-fp32-v1":
        raise ValueError("unsupported model bundle format")
    if sha256(directory / "weights.npz") != manifest["weights_sha256"]:
        raise ValueError("model weight hash mismatch")
    model = PolicyCNN(ModelConfig(**manifest["config"]))
    with np.load(directory / "weights.npz", allow_pickle=False) as arrays:
        expected = model.state_dict()
        if set(arrays.files) != set(expected):
            raise ValueError("model tensor names do not match the architecture")
        state = {}
        for name, tensor in expected.items():
            value = arrays[name]
            if value.shape != tuple(tensor.shape) or value.dtype != np.float32:
                raise ValueError(f"invalid tensor shape or dtype: {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite weights: {name}")
            state[name] = torch.from_numpy(value.copy())
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, manifest
