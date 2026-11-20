"""Stage two physically distinct precisions without changing the verified CNN logic.

Only the module name and model-specific constant include are substituted. Each
frozen model file is checked against its committed manifest before it is used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).absolute().parents[1]
MODEL_SETS = ("original", "structured")


def model_directory(bits: int, model_set: str = "original") -> Path:
    if model_set not in MODEL_SETS:
        raise ValueError("model_set must be original or structured")
    if bits not in (4, 8):
        raise ValueError("precision must be 4 or 8")
    prefix = "cnn-" if model_set == "original" else "cnn-structured-"
    return ROOT / "model" / f"{prefix}w{bits}a{bits}"


def validate_architecture(manifest: dict) -> None:
    shapes = [[16, 3, 3, 3]] + [[16, 16, 3, 3]] * 7 + [[4, 16, 1, 1]]
    layers = manifest.get("layers", [])
    names = [f"conv{i}" for i in range(8)] + ["head"]
    if (manifest.get("layer_count") != 9 or len(layers) != 9 or
            [layer.get("name") for layer in layers] != names or
            [layer.get("weight_shape") for layer in layers] != shapes or
            [layer.get("bias_shape") for layer in layers] != [[16]] * 8 + [[4]] or
            sum(layer.get("weight_count", 0) for layer in layers) != 16624 or
            sum(layer.get("bias_count", 0) for layer in layers) != 132):
        raise ValueError("model architecture is incompatible with the fixed CNN RTL")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def specialize(source: str, header: str, bits: int) -> str:
    if bits not in (4, 8):
        raise ValueError("precision must be 4 or 8")
    declaration = "module mazebits_cnn #("
    include = '`include "quantization.vh"'
    if source.count(declaration) != 1 or source.count(include) != 1:
        raise ValueError("expected exactly one CNN declaration and precision include")
    for name in ("W_BITS", "A_BITS"):
        values = re.findall(r"localparam integer " + name + r" = (\d+);", header)
        if values != [str(bits)]:
            raise ValueError("precision header does not match requested hardware")
    return source.replace(declaration, f"module mazebits_cnn_w{bits}a{bits} #(").replace(include, header.rstrip())

def stage(destination: Path, model_set: str = "original") -> dict:
    model_directory(8, model_set)  # Validate selection before creating outputs.
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "rtl/mazebits_cnn.v"
    materialized = {}

    def capture(path: Path) -> bytes:
        data = path.read_bytes()
        materialized[path] = hashlib.sha256(data).hexdigest()
        return data

    def decode(data: bytes) -> str:
        return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    source = decode(capture(source_path))
    capture(Path(__file__).absolute())
    generated = []
    # Validate and cache every input before writing any model-dependent artifact.
    models = {}
    identities = []
    for bits in (8, 4):
        model = model_directory(bits, model_set)
        manifest_path = model / "manifest.json"
        manifest = json.loads(capture(manifest_path))
        if manifest["bits"] != bits or manifest["format"] != "mazebits-fixed-v1":
            raise ValueError("wrong model manifest")
        validate_architecture(manifest)
        identities.append({key: manifest["provenance"].get(key) for key in
                           ("source_weights_sha256", "dataset_manifest_sha256", "calibration_sha256")})
        contents = {}
        for filename in ("weights.mem", "biases.mem", "quantization.vh"):
            path = model / filename
            data = capture(path)
            record = manifest["files"][filename]
            if materialized[path] != record["sha256"] or len(data) != record["bytes"]:
                raise ValueError(f"model integrity check failed: {path}")
            contents[filename] = data
        models[bits] = (contents, specialize(source, decode(contents["quantization.vh"]), bits))
    if identities[0] != identities[1] or any(not isinstance(value, str) or
            not re.fullmatch(r"[0-9a-f]{64}", value) for value in identities[0].values()):
        raise ValueError("paired models must share the same floating-point source, dataset, and calibration")
    float_model = ROOT / "model" / ("cnn-fp32" if model_set == "original" else "cnn-structured-fp32")
    float_manifest = json.loads(capture(float_model / "manifest.json"))
    expected_source = hashlib.sha256(capture(float_model / "weights.npz")).hexdigest()
    if (identities[0]["source_weights_sha256"] != expected_source or
            float_manifest.get("weights_sha256") != expected_source or
            float_manifest.get("config") != {"channels": 16, "depth": 8}):
        raise ValueError("selected integer model set does not match its floating-point checkpoint")
    sources = list(materialized)
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    if before != {p.relative_to(ROOT).as_posix(): digest for p, digest in materialized.items()}:
        raise RuntimeError("source changed while staging")
    expected_generated = {}
    for bits, (contents, text) in models.items():
        generated_source = destination / f"mazebits_cnn_w{bits}a{bits}.v"
        source_bytes = text.encode("utf-8")
        generated_source.write_bytes(source_bytes)
        generated.append(generated_source)
        expected_generated[generated_source.name] = hashlib.sha256(source_bytes).hexdigest()
        for stem in ("weights", "biases"):
            target = destination / f"{stem}_w{bits}a{bits}.mem"
            data = contents[f"{stem}.mem"]
            target.write_bytes(data)
            generated.append(target)
            expected_generated[target.name] = hashlib.sha256(data).hexdigest()
    if expected_generated != {p.name: sha(p) for p in generated}:
        raise RuntimeError("generated files changed while staging")
    if before != {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}:
        raise RuntimeError("source changed while staging")
    evidence = {"format": "mazebits-dual-cnn-stage-v1", "precisions": [8, 4], "model_set": model_set,
                "source_checkpoint": identities[0],
                "model_directories": {str(bits): model_directory(bits, model_set).relative_to(ROOT).as_posix() for bits in (8, 4)},
                "source_sha256": before,
                "generated_sha256": {p.name: sha(p) for p in generated}}
    (destination / "dual_cnn_provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    return evidence

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model-set", choices=MODEL_SETS, default="original")
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, args.model_set), indent=2))
