"""Stage two physically distinct precisions without changing the verified CNN logic.

Only the module name and model-specific constant include are substituted. Each
frozen model file is checked against its committed manifest before it is used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).absolute().parents[1]

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

def stage(destination: Path) -> dict:
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "rtl/mazebits_cnn.v"
    source = source_path.read_text(encoding="utf-8")
    sources = [source_path, Path(__file__).absolute()]
    generated = []
    # Validate every input before writing any model-dependent artifact.
    models = {}
    for bits in (8, 4):
        model = ROOT / f"model/cnn-w{bits}a{bits}"
        manifest_path = model / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["bits"] != bits or manifest["format"] != "mazebits-fixed-v1":
            raise ValueError("wrong model manifest")
        sources.append(manifest_path)
        for filename in ("weights.mem", "biases.mem", "quantization.vh"):
            path = model / filename
            record = manifest["files"][filename]
            if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError(f"model integrity check failed: {path}")
            sources.append(path)
        header = (model / "quantization.vh").read_text(encoding="utf-8")
        models[bits] = (model, specialize(source, header, bits))
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    for bits, (model, text) in models.items():
        generated_source = destination / f"mazebits_cnn_w{bits}a{bits}.v"
        generated_source.write_text(text, encoding="utf-8", newline="\n")
        generated.append(generated_source)
        for stem in ("weights", "biases"):
            target = destination / f"{stem}_w{bits}a{bits}.mem"
            shutil.copyfile(model / f"{stem}.mem", target)
            generated.append(target)
    if before != {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}:
        raise RuntimeError("source changed while staging")
    evidence = {"format": "mazebits-dual-cnn-stage-v1", "precisions": [8, 4],
                "source_sha256": before,
                "generated_sha256": {p.name: sha(p) for p in generated}}
    (destination / "dual_cnn_provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    return evidence

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.destination), indent=2))
