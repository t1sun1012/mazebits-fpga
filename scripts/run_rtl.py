"""Compile and verify the CNN RTL against a selected frozen model/vector bundle.

No ML packages are needed. Run from any directory:
  python scripts/run_rtl.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
  python scripts/run_rtl.py --sim iverilog
Use --smoke for the first fixture plus repeat/reset/protocol checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

# Preserve mapped drive paths: Xilinx batch launchers cannot use UNC cwd paths.
ROOT = Path(__file__).absolute().parents[1]
MODEL_SETS = ("original", "structured")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(directory: Path, captured: dict[str, bytes] | None = None) -> dict:
    manifest_bytes = ((directory / "manifest.json").read_bytes()
                      if captured is None else captured["manifest.json"])
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    for name, record in manifest["files"].items():
        if not name or name in (".", "..") or "/" in name or "\\" in name or Path(name).name != name:
            raise ValueError(f"unsafe manifest filename: {name}")
        path = directory / name
        contents = path.read_bytes() if captured is None else captured[name]
        if hashlib.sha256(contents).hexdigest() != record["sha256"] or len(contents) != record["bytes"]:
            raise ValueError(f"hash mismatch: {path}")
    return manifest


def words(path: Path) -> list[str]:
    return path.read_text(encoding="ascii").split()


def model_paths(bits: int, model_set: str = "original") -> tuple[Path, Path]:
    if bits not in (4, 8):
        raise ValueError("precision must be 4 or 8")
    if model_set not in MODEL_SETS:
        raise ValueError("model_set must be original or structured")
    stem = "cnn-" + ("structured-" if model_set == "structured" else "") + f"w{bits}a{bits}"
    return ROOT / "model" / stem, ROOT / "test_vectors" / stem


def checked_words(path: Path, count: int, bits: int, captured: bytes | None = None) -> list[str]:
    values = words(path) if captured is None else captured.decode("ascii").split()
    if len(values) != count:
        raise ValueError(f"incorrect frozen trace length: {path.name}")
    for value in values:
        if not re.fullmatch(r"[0-9a-fA-F]+", value) or int(value, 16) >= 1 << bits:
            raise ValueError(f"invalid memory word: {path.name}")
    return values


def check_staged_inputs(evidence: dict, destination: Path) -> None:
    if evidence["bundle_source_sha256"] != {
        name: sha(ROOT / name) for name in evidence["bundle_source_sha256"]
    }:
        raise RuntimeError("model or reference bundle changed during verification")
    if evidence["staged_sha256"] != {
        name: sha(destination / name) for name in evidence["staged_sha256"]
    }:
        raise RuntimeError("staged memories or precision header changed during verification")


def stage(bits: int, destination: Path, model_set: str = "original") -> dict:
    model, vectors = model_paths(bits, model_set)
    model_files = {"weights.mem", "biases.mem", "quantization.vh"}
    vector_files = {"inputs.mem", "scores.mem", "actions.mem", "head_acc.mem"} | {
        f"conv{i}_{kind}.mem" for i in range(8) for kind in ("acc", "act")}
    sources = [model / "manifest.json", vectors / "manifest.json",
               *(model / name for name in sorted(model_files)),
               *(vectors / name for name in sorted(vector_files))]
    # Every validation, transformation, and simulator copy uses these captured
    # bytes. Hashing the same bytes prevents an edit/read/hash race (including
    # a file temporarily changed and restored during staging).
    captured = {path: path.read_bytes() for path in sources}
    source_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(contents).hexdigest()
                     for path, contents in captured.items()}
    mm = read_manifest(model, {p.name: b for p, b in captured.items() if p.parent == model})
    vm = read_manifest(vectors, {p.name: b for p, b in captured.items() if p.parent == vectors})
    if mm["format"] != "mazebits-fixed-v1" or vm["format"] != "mazebits-fixed-vectors-v1":
        raise ValueError("unsupported model or vector format")
    if mm["bits"] != bits or vm["bits"] != bits:
        raise ValueError("selected precision does not match model/vector bundle")
    identities = {"manifest_sha256": "manifest.json", "weights_sha256": "weights.mem",
                  "biases_sha256": "biases.mem", "header_sha256": "quantization.vh"}
    if any(vm["source_model"][key] != source_hashes[(model / filename).relative_to(ROOT).as_posix()] for key, filename in identities.items()):
        raise ValueError("vectors belong to a different model")
    if vm["case_count"] != 8:
        raise ValueError("testbench requires eight frozen fixtures")
    expected_shapes = [[16, 3, 3, 3], *([[16, 16, 3, 3]] * 7), [4, 16, 1, 1]]
    if mm["layer_count"] != 9 or [layer["weight_shape"] for layer in mm["layers"]] != expected_shapes:
        raise ValueError("model architecture does not match the fixed CNN testbench")
    if set(mm["files"]) != model_files or set(vm["files"]) != vector_files:
        raise ValueError("model/vector files do not match the testbench contract")
    header = captured[model / "quantization.vh"].decode("utf-8")
    for name, expected in {"W_BITS": bits, "A_BITS": bits, "BIAS_BITS": 32,
                           "ACC_BITS": 32, "N_LAYERS": 9, "GRID_SIZE": 8,
                           "WEIGHT_WORDS": 16624, "BIAS_WORDS": 132}.items():
        if re.findall(r"localparam integer " + name + r" = (\d+);", header) != [str(expected)]:
            raise ValueError(f"precision header disagrees with testbench: {name}")
    checked_words(model / "weights.mem", 16624, bits, captured[model / "weights.mem"])
    checked_words(model / "biases.mem", 132, 32, captured[model / "biases.mem"])
    checked_words(vectors / "inputs.mem", 8 * 3 * 64, 1, captured[vectors / "inputs.mem"])
    checked_words(vectors / "scores.mem", 8 * 4, 32, captured[vectors / "scores.mem"])
    checked_words(vectors / "actions.mem", 8, 2, captured[vectors / "actions.mem"])
    acc_layers = [checked_words(vectors / f"conv{i}_acc.mem", 8 * 1024, 32, captured[vectors / f"conv{i}_acc.mem"]) for i in range(8)]
    act_layers = [checked_words(vectors / f"conv{i}_act.mem", 8 * 1024, bits, captured[vectors / f"conv{i}_act.mem"]) for i in range(8)]
    head = checked_words(vectors / "head_acc.mem", 8 * 256, 32, captured[vectors / "head_acc.mem"])
    acc, act = [], []
    for case in range(8):
        for layer in range(8):
            acc.extend(acc_layers[layer][case * 1024 : (case + 1) * 1024])
            act.extend(act_layers[layer][case * 1024 : (case + 1) * 1024])
        acc.extend(head[case * 256 : (case + 1) * 256])
    # Reject concurrent edits before writing outputs; keep checking again
    # after staging and after simulation. Copies use captured, validated bytes.
    if source_hashes != {name: sha(ROOT / name) for name in source_hashes}:
        raise RuntimeError("model or reference bundle changed during staging")
    staged_bytes = {name: captured[model / name] for name in model_files}
    staged_bytes.update({name: captured[vectors / name]
                         for name in ("inputs.mem", "scores.mem", "actions.mem")})
    staged_bytes["expected_acc.mem"] = ("\n".join(acc) + "\n").encode("ascii")
    staged_bytes["expected_act.mem"] = ("\n".join(act) + "\n").encode("ascii")
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in sorted(staged_bytes.items()):
        (destination / name).write_bytes(contents)
    evidence = {
        "model_set": model_set,
        "model_directory": model.relative_to(ROOT).as_posix(),
        "vectors_directory": vectors.relative_to(ROOT).as_posix(),
        "model_manifest_sha256": source_hashes[(model / "manifest.json").relative_to(ROOT).as_posix()],
        "vectors_manifest_sha256": source_hashes[(vectors / "manifest.json").relative_to(ROOT).as_posix()],
        "bundle_source_sha256": source_hashes,
        "staged_sha256": {name: hashlib.sha256(contents).hexdigest()
                           for name, contents in sorted(staged_bytes.items())},
    }
    check_staged_inputs(evidence, destination)
    return evidence


def tool(name: str, vivado_bin: Path | None = None) -> str:
    if vivado_bin:
        candidate = vivado_bin / (name + (".bat" if os.name == "nt" else ""))
        if candidate.is_file():
            return str(candidate.absolute())
        raise FileNotFoundError(candidate)
    result = shutil.which(name)
    if not result:
        raise FileNotFoundError(f"{name} is not on PATH; specify --vivado-bin for XSim")
    return result


def run(command: list[str], cwd: Path, log_name: str, timeout: int) -> str:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace",
                            timeout=timeout, check=False)
    (cwd / log_name).write_text(result.stdout, encoding="utf-8", newline="\n")
    print(result.stdout[-6000:], flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.2f}s", flush=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}; see {cwd / log_name}")
    return result.stdout


def verify(args, bits: int) -> dict:
    destination = args.output_dir.absolute() / f"w{bits}a{bits}"
    evidence = stage(bits, destination, args.model_set)
    # The legacy core test has one authored engine; unrelated board tops are excluded.
    sources = [ROOT / "rtl/mazebits_cnn.v"]
    tb = ROOT / "tb/tb_mazebits_cnn.sv"
    provenance_sources = [*sources, tb, Path(__file__).absolute()]
    source_hashes = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in provenance_sources
    }
    count = 1 if args.smoke else 8
    started = time.perf_counter()
    if args.sim == "iverilog":
        compiler, runtime = tool("iverilog"), tool("vvp")
        version = run([compiler, "-V"], destination, "version.log", 60)
        run([compiler, "-g2012", "-Wall", "-s", "tb_mazebits_cnn", "-I", str(destination),
             "-o", "simulation.vvp", *map(str, sources), str(tb)],
            destination, "compile.log", args.timeout)
        output = run([runtime, "simulation.vvp", f"+CASES={count}"],
                     destination, "simulation.log", args.timeout)
    else:
        compiler, elaborator, runtime = (tool(n, args.vivado_bin) for n in ("xvlog", "xelab", "xsim"))
        version = run([compiler, "-version"], destination, "version.log", 60)
        run([compiler, "--sv", "--include", str(destination), *map(str, sources), str(tb)],
            destination, "compile.log", args.timeout)
        run([elaborator, "tb_mazebits_cnn", "--snapshot", "maze_tb", "--timescale", "1ns/1ps"],
            destination, "elaborate.log", args.timeout)
        sim_command = [runtime, "maze_tb", "--runall"]
        if args.smoke:
            sim_command.extend(["--testplusarg", "SMOKE"])
        output = run(sim_command, destination, "simulation.log", args.timeout)
    match = re.search(r"RTL_PASS cases=(\d+) words=(\d+) repeat_cases=(\d+) cycles=(\d+) "
                      r"reset_abort=1 busy_start_ignored=1", output)
    if "RTL_ARITHMETIC_PASS rounding_cases=13 argmax_cases=3" not in output:
        raise RuntimeError("arithmetic boundary checks did not pass")
    if not match or re.search(r"\b(?:FATAL|ERROR):", output):
        raise RuntimeError(f"simulator did not report a clean pass: {destination / 'simulation.log'}")
    values = list(map(int, match.groups()))
    if values[:3] != [count, count * 16645, 1]:
        raise RuntimeError("incorrect verification coverage")
    cases = [
        dict(zip(("case", "repeated", "cycles", "traces", "words"), map(int, row)))
        for row in re.findall(
            r"RTL_CASE case=(\d+) repeated=(\d+) cycles=(\d+) traces=(\d+) words=(\d+)", output)
    ]
    if source_hashes != {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in provenance_sources
    }:
        raise RuntimeError("RTL, testbench, or runner source changed during verification; rerun")
    check_staged_inputs(evidence, destination)
    report = {
        "format": "mazebits-rtl-verification-v1",
        "precision": f"W{bits}A{bits}",
        "status": "pass",
        "simulator": args.sim,
        "tool_version": version.strip().splitlines()[:8],
        "cases": values[0],
        "compared_words": values[1],
        "repeat_cases": values[2],
        "cycles_per_decision": values[3],
        "clock_period_ns": 10,
        "simulated_latency_ms_at_100mhz": values[3] / 100000,
        "timing_note": "Cycle simulation at a nominal 100 MHz; not proof of routed timing closure.",
        "protocol_checks": ["reset abort/restart", "ignore busy start", "snapshot inputs",
                            "ignore start on finish edge", "one-cycle done", "hold busy/idle outputs", "repeat inference"],
        "arithmetic_boundary_cases": 13,
        "argmax_signed_and_tie_cases": 3,
        "case_results": cases,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
        "source_sha256": source_hashes,
        **evidence,
    }
    (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", "--simulator", dest="sim", choices=("iverilog", "xsim"), default="iverilog")
    parser.add_argument("--bits", type=int, choices=(4, 8), nargs="+", default=[8, 4])
    parser.add_argument("--vivado-bin", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-set", choices=MODEL_SETS, default="original")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per external tool command")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "runs" / ("rtl-structured" if args.model_set == "structured" else "rtl")
    reports = [verify(args, bits) for bits in args.bits]
    output = args.output_dir.absolute() / "verification.json"
    output.write_text(json.dumps({"models": reports}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: {output}")


if __name__ == "__main__":
    main()
