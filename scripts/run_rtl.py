"""Compile and verify the CNN RTL against frozen Step 3 memory vectors.

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

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError(f"unsafe manifest filename: {name}")
        path = directory / name
        if sha(path) != record["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
    return manifest


def words(path: Path) -> list[str]:
    return path.read_text(encoding="ascii").split()


def stage(bits: int, destination: Path) -> dict:
    model = ROOT / f"model/cnn-w{bits}a{bits}"
    vectors = ROOT / f"test_vectors/cnn-w{bits}a{bits}"
    mm = read_manifest(model)
    vm = read_manifest(vectors)
    if vm["source_model"]["manifest_sha256"] != sha(model / "manifest.json"):
        raise ValueError("vectors belong to a different model")
    if vm["case_count"] != 8:
        raise ValueError("testbench requires the eight frozen Step 3 fixtures")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("weights.mem", "biases.mem", "quantization.vh"):
        shutil.copyfile(model / name, destination / name)
    for name in ("inputs.mem", "scores.mem", "actions.mem"):
        shutil.copyfile(vectors / name, destination / name)
    acc_layers = [words(vectors / f"conv{i}_acc.mem") for i in range(8)]
    act_layers = [words(vectors / f"conv{i}_act.mem") for i in range(8)]
    head = words(vectors / "head_acc.mem")
    acc, act = [], []
    for case in range(8):
        for layer in range(8):
            acc.extend(acc_layers[layer][case * 1024 : (case + 1) * 1024])
            act.extend(act_layers[layer][case * 1024 : (case + 1) * 1024])
        acc.extend(head[case * 256 : (case + 1) * 256])
    if len(acc) != 67584 or len(act) != 65536:
        raise ValueError("incorrect frozen trace lengths")
    (destination / "expected_acc.mem").write_text("\n".join(acc) + "\n", encoding="ascii", newline="\n")
    (destination / "expected_act.mem").write_text("\n".join(act) + "\n", encoding="ascii", newline="\n")
    return {
        "model_manifest_sha256": sha(model / "manifest.json"),
        "vectors_manifest_sha256": sha(vectors / "manifest.json"),
        "staged_sha256": {p.name: sha(p) for p in sorted(destination.glob("*.mem"))},
    }


def tool(name: str, vivado_bin: Path | None = None) -> str:
    if vivado_bin:
        candidate = vivado_bin / (name + (".bat" if os.name == "nt" else ""))
        if candidate.is_file():
            return str(candidate.resolve())
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
    destination = args.output_dir.resolve() / f"w{bits}a{bits}"
    evidence = stage(bits, destination)
    # The legacy core test has one authored engine; unrelated board tops are excluded.
    sources = [ROOT / "rtl/mazebits_cnn.v"]
    tb = ROOT / "tb/tb_mazebits_cnn.sv"
    provenance_sources = [*sources, tb, Path(__file__).resolve()]
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/rtl")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per external tool command")
    args = parser.parse_args()
    reports = [verify(args, bits) for bits in args.bits]
    output = args.output_dir.resolve() / "verification.json"
    output.write_text(json.dumps({"models": reports}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: {output}")


if __name__ == "__main__":
    main()
