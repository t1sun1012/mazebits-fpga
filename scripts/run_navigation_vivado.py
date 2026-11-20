"""Build a selected paired-CNN navigation/VGA top and record source-verified implementation evidence.

python scripts/run_navigation_vivado.py --stage bitstream --vivado C:/Xilinx/Vivado/2024.1/bin/vivado.bat
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from stage_navigation import stage, model_directory

# Keep mapped-drive paths: cmd.exe cannot use UNC working directories for .bat tools.
ROOT = Path(__file__).absolute().parents[1]
RTL = ["button_conditioner.v", "maze_editor.v", "maze_navigation.v", "maze_navigation_vga.v", "mazebits_navigation_top.v", "mazebits_cnn.v"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

MODEL_SETS = ("original", "structured")

def build_paths(model_set="original"):
    if model_set not in MODEL_SETS:
        raise ValueError("model set must be original or structured")
    folder = "navigation" if model_set == "original" else "navigation-structured"
    bit_name = "mazebits_navigation.bit" if model_set == "original" else "mazebits_navigation_structured.bit"
    return ROOT / "runs/vivado" / folder, bit_name

def build_command(vivado, build_stage, model_set):
    build_paths(model_set)
    if build_stage not in ("synth", "route", "bitstream"):
        raise ValueError("invalid build stage")
    return [str(vivado), "-mode", "batch", "-nojournal", "-nolog", "-source",
            str(ROOT / "scripts/build_navigation.tcl"), "-tclargs", build_stage, model_set]

def clear_stale_results(out, bit_name):
    # Exact files only in the explicitly selected output directory. Never
    # recurse or touch the other model set's previously tested bitstream.
    for name in (bit_name, "implementation.json", "build_summary.txt"):
        (Path(out) / name).unlink(missing_ok=True)

def validate_staging(evidence, model_set, out):
    if evidence.get("model_set") != model_set:
        raise ValueError("staged model set does not match requested implementation")
    expected_names = {f"mazebits_cnn_w{bits}a{bits}.v" for bits in (8, 4)}
    expected_names |= {f"{kind}_w{bits}a{bits}.mem" for bits in (8, 4) for kind in ("weights", "biases")}
    if set(evidence["generated_sha256"]) != expected_names:
        raise ValueError("incomplete staged model bundle")
    for bits in (8, 4):
        directory = model_directory(bits, model_set)
        for name in ("manifest.json", "weights.mem", "biases.mem", "quantization.vh"):
            path = directory / name
            key = path.relative_to(ROOT).as_posix()
            if evidence["source_sha256"].get(key) != sha(path):
                raise ValueError("staged model provenance does not match selected bundle")
    for name, digest in evidence["generated_sha256"].items():
        if sha(Path(out) / name) != digest:
            raise ValueError("staged artifact changed before implementation")

def source_hashes(model_set="original"):
    build_paths(model_set)
    files = [ROOT / "rtl" / name for name in RTL]
    files += [ROOT / "scripts/build_navigation.tcl", Path(__file__).absolute(),
              ROOT / "constraints/mazebits_navigation.xdc", ROOT / "scripts/stage_navigation.py"]
    for bits in (8, 4):
        files += [model_directory(bits, model_set) / name for name in ("manifest.json", "quantization.vh", "weights.mem", "biases.mem")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def report_hashes(out, build_stage):
    """Bind only reports produced by this build stage, excluding older/GUI runs."""
    if build_stage not in ("synth", "route", "bitstream"):
        raise ValueError(f"Unknown build stage: {build_stage}")
    names = ["utilization_synth.rpt", "timing_synth.rpt"]
    if build_stage != "synth":
        names += ["utilization_route.rpt", "utilization_hierarchy.rpt",
                  "timing_route.rpt", "drc_route.rpt", "clock_interaction.rpt"]
    return {name: sha(Path(out) / name) for name in sorted(names)}

def utilization(path):
    text = path.read_text(encoding="utf-8")
    result = {}
    for label, key in (("Slice LUTs", "luts"), ("Slice Registers", "registers"),
                       ("Block RAM Tile", "bram_36k_tiles"), ("RAMB36/FIFO", "ramb36"),
                       ("RAMB18", "ramb18"), ("DSPs", "dsps")):
        match = re.search(r"^\|\s*" + re.escape(label) + r"\*?\s*\|\s*([\d.]+)\s*\|",
                          text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing utilization row {label}: {path}")
        number = float(match.group(1))
        result[key] = int(number) if number.is_integer() else number
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("synth", "route", "bitstream"), default="route")
    parser.add_argument("--model-set", choices=MODEL_SETS, default="original")
    parser.add_argument("--vivado", default=shutil.which("vivado"))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not args.vivado:
        parser.error("Vivado is not on PATH; supply --vivado")
    out, bit_name = build_paths(args.model_set)
    out.mkdir(parents=True, exist_ok=True)
    try:
        model_stage = stage(out, model_set=args.model_set)
        validate_staging(model_stage, args.model_set, out)
        clear_stale_results(out, bit_name)
        (out / "build_failure.json").unlink(missing_ok=True)
        staged_names = [f"mazebits_cnn_w{bits}a{bits}.v" for bits in (8,4)] + [f"{kind}_w{bits}a{bits}.mem" for bits in (8,4) for kind in ("weights", "biases")]
        staged_hashes = {name: sha(out / name) for name in staged_names}
        before = source_hashes(args.model_set)
        before.update(model_stage["source_sha256"])
        command = build_command(args.vivado, args.stage, args.model_set)
        print("RUN", subprocess.list2cmdline(command), flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", timeout=args.timeout, check=False)
        output = result.stdout
        (out / "build.log").write_text(output, encoding="utf-8", newline="\n")
        print(output[-5000:], flush=True)
        if result.returncode or re.search(r"^(?:ERROR|CRITICAL WARNING):", output, re.MULTILINE):
            raise RuntimeError(f"Vivado build failed; see {out / 'build.log'}")
        if before != {name: sha(ROOT / name) for name in before} or staged_hashes != {name: sha(out / name) for name in staged_names}:
            raise RuntimeError("Implementation source changed during build; rerun for valid evidence")
        report = {
            "format": "mazebits-navigation-implementation-v1", "status": "pass",
            "stage": args.stage, "model_set": args.model_set, "top": "mazebits_navigation_top", "part": "xc7a100tcsg324-1",
            "clock_period_ns": 10.0, "pixel_clock_enable_hz": 25000000,
            "video_total": [800, 525], "video_active": [640, 480],
            "frame_hz": 25000000 / (800 * 525),
            "scope": "Editable maze, two independently sized W8A8/W4A4 CNN engines, independent navigation controllers, and dual VGA panels.",
            "source_sha256": before, "generated_sha256": staged_hashes, "model_stage": model_stage, "elapsed_wall_seconds": round(time.perf_counter()-started, 3),
            "synthesis_resources": utilization(out / "utilization_synth.rpt"),
        }
        if args.stage != "synth":
            summary = dict(line.split("=", 1) for line in (out / "build_summary.txt").read_text().splitlines() if "=" in line)
            if summary.get("model_set") != args.model_set or summary.get("bitstream_name") != bit_name:
                raise RuntimeError("Vivado summary belongs to a different model set")
            report.update({"vivado_version": summary["vivado_version"],
                           "setup_slack_ns": float(summary["setup_slack_ns"]),
                           "hold_slack_ns": float(summary["hold_slack_ns"]),
                           "routed_resources": utilization(out / "utilization_route.rpt")})
            for key in ("critical_or_error_drc_count", "total_drc_count", "warning_drc_count", "memory_async_drc_count"):
                report[key] = int(summary[key])
        if args.stage == "bitstream":
            path = out / bit_name
            if not path.is_file():
                raise RuntimeError("Vivado returned without creating a bitstream")
            report["bitstream"] = {"name": path.name, "bytes": path.stat().st_size,
                                  "sha256": sha(path), "programmed_on_board": False}
        report["report_sha256"] = report_hashes(out, args.stage)
        (out / "implementation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
        print(f"PASS: {out / 'implementation.json'}", flush=True)
    except Exception as error:
        (out / bit_name).unlink(missing_ok=True)
        (out / "implementation.json").unlink(missing_ok=True)
        (out / "build_failure.json").write_text(json.dumps({
            "status": "failed", "model_set": args.model_set, "stage": args.stage,
            "error": str(error), "usable_bitstream": False,
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
        raise

if __name__ == "__main__":
    main()
