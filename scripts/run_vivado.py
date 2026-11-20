"""Run Vivado and record reproducible implementation evidence (no board programming).

python scripts/run_vivado.py --bits 8 4 --stage bitstream --vivado C:/Xilinx/Vivado/2024.1/bin/vivado.bat
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

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sources(bits):
    paths = [*sorted((ROOT / "rtl").glob("*.v")),
             ROOT / "scripts/build_vivado.tcl", Path(__file__).resolve(),
             ROOT / "constraints/mazebits_smoke.xdc"]
    paths += [ROOT / f"model/cnn-w{bits}a{bits}" / name
              for name in ("manifest.json", "weights.mem", "biases.mem", "quantization.vh")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def resources(path):
    content = path.read_text(encoding="utf-8")
    result = {}
    for label, key in (("Slice LUTs", "luts"), ("Slice Registers", "registers"),
                       ("Block RAM Tile", "bram_36k_tiles"), ("RAMB36/FIFO", "ramb36"),
                       ("RAMB18", "ramb18"), ("DSPs", "dsps")):
        match = re.search(r"^\|\s*" + re.escape(label) + r"\*?\s*\|\s*([\d.]+)\s*\|",
                          content, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing utilization row {label}: {path}")
        number = float(match.group(1))
        result[key] = int(number) if number.is_integer() else number
    return result

def build(args, bits):
    destination = ROOT / f"runs/vivado/w{bits}a{bits}"
    destination.mkdir(parents=True, exist_ok=True)
    before = sources(bits)
    command = [args.vivado, "-mode", "batch", "-nojournal", "-nolog",
               "-source", str(ROOT / "scripts/build_vivado.tcl"),
               "-tclargs", str(bits), args.stage]
    print("RUN", subprocess.list2cmdline(command), flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace",
                            timeout=args.timeout, check=False)
    output = result.stdout
    (destination / "build.log").write_text(output, encoding="utf-8", newline="\n")
    print(output[-5000:], flush=True)
    if result.returncode or re.search(r"^(?:ERROR|CRITICAL WARNING):", output, re.MULTILINE):
        raise RuntimeError(f"Vivado failed or reported critical warnings; see {destination / 'build.log'}")
    if before != sources(bits):
        raise RuntimeError("Implementation sources changed during build; rerun before publishing evidence")
    report = {
        "format": "mazebits-vivado-implementation-v1", "precision": f"W{bits}A{bits}",
        "status": "pass", "stage": args.stage, "part": "xc7a100tcsg324-1",
        "top": "mazebits_smoke_top", "clock_period_ns": 10.0,
        "scope": "Button/LED top with two selectable fixed fixtures; includes wrapper and synthesis pruning. Full editable/VGA system will require new measurements.",
        "source_sha256": before,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
        "synthesis_resources": resources(destination / "utilization_synth.rpt"),
    }
    if args.stage != "synth":
        summary = dict(line.split("=", 1) for line in
                       (destination / "build_summary.txt").read_text().splitlines() if "=" in line)
        report.update({
            "vivado_version": summary["vivado_version"],
            "setup_slack_ns": float(summary["setup_slack_ns"]),
            "hold_slack_ns": float(summary["hold_slack_ns"]),
            "critical_or_error_drc_count": int(summary["critical_or_error_drc_count"]),
            "total_drc_count": int(summary["total_drc_count"]),
            "warning_drc_count": int(summary["warning_drc_count"]),
            "routed_resources": resources(destination / "utilization_route.rpt"),
        })
    if args.stage == "bitstream":
        path = destination / f"mazebits_smoke_w{bits}a{bits}.bit"
        if not path.is_file():
            raise RuntimeError("Vivado returned without a bitstream")
        report["bitstream"] = {"name": path.name, "bytes": path.stat().st_size,
                               "sha256": sha(path), "programmed_on_board": False}
    report["report_sha256"] = {p.name: sha(p) for p in sorted(destination.glob("*.rpt"))}
    (destination / "implementation.json").write_text(json.dumps(report, indent=2) + "\n",
                                                      encoding="utf-8", newline="\n")
    print(f"PASS W{bits}A{bits}: {destination / 'implementation.json'}", flush=True)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(8, 4), nargs="+", default=[8, 4])
    parser.add_argument("--stage", choices=("synth", "route", "bitstream"), default="route")
    parser.add_argument("--vivado", default=shutil.which("vivado"))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not args.vivado:
        parser.error("Vivado is not on PATH; pass --vivado with its executable or .bat path")
    reports = [build(args, bits) for bits in args.bits]
    print(json.dumps([{"precision": r["precision"], "stage": r["stage"],
                       "status": r["status"]} for r in reports]))

if __name__ == "__main__":
    main()
