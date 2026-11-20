"""Verify reset, debounce, request capture, and status LEDs using a stub CNN.

Numerical inference verification is in run_rtl.py. No ML dependencies needed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("iverilog", "xsim"), default="iverilog")
    parser.add_argument("--vivado-bin", type=Path)
    parser.add_argument("--bits", type=int, choices=(8, 4), nargs="+", default=[8, 4])
    parser.add_argument("--out", type=Path, default=ROOT / "runs/board-controls")
    args = parser.parse_args()
    summary = {"scope": "board controls with a stub CNN, not numerical inference", "models": {}}
    sources = [ROOT/"tb/tb_mazebits_smoke_top.sv", ROOT/"rtl/mazebits_smoke_top.v"]
    summary["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def executable(name: str) -> str:
        if args.vivado_bin:
            suffix = ".bat" if sys.platform == "win32" else ""
            result = args.vivado_bin / (name + suffix)
            if result.is_file():
                return str(result.resolve())
        result = shutil.which(name)
        if not result:
            raise FileNotFoundError(f"Cannot locate {name}")
        return result

    for bits in args.bits:
        out = args.out.resolve() / f"w{bits}a{bits}"
        out.mkdir(parents=True, exist_ok=True)
        include = ROOT / f"model/cnn-w{bits}a{bits}"
        commands = []
        if args.sim == "iverilog":
            commands = [
                [executable("iverilog"), "-g2012", "-I", str(include), "-s", "tb_mazebits_smoke_top", "-o", "board.vvp", *map(str,sources)],
                [executable("vvp"), "board.vvp"],
            ]
        else:
            commands = [
                [executable("xvlog"), "--sv", "-i", str(include), *map(str,sources)],
                [executable("xelab"), "tb_mazebits_smoke_top", "-s", "board_controls"],
                [executable("xsim"), "board_controls", "-runall"],
            ]
        combined = ""
        for index, command in enumerate(commands):
            print("RUN", subprocess.list2cmdline(command), flush=True)
            result = subprocess.run(command, cwd=out, capture_output=True, text=True, errors="replace", timeout=120)
            text = result.stdout + result.stderr
            (out/f"command-{index}.log").write_text(text, encoding="utf-8", newline="\n")
            print(text[-3000:], flush=True)
            if result.returncode:
                raise RuntimeError(f"Simulator command failed: {command[0]}")
            combined += text
        marker = f"PASS_BOARD bits={bits} requests=3"
        if marker not in combined or "FATAL" in combined.upper():
            raise RuntimeError(f"Missing board-control PASS marker for W{bits}A{bits}")
        summary["models"][f"w{bits}a{bits}"] = {"status": "passed", "accepted_requests": 3, "simulator": args.sim}
    args.out.resolve().mkdir(parents=True, exist_ok=True)
    (args.out.resolve()/"verification.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8", newline="\n")
    print("Board control verification passed", flush=True)


if __name__ == "__main__":
    main()
