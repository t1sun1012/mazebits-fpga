"""Verify the Step 5 editor, physical-input handling, and VGA output.

python scripts/run_editor.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
python scripts/run_editor.py --sim iverilog
Uses only Python's standard library; does not train or run the CNN.
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
TESTS = {
    "editor": ("tb_maze_editor", ["maze_editor.v"], "PASS_EDITOR"),
    "buttons": ("tb_button_conditioner", ["button_conditioner.v"], "PASS_BUTTONS"),
    "vga": ("tb_maze_vga", ["maze_vga.v"], "PASS_VGA"),
    "top": ("tb_editor_top", ["button_conditioner.v", "maze_editor.v", "maze_vga.v", "mazebits_editor_top.v"], "PASS_EDITOR_TOP"),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def executable(name, vivado_bin=None):
    if vivado_bin:
        result = vivado_bin / (name + (".bat" if os.name == "nt" else ""))
        if result.is_file():
            return str(result.resolve())
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"Cannot locate {name}; put it on PATH or supply --vivado-bin")
    return found

def run(command, out, log, timeout):
    print("RUN", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", timeout=timeout, check=False)
    output = result.stdout
    (out / log).write_text(output, encoding="utf-8", newline="\n")
    print(output[-4000:], flush=True)
    if result.returncode or re.search(r"\b(?:FATAL|ERROR):", output):
        raise RuntimeError(f"Simulation command failed; see {out / log}")
    return output

def verify(args, name):
    top, modules, marker = TESTS[name]
    files = [ROOT / "rtl" / module for module in modules] + [ROOT / "tb" / (top + ".sv"), Path(__file__).resolve()]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    out = args.output_dir.resolve() / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hdl_files = list(map(str, files[:-1]))
    if args.sim == "iverilog":
        compiler, runtime = executable("iverilog"), executable("vvp")
        version = run([compiler, "-V"], out, "version.log", 60)
        run([compiler, "-g2012", "-Wall", "-s", top, "-o", "test.vvp", *hdl_files],
            out, "compile.log", args.timeout)
        output = run([runtime, "test.vvp"], out, "simulation.log", args.timeout)
    else:
        compiler, elaborator, runtime = (executable(n,args.vivado_bin) for n in ("xvlog","xelab","xsim"))
        version = run([compiler, "-version"], out, "version.log", 60)
        run([compiler, "--sv", *hdl_files], out, "compile.log", args.timeout)
        run([elaborator, top, "--snapshot", top, "--timescale", "1ns/1ps"],
            out, "elaborate.log", args.timeout)
        output = run([runtime, top, "--runall"], out, "simulation.log", args.timeout)
    matches = re.findall(r"^" + marker + r"(?:\s+[^\r\n]*)?$", output, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"Missing or duplicate {marker} marker: {out / 'simulation.log'}")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", matches[0]))
    if name == "editor" and any(fields.get(k) != v for k,v in (("wall_cells","64"),("boundaries","8"),("priorities","4"))):
        raise RuntimeError("Editor coverage marker is incomplete")
    if name == "buttons" and (fields.get("width") != "5" or fields.get("thresholds") != "1,4"):
        raise RuntimeError("Button coverage marker is incomplete")
    if name == "vga":
        frames = int(fields.get("frames","0"))
        if frames < 2 or int(fields.get("pixels","0")) != frames * 800 * 525 or int(fields.get("active_pixels","0")) != frames * 640 * 480:
            raise RuntimeError("VGA complete-frame coverage is incomplete")
        for filename in ("baseline_frame.ppm","edited_frame.ppm"):
            if not (out / filename).is_file():
                raise RuntimeError(f"Missing actual HDL-rendered frame {filename}")
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    if before != after:
        raise RuntimeError("Source changed during verification; rerun before publishing evidence")
    report = {
        "test": name, "top": top, "status": "pass", "simulator": args.sim,
        "tool_version": version.strip().splitlines()[:8], "pass_marker": matches[0],
        "coverage": fields, "source_sha256": before,
        "elapsed_wall_seconds": round(time.perf_counter()-started,3),
        "artifacts": {p.name: {"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(out.glob("*.ppm"))},
    }
    (out/"verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim","--simulator",dest="sim",choices=("iverilog","xsim"),default="iverilog")
    parser.add_argument("--vivado-bin",type=Path)
    parser.add_argument("--tests",choices=tuple(TESTS),nargs="+",default=list(TESTS))
    parser.add_argument("--output-dir",type=Path,default=ROOT/"runs/editor")
    parser.add_argument("--timeout",type=int,default=300,help="Timeout seconds per external command")
    args=parser.parse_args()
    reports=[verify(args,name) for name in args.tests]
    out=args.output_dir.resolve()/"verification.json"
    out.write_text(json.dumps({"format":"mazebits-editor-verification-v1","tests":reports},indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"PASS: {out}",flush=True)

if __name__=="__main__":
    main()
