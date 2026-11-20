"""Verify navigation controls, physical inputs, VGA, and both real integer CNNs.

python scripts/run_navigation.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
Frozen vectors make the HDL suite independent of Python ML dependencies.
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
from stage_navigation import stage

# Resolve would turn mapped X: paths into UNC paths, which Xilinx .bat launchers
# cannot use as their cmd.exe working directory. Keep the user's absolute drive.
ROOT = Path(__file__).absolute().parents[1]
TESTS = {
    "controller": ("tb_maze_navigation", ["maze_navigation.v"], "PASS_NAVIGATION"),
    "top": ("tb_navigation_top", ["button_conditioner.v", "maze_editor.v", "maze_navigation.v",
                                  "maze_navigation_vga.v", "mazebits_navigation_top.v"], "PASS_NAVIGATION_TOP"),
    "vga": ("tb_navigation_vga", ["maze_navigation_vga.v"], "PASS_NAVIGATION_VGA"),
    "cnn": ("tb_navigation_cnn", ["maze_navigation.v"], "PASS_NAVIGATION_CNN"),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def executable(name, vivado_bin=None):
    if vivado_bin:
        candidate = Path(vivado_bin) / (name + (".bat" if __import__("os").name == "nt" else ""))
        if candidate.is_file():
            return str(candidate.absolute())
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"Cannot locate {name}; use PATH or --vivado-bin")
    return found

def run(command, directory, log_name, timeout):
    print("RUN", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout)
    (directory / log_name).write_text(result.stdout, encoding="utf-8", newline="\n")
    print(result.stdout[-7000:], flush=True)
    if result.returncode or re.search(r"\b(?:FATAL|ERROR):", result.stdout):
        raise RuntimeError(f"Simulation failed; see {directory / log_name}")
    return result.stdout

def verify_vector_bundle(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["format"] != "mazebits-navigation-v1" or manifest["case_count"] != 4 or
        manifest["event_stride_words"] != 11 or manifest["max_steps"] != 64 or
        manifest["cycles_per_decision"] != 3217153):
        raise ValueError("incompatible navigation vector contract")
    hashes = {}
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("vector filename must be local")
        path = directory / name
        if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"navigation vector corruption: {name}")
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in manifest["source_sha256"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"reference source changed; regenerate navigation vectors: {name}")
        hashes[name] = expected
    hashes[(directory / "manifest.json").relative_to(ROOT).as_posix()] = sha(directory / "manifest.json")
    if manifest["scalar_checked_decisions"] != sum(e["steps"] for e in manifest["episodes"]):
        raise ValueError("not all frozen decisions were checked with the scalar oracle")
    return manifest, hashes

def verify(args, name):
    top, modules, marker = TESTS[name]
    directory = args.output_dir.absolute() / name
    directory.mkdir(parents=True, exist_ok=True)
    hdl_files = [ROOT / "rtl" / module for module in modules] + [ROOT / "tb" / (top + ".sv")]
    files = hdl_files + [Path(__file__).absolute(), ROOT / "scripts/stage_navigation.py"]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    generated_hashes = {}
    vectors = None
    if name == "cnn":
        evidence = stage(directory)
        source_hashes.update(evidence["source_sha256"])
        generated_hashes.update(evidence["generated_sha256"])
        hdl_files = [directory / f"mazebits_cnn_w{bits}a{bits}.v" for bits in (8,4)] + hdl_files
        vector_dir = ROOT / "test_vectors/navigation"
        vectors, hashes = verify_vector_bundle(vector_dir)
        source_hashes.update(hashes)
        for filename in vectors["files"]:
            shutil.copyfile(vector_dir / filename, directory / filename)
            generated_hashes[filename] = sha(directory / filename)
    started = time.perf_counter()
    if args.sim == "iverilog":
        compiler, runtime = executable("iverilog"), executable("vvp")
        version = run([compiler,"-V"],directory,"version.log",60)
        run([compiler,"-g2012","-Wall","-s",top,"-o","test.vvp",*map(str,hdl_files)],
            directory,"compile.log",args.timeout)
        output = run([runtime,"test.vvp"],directory,"simulation.log",args.timeout)
    else:
        compiler,elaborator,runtime = (executable(n,args.vivado_bin) for n in ("xvlog","xelab","xsim"))
        version = run([compiler,"-version"],directory,"version.log",60)
        run([compiler,"--sv",*map(str,hdl_files)],directory,"compile.log",args.timeout)
        run([elaborator,top,"--snapshot",top,"--timescale","1ns/1ps"],directory,"elaborate.log",args.timeout)
        output = run([runtime,top,"--runall"],directory,"simulation.log",args.timeout)
    matches = re.findall(r"^"+marker+r"(?:\s+[^\r\n]*)?$",output,re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"Missing or duplicate {marker} marker")
    fields = dict(re.findall(r"(\w+)=([^\s]+)",matches[0]))
    if name == "cnn":
        expected_decisions = sum(e["steps"] for e in vectors["episodes"])
        expected = {"cases":"4","episodes":"8","decisions":str(expected_decisions),
                    "score_words":str(expected_decisions*4),"cycles_per_decision":"3217153",
                    "reset_abort":"1","busy_restart":"1","snapshot":"1","hold_terminal":"1"}
        if any(fields.get(key) != value for key,value in expected.items()):
            raise RuntimeError("Incomplete real-CNN closed-loop verification coverage")
        if len(re.findall(r"^NAV_EVENT ",output,re.MULTILINE)) != expected_decisions:
            raise RuntimeError("Missing per-decision log records")
    if name == "controller":
        expected = {"bounds":"4","legal_moves":"4","wall":"1","loop":"1","goal":"1",
                    "start_goal":"1","limit":"1","snapshot":"1","restart":"1",
                    "stale_done":"1","reset":"1","pacing":"1"}
        if any(fields.get(key) != value for key,value in expected.items()):
            raise RuntimeError("Incomplete controller boundary/protocol coverage")
    if name == "vga":
        frames = int(fields.get("frames","0"))
        if frames < 3 or int(fields.get("pixels","0")) != frames*800*525 or int(fields.get("active_pixels","0")) != frames*640*480:
            raise RuntimeError("Incomplete raster coverage")
        for filename in ("navigation_running.ppm","navigation_result.ppm"):
            if not (directory/filename).is_file():
                raise RuntimeError(f"Missing physical RGB capture: {filename}")
    if source_hashes != {name:sha(ROOT/name) for name in source_hashes}:
        raise RuntimeError("Source changed during verification; rerun")
    if generated_hashes != {name:sha(directory/name) for name in generated_hashes}:
        raise RuntimeError("Staged source, memories, or vectors changed during simulation")
    report = {"test":name,"top":top,"status":"pass","simulator":args.sim,
              "tool_version":version.strip().splitlines()[:8],"pass_marker":matches[0],
              "coverage":fields,"source_sha256":source_hashes,"staged_sha256":generated_hashes,
              "elapsed_wall_seconds":round(time.perf_counter()-started,3),
              "artifacts":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(directory.glob("*.ppm"))}}
    (directory/"verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim","--simulator",dest="sim",choices=("iverilog","xsim"),default="iverilog")
    parser.add_argument("--vivado-bin",type=Path)
    parser.add_argument("--tests",choices=tuple(TESTS),nargs="+",default=list(TESTS))
    parser.add_argument("--output-dir",type=Path,default=ROOT/"runs/navigation")
    parser.add_argument("--timeout",type=int,default=1800,help="Timeout seconds per external command")
    args=parser.parse_args()
    reports=[verify(args,name) for name in args.tests]
    out=args.output_dir.absolute()/"verification.json"
    out.write_text(json.dumps({"format":"mazebits-navigation-verification-v1","tests":reports},indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"PASS: {out}",flush=True)

if __name__ == "__main__":
    main()
