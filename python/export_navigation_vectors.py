"""Freeze complete trajectories from the independently cross-checked integer CNNs.

Every score is checked against the slow scalar Python-integer implementation.
The vectors are simulation inputs, never route assistance in the FPGA design.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.fixed_point import infer
from mazebits.maze import step
from mazebits.quantized_io import load_quantized
from mazebits.scalar_fixed import infer_scalar

ROOT = Path(__file__).resolve().parents[1]
CASES = ("open", "detour", "blocked", "at-goal")
CYCLES = 3217153
MAX_STEPS = 64
STRIDE = 11
STATUS = {"run": 1, "success": 2, "blocked": 3, "loop": 4, "timeout": 5}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cell(position):
    return position[1] * 8 + position[0]

def words(path, values, digits=8):
    path.write_text("".join(f"{int(value) & ((1 << (digits*4))-1):0{digits}x}\n" for value in values),
                    encoding="ascii", newline="\n")

def export(destination: Path, *, scalar=True):
    destination.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__).resolve(), *[ROOT / "python/mazebits" / name for name in
                    ("scalar_fixed.py", "fixed_point.py", "quantized_io.py", "maze.py", "encoding.py", "examples.py")]]
    for bits in (8,4):
        source_files.extend(ROOT / f"model/cnn-w{bits}a{bits}" / name for name in
                            ("manifest.json", "quantization.vh", "weights.mem", "biases.mem"))
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}
    scenarios = []
    for name in CASES:
        maze = EXAMPLES[name]
        wall_bits = sum(1 << cell(wall) for wall in maze.walls)
        scenarios.append({"name": name, "walls_hex": f"{wall_bits:016x}",
                          "start": cell(maze.start), "goal": cell(maze.goal)})
    words(destination / "walls.mem", [int(s["walls_hex"],16) for s in scenarios],16)
    words(destination / "starts.mem", [s["start"] for s in scenarios],2)
    words(destination / "goals.mem", [s["goal"] for s in scenarios],2)
    episodes = []
    scalar_checks = 0
    for bits in (8,4):
        policy = load_quantized(ROOT / f"model/cnn-w{bits}a{bits}")
        packed, finals = [], []
        for name in CASES:
            maze = EXAMPLES[name]
            position, visited = maze.start, {maze.start}
            trail = 1 << cell(position)
            outcome = "success" if position == maze.goal else "run"
            events = []
            for index in range(MAX_STEPS):
                if outcome != "run":
                    break
                encoded = np.asarray(encode_state(maze, position), dtype=np.int64).reshape(1,3,8,8)
                scores = infer(policy, encoded)[0]
                if scalar:
                    scalar_scores, _ = infer_scalar(policy, encoded)
                    if not np.array_equal(scores, scalar_scores[0]):
                        raise AssertionError(f"independent scalar disagreement: W{bits} {name} {index}")
                    scalar_checks += 1
                action = int(scores.argmax())
                moved = step(maze, position, action)
                position = moved.position
                trail |= 1 << cell(position)
                if moved.done:
                    outcome = "success"
                elif moved.blocked:
                    outcome = "blocked"
                elif position in visited:
                    outcome = "loop"
                elif index + 1 == MAX_STEPS:
                    outcome = "timeout"
                visited.add(position)
                events.append({"action": action, "position": cell(position), "trail_hex": f"{trail:016x}",
                               "status": STATUS[outcome], "steps": index+1, "total_cycles": (index+1)*CYCLES,
                               "scores": [int(v) for v in scores]})
            for event in events:
                trail_value = int(event["trail_hex"],16)
                packed.extend([event["action"],event["position"],trail_value & 0xffffffff,trail_value >> 32,
                               event["status"],event["steps"],event["total_cycles"],*event["scores"]])
            packed.extend([0] * ((MAX_STEPS-len(events))*STRIDE))
            finals.extend([STATUS[outcome],len(events),cell(position),trail & 0xffffffff,trail >> 32,len(events)*CYCLES])
            episodes.append({"bits": bits, "case": name, "outcome": outcome, "steps": len(events),
                             "final_position": cell(position), "trail_hex": f"{trail:016x}", "events": events})
            print(f"W{bits}A{bits} {name}: {outcome}, {len(events)} attempts", flush=True)
        words(destination / f"events_w{bits}a{bits}.mem", packed)
        words(destination / f"final_w{bits}a{bits}.mem", finals)
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}:
        raise RuntimeError("reference changed during export")
    manifest = {"format":"mazebits-navigation-v1", "case_count":len(CASES), "cases":scenarios,
                "max_steps":MAX_STEPS,"event_stride_words":STRIDE,"cycles_per_decision":CYCLES,
                "event_word_order":["action","position","trail_low","trail_high","status","steps",
                                    "total_cycles","score_up","score_right","score_down","score_left"],
                "final_word_order":["status","steps","position","trail_low","trail_high","total_cycles"],
                "scalar_checked_decisions":scalar_checks,"source_sha256":source_hashes,
                "files":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(destination.glob("*.mem"))},
                "episodes":episodes}
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")
    return manifest

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"test_vectors/navigation")
    args=parser.parse_args()
    manifest=export(args.output_dir)
    print(f"PASS scalar_checked_decisions={manifest['scalar_checked_decisions']}")
