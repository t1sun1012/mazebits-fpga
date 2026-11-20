"""Summarize the frozen Step 6 quantization evidence without re-running inference.

The report rejects mismatched model, RTL, and routed-report identities. It is
scoped to the recorded baseline build, not predictions for a newly trained model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).absolute().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hierarchy_resources(text: str, instance: str) -> dict:
    rows = [line.split("|")[1:-1] for line in text.splitlines() if line.startswith("|")]
    matches = [[cell.strip() for cell in row] for row in rows if row and row[0].strip() == instance]
    if len(matches) != 1 or len(matches[0]) != 10:
        raise ValueError(f"expected exactly one routed hierarchy row for {instance}")
    row = matches[0]
    lut, ff, ramb36, ramb18, dsp = [int(row[index]) for index in (2, 6, 7, 8, 9)]
    return {"luts": lut, "registers": ff, "ramb36": ramb36, "ramb18": ramb18,
            "bram_36k_equivalents": ramb36 + ramb18 / 2, "dsps": dsp,
            "allocated_bram_bits_including_parity_capacity": ramb36 * 36864 + ramb18 * 18432}


def require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"evidence identity mismatch: {description}")


def summarize(root: Path = ROOT) -> dict:
    root = Path(root)
    consumed = {}

    def read(path):
        file = root / path
        consumed[path] = sha(file)
        return file.read_text(encoding="utf-8")

    def data(path):
        return json.loads(read(path))

    implementation = data("results/navigation/implementation.json")
    verification = data("results/navigation/verification.json")
    published = data("results/navigation/published_report_hashes.json")
    comparison = data("results/quantization/comparison.json")
    paired = data("results/quantization/paired_comparison.json")
    dataset = data("results/dataset/manifest.json")
    require_equal(implementation["status"], "pass", "routed build status")
    require_equal(comparison["provenance"]["dataset_manifest_sha256"],
                  consumed["results/dataset/manifest.json"], "validation dataset")
    cnn_checks = [check for check in verification["tests"] if check["test"] == "cnn"]
    if len(cnn_checks) != 1 or cnn_checks[0]["status"] != "pass":
        raise ValueError("expected a passing real dual-CNN simulation record")
    cnn_check = cnn_checks[0]
    cycles = int(cnn_check["coverage"]["cycles_per_decision"])
    if cycles <= 0 or implementation["clock_period_ns"] <= 0:
        raise ValueError("cycle count and clock period must be positive")
    clock_hz = 1e9 / implementation["clock_period_ns"]
    hierarchy_name = "results/navigation/utilization_hierarchy.rpt"
    hierarchy = read(hierarchy_name)
    require_equal(consumed[hierarchy_name], published["utilization_hierarchy.rpt"], "published hierarchy report")
    source = read("rtl/mazebits_cnn.v")
    top = read("rtl/mazebits_navigation_top.v")
    for filename in ("rtl/mazebits_cnn.v", "rtl/mazebits_navigation_top.v"):
        require_equal(consumed[filename], implementation["source_sha256"][filename], filename)
    banks = re.findall(r"reg\s+\[A_BITS-1:0\]\s+(bank_[ab])\s+\[0:(\d+)\]", source)
    if len(banks) != 2 or {name for name, _ in banks} != {"bank_a", "bank_b"}:
        raise ValueError("cannot identify the two activation banks in the recorded RTL")
    words = sum(int(last) + 1 for _, last in banks)
    delay = re.search(r"parameter integer STEP_DELAY_CYCLES\s*=\s*(\d+)", top)
    if delay is None:
        raise ValueError("cannot identify the recorded animation delay")
    result = {
        "format": "mazebits-benchmark-summary-v1",
        "scope": "Frozen Step 6 model, software validation, and routed dual-engine FPGA build",
        "part": implementation["part"], "clock_hz": clock_hz,
        "bitstream_sha256": implementation["bitstream"]["sha256"],
        "timing_evidence": "RTL cycle counts corroborated by photographed board counters; clock from routed constraints, not an independent oscillator measurement",
        "animation_delay_cycles": int(delay.group(1)),
        "animation_delay_ms_per_nonterminal_action": int(delay.group(1)) / clock_hz * 1000,
        "dataset": {"partition": "validation", "layouts": dataset["splits"]["val"]["layouts"],
                    "states": dataset["splits"]["val"]["samples"],
                    "episodes": dataset["splits"]["val"]["episodes"],
                    "family": dataset["distribution"]["family"],
                    "test_partition_evaluated": False},
        "models": {},
        "paired_outcomes": paired["outcomes"],
    }
    for bits in (8, 4):
        tag = f"w{bits}a{bits}"
        folder = f"model/cnn-{tag}"
        filename = f"{folder}/manifest.json"
        manifest = data(filename)
        metrics = comparison[tag]
        require_equal(manifest["bits"], bits, "model precision")
        for expected in (metrics["model_manifest_sha256"],
                         implementation["model_stage"]["source_sha256"][filename],
                         cnn_check["source_sha256"][filename]):
            require_equal(consumed[filename], expected, filename)
        for name, info in manifest["files"].items():
            artifact = f"{folder}/{name}"
            read(artifact)
            require_equal(consumed[artifact], info["sha256"], artifact)
            require_equal(consumed[artifact], implementation["source_sha256"][artifact], artifact)
        require_equal(manifest["provenance"]["source_weights_sha256"],
                      comparison["provenance"]["source_weights_sha256"], "source floating-point model")
        weights = manifest["files"]["weights.mem"]
        biases = manifest["files"]["biases.mem"]
        parameter_bits = weights["words"] * weights["word_bits"] + biases["words"] * biases["word_bits"]
        require_equal(parameter_bits, manifest["logical_storage"]["total_bits"], "packed parameter bits")
        require_equal((parameter_bits + 7) // 8, metrics["logical_parameter_bytes"], "packed parameter bytes")
        require_equal(metrics["states"], result["dataset"]["states"], "state count")
        require_equal(metrics["rollouts"]["episodes"], result["dataset"]["episodes"], "episode count")
        result["models"][tag] = {
            "model_manifest_sha256": consumed[filename],
            "cycles_per_decision": cycles,
            "nominal_inference_ms": cycles / clock_hz * 1000,
            "packed_parameter_bytes": (parameter_bits + 7) // 8,
            "packed_weight_bytes": weights["words"] * weights["word_bits"] // 8,
            "bias_bytes": biases["words"] * biases["word_bits"] // 8,
            "logical_activation_bank_bytes": words * bits // 8,
            "activation_memory_scope": "Two feature banks only; excludes parameters, registers, control, and BRAM allocation overhead",
            "routed_cnn_resources": hierarchy_resources(hierarchy, f"cnn{bits}"),
            "optimal_action_accuracy": metrics["optimal_action_accuracy"],
            "success_count": metrics["rollouts"]["success_count"],
            "success_rate": metrics["rollouts"]["success_rate"],
            "outcomes": metrics["rollouts"]["outcomes"],
            "successful_mean_path_ratio": metrics["rollouts"]["successful_mean_path_ratio"],
        }
    a, b = result["models"]["w8a8"], result["models"]["w4a4"]
    result["w4_vs_w8"] = {
        "parameter_byte_reduction_percent": 100 * (1 - b["packed_parameter_bytes"] / a["packed_parameter_bytes"]),
        "routed_cnn_bram_reduction_percent": 100 * (1 - b["routed_cnn_resources"]["bram_36k_equivalents"] / a["routed_cnn_resources"]["bram_36k_equivalents"]),
        "routed_cnn_lut_reduction_percent": 100 * (1 - b["routed_cnn_resources"]["luts"] / a["routed_cnn_resources"]["luts"]),
        "validation_success_change_percentage_points": 100 * (b["success_rate"] - a["success_rate"]),
        "decision_speedup": a["nominal_inference_ms"] / b["nominal_inference_ms"],
    }
    result["source_sha256"] = consumed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args()
    result = summarize()
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
        print(f"Saved {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
