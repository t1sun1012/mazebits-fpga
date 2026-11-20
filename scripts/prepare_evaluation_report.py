"""Reproduce the improved 8/4-bit evaluation from frozen results, without inference.

Reads only validation metadata and saved measurements. It never opens reserved
final-test arrays or runs a model. Run from any directory with the project Python:
    python scripts/prepare_evaluation_report.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).absolute().parents[1]
PRECISIONS = ("w8a8", "w4a4")
IDENTITY = ("layout_index", "goal_index", "start", "goal", "optimal_steps")
OUTCOMES = ("success", "blocked", "loop", "timeout")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Evidence:
    """Bind each calculation to the exact bytes read, then detect source changes."""
    def __init__(self):
        self.source_sha256 = {}

    def read(self, relative, expected=None):
        path = ROOT / relative
        data = path.read_bytes()
        digest = sha(data)
        if expected is not None and digest != expected:
            raise ValueError(f"Evidence identity mismatch: {relative}")
        key = path.relative_to(ROOT).as_posix()
        if key in self.source_sha256 and self.source_sha256[key] != digest:
            raise RuntimeError(f"Evidence changed while reading: {relative}")
        self.source_sha256[key] = digest
        return data

    def json(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def unchanged(self):
        for name, digest in self.source_sha256.items():
            if sha((ROOT / name).read_bytes()) != digest:
                raise RuntimeError(f"Evidence changed during report creation: {name}")


def summary(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    successes = [row for row in rows if row["outcome"] == "success"]
    ratios = [r["steps"] / r["optimal_steps"] for r in successes if r["optimal_steps"] > 0]
    return {
        "episodes": len(rows), "success_count": len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "outcomes": {key: outcomes[key] for key in OUTCOMES},
        "attempted_actions": sum(r["steps"] for r in rows),
        "successful_path_ratio_count": len(ratios),
        "successful_zero_step_count": sum(r["optimal_steps"] == 0 for r in successes),
        "successful_optimal_count": sum(r["steps"] == r["optimal_steps"] for r in successes),
        "successful_mean_path_ratio": mean(ratios) if ratios else None,
        "successful_mean_steps": mean(r["steps"] for r in successes) if successes else None,
        "successful_mean_BFS_steps": mean(r["optimal_steps"] for r in successes) if successes else None,
    }


def verify_summary(derived, saved):
    for key, value in derived.items():
        if key not in saved:
            continue
        matches = math.isclose(value, saved[key], rel_tol=1e-12, abs_tol=1e-12) if isinstance(value, float) else value == saved[key]
        if not matches:
            raise ValueError(f"Saved rollout summary differs from complete rows: {key}")
    for key in ("episodes", "success_count", "success_rate", "outcomes", "successful_mean_path_ratio"):
        if key not in saved:
            raise ValueError(f"Incomplete saved rollout summary: {key}")


def pair_outcomes(a, b):
    counts = Counter()
    if len(a) != len(b):
        raise ValueError("Paired episode counts differ")
    for left, right in zip(a, b):
        if any(left[key] != right[key] for key in IDENTITY):
            raise ValueError("Paired episode identities differ")
        x, y = left["outcome"] == "success", right["outcome"] == "success"
        counts["both_success" if x and y else "only_8bit_success" if x else "only_4bit_success" if y else "both_fail"] += 1
    return {"episodes": len(a), **{key: counts[key] for key in
            ("both_success", "only_8bit_success", "only_4bit_success", "both_fail")}}


def distance_bucket(distance):
    return "0-9" if distance < 10 else "10-19" if distance < 20 else "20+"


def collect():
    evidence = Evidence()
    evidence.read("scripts/prepare_evaluation_report.py")
    benchmark = evidence.json("results/navigation-structured/benchmark.json")
    for name, expected in benchmark["source_sha256"].items():
        evidence.read(name, expected)
    comparison = evidence.json("results/policy-improvement/quantization/comparison.json")
    dataset_comparison = evidence.json("results/policy-improvement/comparison.json")
    dataset_record = next(item for item in dataset_comparison["datasets"] if item["directory"] == "data/maze-v2")
    metadata = evidence.json("data/maze-v2/val.json", dataset_record["val_metadata_sha256"])
    if metadata["split"] != "val":
        raise ValueError("Only the saved validation partition may be summarized")
    if dataset_record["manifest_sha256"] != comparison["provenance"]["dataset_manifest_sha256"]:
        raise ValueError("Validation metadata and quantization results belong to different datasets")
    board = evidence.json("results/navigation-structured/board-test.json")
    if board["model_set"] != "structured" or board["bitstream_sha256"] != benchmark["bitstream"]["sha256"]:
        raise ValueError("Physical observations refer to a different bitstream")
    implementation = evidence.json("results/navigation-structured/build/implementation.json")
    if implementation["model_set"] != "structured" or implementation["bitstream"]["sha256"] != board["bitstream_sha256"]:
        raise ValueError("Implementation identity differs from the evaluated model set")

    rows, models = {}, {}
    for key in PRECISIONS:
        model_manifest = evidence.json(f"model/cnn-structured-{key}/manifest.json")
        if model_manifest["provenance"]["source_weights_sha256"] != comparison["provenance"]["source_weights_sha256"]:
            raise ValueError("Integer models do not share the evaluated floating-point checkpoint")
        manifest_hash = evidence.source_sha256[f"model/cnn-structured-{key}/manifest.json"]
        if manifest_hash != comparison[key]["model_manifest_sha256"] or manifest_hash != benchmark["models"][key]["model_manifest_sha256"]:
            raise ValueError("Model identity differs between hardware and software results")
        saved = evidence.json(f"results/policy-improvement/quantization/{key}_rollouts.json")
        rows[key] = saved["episodes"]
        if len(rows[key]) != len(metadata["episodes"]):
            raise ValueError("Saved rollout count differs from validation definitions")
        for row, definition in zip(rows[key], metadata["episodes"]):
            if any(row[field] != definition[field] for field in IDENTITY):
                raise ValueError("Saved rollout does not match the fixed validation episode")
            if row["outcome"] not in OUTCOMES or row["steps"] != len(row["actions"]) or len(row["path"]) != row["steps"] + 1:
                raise ValueError("Malformed rollout outcome, attempts, or path")
            if row["path"][0] != row["start"] or (row["outcome"] == "success" and (row["path"][-1] != row["goal"] or row["steps"] < row["optimal_steps"])):
                raise ValueError("Malformed successful rollout")
        aggregate = summary(rows[key])
        verify_summary(aggregate, saved["summary"])
        verify_summary(aggregate, comparison[key]["rollouts"])
        verify_summary(aggregate, {"episodes": benchmark["dataset"]["episodes"], **benchmark["models"][key]})
        quality = comparison[key]
        if quality["states"] != benchmark["dataset"]["states"] or not math.isclose(quality["optimal_action_correct"] / quality["states"], quality["optimal_action_accuracy"]):
            raise ValueError("Inconsistent optimal-action numerator or denominator")
        if quality["optimal_action_accuracy"] != benchmark["models"][key]["optimal_action_accuracy"]:
            raise ValueError("Benchmark action accuracy differs from frozen evaluation")
        hardware = {field: benchmark["models"][key][field] for field in (
            "cycles_per_decision", "nominal_inference_ms", "packed_parameter_bytes",
            "logical_activation_bank_bytes", "routed_cnn_resources")}
        if not math.isclose(hardware["cycles_per_decision"] / benchmark["clock_hz"] * 1000, hardware["nominal_inference_ms"]):
            raise ValueError("Inconsistent cycle-to-time conversion")
        models[key] = {"precision": key.upper(), "model_manifest_sha256": manifest_hash,
            "optimal_action_correct": quality["optimal_action_correct"],
            "optimal_action_accuracy": quality["optimal_action_accuracy"],
            "float_action_agreement": quality["float_action_agreement"],
            "float_action_differences": quality["float_action_differences"],
            "rollouts": aggregate, "hardware": hardware}

    pairs = pair_outcomes(rows["w8a8"], rows["w4a4"])
    if pairs["episodes"] != benchmark["dataset"]["episodes"] or len(metadata["layouts"]) != benchmark["dataset"]["layouts"]:
        raise ValueError("Dataset denominators differ from the saved benchmark")
    groups = {}
    for dimension in ("by_family", "by_bfs_distance"):
        labels = [metadata["layouts"][r["layout_index"]]["family"] if dimension == "by_family"
                  else distance_bucket(r["optimal_steps"]) for r in rows["w8a8"]]
        groups[dimension] = {}
        for label in sorted(set(labels)):
            indices = [i for i, value in enumerate(labels) if value == label]
            group = {key: [rows[key][i] for i in indices] for key in PRECISIONS}
            groups[dimension][label] = {"episodes": len(indices),
                "layouts": len({r["layout_index"] for r in group["w8a8"]}),
                "paired_outcomes": pair_outcomes(group["w8a8"], group["w4a4"]),
                "models": {key: summary(group[key]) for key in PRECISIONS}}
    paired_success = [(a, b) for a, b in zip(rows["w8a8"], rows["w4a4"])
                      if a["outcome"] == b["outcome"] == "success"]
    paths = {"episodes": len(paired_success),
        "models": {key: summary([pair[index] for pair in paired_success]) for index, key in enumerate(PRECISIONS)},
        "w8_shorter": sum(a["steps"] < b["steps"] for a, b in paired_success),
        "w4_shorter": sum(b["steps"] < a["steps"] for a, b in paired_success),
        "same_length": sum(a["steps"] == b["steps"] for a, b in paired_success),
        "extra_4bit_steps_total": sum(b["steps"] - a["steps"] for a, b in paired_success)}
    reduction = lambda a, b: (a - b) / a * 100
    result = {
        "format": "mazebits-evaluation-report-v1",
        "scope": "Improved checkpoint; saved software validation, verified RTL cycles, routed FPGA resources, and separately identified user-reported board checks.",
        "method": "Aggregation only: no training, inference, or reserved final-test access.",
        "dataset": {**benchmark["dataset"], "name": "maze-v2", "distance_groups": ["0-9", "10-19", "20+"],
                    "family_metadata": "data/maze-v2/val.json", "goals_per_layout": 3, "starts_per_goal": 8},
        "models": models, "paired_outcomes": pairs, "paired_successful_paths": paths, **groups,
        "tradeoffs": {
            "route_success_drop_percentage_points": (models["w8a8"]["rollouts"]["success_rate"] - models["w4a4"]["rollouts"]["success_rate"]) * 100,
            "action_accuracy_drop_percentage_points": (models["w8a8"]["optimal_action_accuracy"] - models["w4a4"]["optimal_action_accuracy"]) * 100,
            "packed_parameter_reduction_percent": reduction(models["w8a8"]["hardware"]["packed_parameter_bytes"], models["w4a4"]["hardware"]["packed_parameter_bytes"]),
            "cnn_bram_reduction_percent": reduction(models["w8a8"]["hardware"]["routed_cnn_resources"]["bram_36k_equivalents"], models["w4a4"]["hardware"]["routed_cnn_resources"]["bram_36k_equivalents"]),
            "cnn_lut_reduction_percent": reduction(models["w8a8"]["hardware"]["routed_cnn_resources"]["luts"], models["w4a4"]["hardware"]["routed_cnn_resources"]["luts"]),
            "per_decision_latency_ratio_w4_over_w8": models["w4a4"]["hardware"]["nominal_inference_ms"] / models["w8a8"]["hardware"]["nominal_inference_ms"]},
        "hardware_build": {field: implementation[field] for field in (
            "part", "clock_period_ns", "setup_slack_ns", "hold_slack_ns", "routed_resources",
            "total_drc_count", "bitstream")},
        "physical_checks": board,
        "limitations": [
            "Validation guided development/model selection; the reserved final test remains unevaluated.",
            "The 4,800 software episodes are not 4,800 physical board runs. Board observations are user-reported examples and controls.",
            "Episodes sharing one of the 200 layouts are correlated; no independent-episode confidence interval is claimed.",
            "Raw deterministic argmax actions are used, with blocked/repeated-position/64-step termination and no BFS repair.",
            "Own-success path ratios use different denominators; paired-success paths provide an equal-episode comparison.",
            "Both precisions have equal scheduled cycles per decision. Latency uses the nominal 100 MHz clock and excludes animation pauses.",
            "Packed parameters, logical activation storage, allocated BRAM and bitstream disk size are different quantities.",
            "Known photographed mazes are development diagnostics, not an independent generalization test.",
            "No CPU speedup, power or energy comparison has been measured.",
        ],
        "source_sha256": evidence.source_sha256,
    }
    evidence.unchanged()
    return result, evidence


def draw(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "svg.hashsalt": "mazebits-evaluation-v1", "svg.fonttype": "none"})
    colors = ["#0891B2", "#8B5CF6"]
    models = [result["models"][k] for k in PRECISIONS]
    specs = [
        ("Validation route success", [m["rollouts"]["success_rate"] * 100 for m in models],
         [f"{m['rollouts']['success_rate'] * 100:.4f}%  ({m['rollouts']['success_count']:,}/4,800)" for m in models], "Successful routes (%)", 120),
        ("Packed parameters", [m["hardware"]["packed_parameter_bytes"] for m in models],
         [f"{m['hardware']['packed_parameter_bytes']:,} bytes" for m in models], "Weight + 32-bit bias storage (bytes)", 22000),
        ("Allocated CNN memory", [m["hardware"]["routed_cnn_resources"]["bram_36k_equivalents"] for m in models],
         [f"{m['hardware']['routed_cnn_resources']['bram_36k_equivalents']:g} BRAM36 equivalents" for m in models], "BRAM36 equivalents", 12),
        ("Inference latency", [m["hardware"]["nominal_inference_ms"] for m in models],
         [f"{m['hardware']['nominal_inference_ms']:.5f} ms" for m in models], "Nominal time per decision at 100 MHz (ms)", 42),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.subplots_adjust(left=.09, right=.97, bottom=.21, top=.84, hspace=.68, wspace=.35)
    fig.suptitle("MazeBits: improved 8-bit and 4-bit policies", x=.055, y=.97, ha="left", fontsize=19, fontweight="bold", color="#14213D")
    fig.text(.055, .91, "Same checkpoint and 4,800 validation episodes | Implemented Artix-7 XC7A100T design", color="#526175", fontsize=11)
    for ax, (title, values, labels, unit, limit) in zip(axes.flat, specs):
        ax.barh([1, 0], values, height=.46, color=colors)
        ax.set_yticks([1, 0], ["W8A8", "W4A4"])
        ax.set_xlim(0, limit); ax.set_ylim(-.55, 1.6)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12, color="#14213D")
        ax.set_xlabel(unit, fontsize=10, labelpad=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.set_axisbelow(True); ax.grid(axis="x", color="#E4EAF0", linewidth=.8)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines['bottom'].set_color('#B4C0CB')
        for ypos, value, label in zip([1, 0], values, labels):
            ax.text(value + limit * .025, ypos, label, va="center", fontsize=9, color="#223048", clip_on=False)
    axes[0, 0].set_xticks([0, 25, 50, 75, 100])
    axes[0, 1].set_xticks([0, 5000, 10000, 15000, 20000])
    axes[1, 0].set_xticks([0, 3, 6, 9, 12])
    axes[1, 1].set_xticks([0, 10, 20, 30, 40])
    fig.text(.055, .055, "Lower precision saves storage and BRAM; this implementation has no per-decision speedup.\nQuality is software validation. Resource counts are routed hardware; latency excludes animation pauses.", fontsize=10, color="#526175", linespacing=1.5)
    for suffix in ("png", "svg"):
        kwargs = {"metadata": {"Date": None, "Creator": "MazeBits evaluation"}} if suffix == "svg" else {"metadata": {"Software": "MazeBits evaluation"}}
        fig.savefig(output / f"comparison.{suffix}", dpi=180, facecolor="white", **kwargs)
        if suffix == "svg":
            path = output / "comparison.svg"
            path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation")
    args = parser.parse_args()
    result, evidence = collect()
    output = args.output_dir.absolute(); output.mkdir(parents=True, exist_ok=True)
    draw(result, output)
    evidence.unchanged()
    result["charts"] = {f"comparison.{suffix}": {
        "sha256": sha((output / f"comparison.{suffix}").read_bytes()),
        "bytes": (output / f"comparison.{suffix}").stat().st_size} for suffix in ("png", "svg")}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {output / 'metrics.json'}")
    print(json.dumps(result["paired_outcomes"]))


if __name__ == "__main__":
    main()
