"""Plot recorded FP32 policy improvements without re-running or altering models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot(comparison_dir: Path, training_dir: Path, output_dir: Path):
    comparison_dir, training_dir, output_dir = map(Path, (comparison_dir, training_dir, output_dir))
    comparison = read_json(comparison_dir / "comparison.json")
    metrics = read_json(training_dir / "metrics.json")
    run = read_json(training_dir / "run.json")
    initial = read_json(training_dir / "initial_metrics.json")
    with (training_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError("training history is empty")
    training_model = read_json(training_dir / "model/manifest.json")
    if comparison["models"]["candidate"]["weights_sha256"] != training_model["weights_sha256"]:
        raise ValueError("comparison candidate differs from the selected training checkpoint")
    if comparison["models"]["baseline"]["weights_sha256"] != initial["initialization"]["weights_sha256"]:
        raise ValueError("training initial checkpoint differs from comparison baseline")
    if sha(comparison_dir / "diagnostics.json") != comparison["diagnostics_sha256"]:
        raise ValueError("diagnostic evidence hash mismatch")
    if not comparison.get("inputs_unchanged_during_evaluation"):
        raise ValueError("comparison has no successful unchanged-input verification")
    selected_epoch = int(metrics["best_epoch"])
    epochs = [0, *[int(row["epoch"]) for row in history]]
    successes = [100 * initial["rollout_metrics"]["success_rate"],
                 *[100 * float(row["val_success_rate"]) for row in history]]
    if selected_epoch not in epochs:
        raise ValueError("selected checkpoint is absent from recorded history")
    selected_index = epochs.index(selected_epoch)
    selected_success = 100 * metrics["validation_rollout_metrics"]["success_rate"]
    if abs(successes[selected_index] - selected_success) > 1e-9:
        raise ValueError("selected checkpoint success differs from history")
    datasets = comparison["datasets"]
    if not datasets:
        raise ValueError("comparison contains no validation datasets")
    training_datasets = [item for item in datasets if item["manifest_sha256"] == run["dataset_manifest_sha256"]]
    if len(training_datasets) != 1:
        raise ValueError("comparison must contain the training run's validation dataset exactly once")
    selection_dataset = training_datasets[0]
    if abs(selection_dataset["models"]["candidate"]["rollouts"]["success_rate"] * 100 - selected_success) > 1e-9:
        raise ValueError("fresh comparison differs from training validation success")
    records = []
    for item in datasets:
        before = item["models"]["baseline"]["rollouts"]
        after = item["models"]["candidate"]["rollouts"]
        if before["episodes"] != after["episodes"] or before["episodes"] <= 0:
            raise ValueError("validation comparison is not paired")
        family_names = sorted(before["by_family"])
        family_label = "Random obstacles" if family_names == ["bernoulli"] else "Mixed maze families"
        name = Path(item["directory"]).name
        records.append({"name": name, "families": family_names, "family_label": family_label,
                        "episodes": before["episodes"], "baseline_count": before["success_count"],
                        "candidate_count": after["success_count"],
                        "baseline_percent": 100 * before["success_rate"],
                        "candidate_percent": 100 * after["success_rate"]})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#334155", "text.color": "#17283c",
                         "xtick.color": "#475569", "ytick.color": "#475569",
                         "svg.hashsalt": "mazebits-policy-improvement-v1"})
    colors = {"baseline": "#4e6c96", "candidate": "#087f6d", "initial": "#bd6410"}
    fig, (comparison_axis, training_axis) = plt.subplots(1, 2, figsize=(12.8, 5.5))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.76, bottom=0.22, wspace=0.28)
    x = np.arange(len(records))
    width = 0.34
    for label, shift in (("baseline", -width / 2), ("candidate", width / 2)):
        values = [record[f"{label}_percent"] for record in records]
        bars = comparison_axis.bar(x + shift, values, width, color=colors[label],
                                   label="Original FP32" if label == "baseline" else "Improved FP32", zorder=3)
        for bar, record, value in zip(bars, records, values):
            ypos = value - 6 if value > 14 else value + 3
            comparison_axis.text(bar.get_x() + bar.get_width()/2, ypos,
                                 f"{value:.2f}%", ha="center", va="center", fontweight="bold",
                                 color="white" if value > 14 else "#17283c", fontsize=10)
    comparison_axis.set_xticks(x, [f"{r['name']}\n{r['family_label']}\n{r['episodes']:,} paired episodes" for r in records])
    comparison_axis.tick_params(axis="x", length=0, pad=10)
    comparison_axis.set_title("A  Same scenarios, different checkpoints", loc="left", fontsize=12, fontweight="bold", pad=16)
    comparison_axis.set_ylabel("Routes reaching the goal (%)")
    handles, labels = comparison_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.065, 0.07),
               ncol=2, frameon=False, fontsize=10, borderaxespad=0)
    training_axis.plot(epochs, successes, color=colors["candidate"], linewidth=2.1, zorder=3)
    training_axis.scatter([0], [successes[0]], color=colors["initial"], s=45, zorder=5, clip_on=False)
    training_axis.axvline(selected_epoch, color="#94a3b8", linestyle=(0,(4,3)), linewidth=1, zorder=2)
    training_axis.scatter([selected_epoch], [selected_success], color=colors["candidate"],
                          marker="D", edgecolor="white", linewidth=0.8, s=55, zorder=5, clip_on=False)
    training_axis.set_title(f"B  Fine-tuning on {Path(selection_dataset['directory']).name}",
                            loc="left", fontsize=12, fontweight="bold", pad=16)
    training_axis.set_xlabel("Fine-tuning epoch (0 = original checkpoint)", labelpad=10)
    training_axis.set_ylabel("Validation route success (%)")
    training_axis.set_xlim(-1, max(epochs) + 1)
    training_axis.text(0.04, 0.12, f"Initial: {successes[0]:.2f}%\n"
                       f"Selected epoch {selected_epoch}: {selected_success:.2f}%\n"
                       f"{metrics['validation_rollout_metrics']['success_count']:,} / "
                       f"{metrics['validation_rollout_metrics']['episodes']:,} successful routes",
                       transform=training_axis.transAxes, fontsize=10, linespacing=1.6,
                       bbox={"facecolor":"#f1f5f9", "edgecolor":"none", "pad":9})
    for axis in (comparison_axis, training_axis):
        axis.set_ylim(0, 100)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", color="#d9e1e9", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#cbd5e1")
    same_architecture = comparison["models"]["baseline"]["config"] == comparison["models"]["candidate"]["config"]
    fig.text(0.065, 0.94, "MazeBits | improving the base policy", fontsize=20, fontweight="bold", ha="left")
    description = "Same CNN architecture; broader training mazes." if same_architecture else "Paired comparison of two floating-point policies."
    fig.text(0.065, 0.89, description + "  Full navigation rollouts with raw model actions.", fontsize=11, color="#475569")
    fig.text(0.065, 0.03, "Validation results; checkpoint selected using validation. Final-test partitions reserved.\n"
             "Known photo diagnostics are reported separately; no quantized or FPGA timing claims in this figure.",
             fontsize=9, color="#64748b", linespacing=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("policy_improvement.png", "policy_improvement.svg", "policy_improvement_plot.json")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("plot outputs already exist; choose a new output directory")
    fig.savefig(output_dir / names[0], dpi=220, facecolor="white",
                metadata={"Software":"MazeBits matplotlib report"})
    fig.savefig(output_dir / names[1], facecolor="white", metadata={"Date":None})
    plt.close(fig)
    input_paths = [comparison_dir / "comparison.json", comparison_dir / "diagnostics.json",
                   training_dir / "history.csv", training_dir / "metrics.json",
                   training_dir / "run.json", training_dir / "initial_metrics.json",
                   training_dir / "model/manifest.json", Path(__file__).absolute()]
    report = {"format":"mazebits-policy-improvement-plot-v1", "datasets":records,
              "selected_epoch":selected_epoch, "epochs_completed":max(epochs),
              "initial_success_percent":successes[0], "selected_success_percent":selected_success,
              "y_axis_limits_percent":[0,100], "contains_quantization_results":False,
              "input_sha256":{path.as_posix():sha(path) for path in input_paths},
              "output_sha256":{name:sha(output_dir / name) for name in names[:2]}}
    (output_dir / names[2]).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"Saved {output_dir / names[0]} and SVG; selected epoch {selected_epoch}.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir",type=Path,required=True)
    parser.add_argument("--training-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    plot(args.comparison_dir,args.training_dir,args.output_dir)


if __name__ == "__main__":
    main()
