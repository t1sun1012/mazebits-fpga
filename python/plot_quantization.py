"""Render the measured precision comparison; requires requirements-plots.txt."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/quantization"))
    args = parser.parse_args()
    data = json.loads((args.results_dir / "comparison.json").read_text(encoding="utf-8"))
    names = ["FP32", "W8A8", "W4A4"]
    keys = ["float32", "w8a8", "w4a4"]
    success = [100 * data[key]["rollouts"]["success_rate"] for key in keys]
    memory = [16756 * 4] + [data[key]["logical_parameter_bytes"] for key in keys[1:]]
    colors = ["#64748b", "#2563eb", "#0f766e"]
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "mazebits-step-03"})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), layout="constrained")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    bars = axes[0].bar(names, success, color=colors, width=0.6)
    axes[0].bar_label(bars, labels=[f"{value:.2f}%" for value in success], padding=4)
    axes[0].set(title="Goal-reaching validation runs", ylabel="Success (%)", ylim=(0, 110))
    bars = axes[1].bar(names, [value / 1024 for value in memory], color=colors, width=0.6)
    axes[1].bar_label(bars, labels=[f"{value:,} B" for value in memory], padding=4)
    axes[1].set(title="Packed weights and biases", ylabel="Logical parameter storage (KiB)",
                ylim=(0, 73))
    fig.suptitle("MazeBits: measured precision tradeoff\n"
                 "3,600 validation runs; shared source checkpoint; no fine-tuning", fontsize=12)
    fig.supxlabel("Logical storage excludes FPGA allocation overhead; board resource and timing results are pending.",
                   fontsize=9)
    fig.savefig(args.results_dir / "comparison.png", dpi=180,
                metadata={"Software": "MazeBits quantization report"})
    fig.savefig(args.results_dir / "comparison.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
