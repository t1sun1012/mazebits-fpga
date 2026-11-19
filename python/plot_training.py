"""Render the recorded training history; requires requirements-plots.txt."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/cnn-baseline"))
    args = parser.parse_args()
    with (args.results_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = json.loads((args.results_dir / "metrics.json").read_text(encoding="utf-8"))
    epochs = [int(row["epoch"]) for row in rows]
    success = [100 * float(row["val_success_rate"]) for row in rows]
    best = metrics["best_epoch"]
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "mazebits-step-02"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    axes[0].semilogy(epochs, [float(row["train_loss"]) for row in rows],
                     color="#2563eb", label="Training (augmented)")
    axes[0].semilogy(epochs, [float(row["val_loss"]) for row in rows],
                     color="#f97316", label="Validation")
    axes[0].set(title="Optimal-action set loss", xlabel="Epoch", ylabel="Loss (log scale)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(epochs, success, color="#0f766e")
    axes[1].scatter([best], [success[best - 1]], color="#0f766e", zorder=3)
    axes[1].set(title="Complete validation navigation runs",
                xlabel="Epoch", ylabel="Goal-reaching success (%)", ylim=(80, 100.5))
    axes[1].text(0.96, 0.12, f"Selected epoch {best}\n{success[best - 1]:.2f}% success",
                 transform=axes[1].transAxes, ha="right", color="#115e59")
    for axis in axes:
        axis.axvline(best, color="#64748b", linewidth=1, linestyle="--", alpha=0.7)
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MazeBits: CPU imitation learning\n"
                 "3,600 validation runs on 150 unseen layouts; test partition reserved",
                 fontsize=12)
    fig.savefig(args.results_dir / "training.png", dpi=180,
                metadata={"Software": "MazeBits matplotlib report"})
    fig.savefig(args.results_dir / "training.svg", metadata={"Date": None})
    plt.close(fig)
    print("Wrote training.png and training.svg")


if __name__ == "__main__":
    main()
