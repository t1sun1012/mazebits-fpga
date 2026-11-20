"""Reload published memory files and reproduce every integer validation output."""

import argparse
import json
from pathlib import Path

import numpy as np

from mazebits.model_io import load_model, sha256, write_json
from mazebits.quantized_evaluation import evaluate_integer, float_scores
from mazebits.quantized_io import load_quantized
from mazebits.training import configure_cpu, load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/maze-v1"))
    parser.add_argument("--model-root", type=Path, default=Path("model"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/quantization"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("verification output must be new or empty")
    configure_cpu(311, 1)
    expected = json.loads((args.results_dir / "comparison.json").read_text(encoding="utf-8"))
    dataset_hash = sha256(args.data_dir / "manifest.json")
    if dataset_hash != expected["provenance"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest mismatch")
    calibration_hash = sha256(args.results_dir / "calibration.json")
    if calibration_hash != expected["provenance"]["calibration_sha256"]:
        raise ValueError("calibration report hash mismatch")
    fp_model, fp_manifest = load_model(args.model_root / "cnn-fp32")
    if fp_manifest["weights_sha256"] != expected["provenance"]["source_weights_sha256"]:
        raise ValueError("floating-point source model mismatch")
    inputs, targets, metadata = load_split(args.data_dir, "val")
    fp_scores = float_scores(fp_model, inputs)
    verification = {"dataset_manifest_sha256": dataset_hash,
                    "calibration_sha256": calibration_hash,
                    "test_status": "reserved; no test arrays or episodes loaded",
                    "models": {}}
    with np.load(args.results_dir / "validation_predictions.npz", allow_pickle=False) as stored:
        if not np.array_equal(fp_scores.argmax(axis=1), stored["float32_actions"]):
            raise ValueError("floating-point reference actions changed")
        for bits in (8, 4):
            tag = f"w{bits}a{bits}"
            directory = args.model_root / f"cnn-{tag}"
            manifest_hash = sha256(directory / "manifest.json")
            if manifest_hash != expected[tag]["model_manifest_sha256"]:
                raise ValueError("quantized manifest identity mismatch")
            policy = load_quantized(directory)
            if policy.provenance["calibration_sha256"] != calibration_hash:
                raise ValueError("model calibration identity mismatch")
            metrics, rollouts, scores = evaluate_integer(
                policy, inputs, targets, metadata, fp_scores)
            metrics["logical_parameter_bytes"] = expected[tag]["logical_parameter_bytes"]
            metrics["model_manifest_sha256"] = manifest_hash
            if metrics != expected[tag]:
                raise ValueError(f"{tag} validation metrics differ")
            if not np.array_equal(scores, stored[tag + "_scores"]):
                raise ValueError(f"{tag} integer scores differ")
            expected_rollouts = json.loads(
                (args.results_dir / (tag + "_rollouts.json")).read_text(encoding="utf-8"))
            if rollouts != expected_rollouts:
                raise ValueError(f"{tag} validation trajectories differ")
            verification["models"][tag] = {
                "model_manifest_sha256": manifest_hash,
                "score_words_compared": int(scores.size),
                "score_mismatches": 0,
                "trajectories_compared": len(rollouts["episodes"]),
                "trajectory_mismatches": 0,
                "scores_sha256_int32_little_endian": metrics["scores_sha256_int32_little_endian"],
            }
            print(json.dumps({tag: verification["models"][tag]}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "verification.json", verification)


if __name__ == "__main__":
    main()
