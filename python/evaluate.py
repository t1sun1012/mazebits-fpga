"""Reload an exported FP32 policy and reproduce validation-only evaluation."""

import argparse
import json
import math
from pathlib import Path

from mazebits.evaluation import evaluate_rollouts
from mazebits.model_io import load_model, sha256, write_json
from mazebits.training import configure_cpu, load_split, state_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("model/cnn-fp32"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/maze-v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-rollouts", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("evaluation output must be new or empty")
    configure_cpu(311, 1)
    model, manifest = load_model(args.model_dir)
    dataset_hash = sha256(args.data_dir / "manifest.json")
    provenance = manifest["provenance"]
    if dataset_hash != provenance["dataset_manifest_sha256"]:
        raise ValueError("dataset differs from the model's recorded training dataset")
    inputs, targets, metadata = load_split(args.data_dir, "val")
    states = state_metrics(model, inputs, targets)
    rollouts = evaluate_rollouts(model, metadata)
    expected = provenance["metrics"]
    expected_states = expected["validation_state_metrics"]
    state_counts_match = all(states[key] == expected_states[key] for key in states
                             if key != "optimal_set_loss")
    loss_matches = math.isclose(states["optimal_set_loss"],
                                expected_states["optimal_set_loss"],
                                rel_tol=1e-6, abs_tol=1e-7)
    if (not state_counts_match or not loss_matches
            or rollouts["summary"] != expected["validation_rollout_metrics"]):
        raise ValueError("fresh evaluation differs from the saved training results")
    full_trajectory_match = None
    if args.reference_rollouts:
        reference = json.loads(args.reference_rollouts.read_text(encoding="utf-8"))
        full_trajectory_match = rollouts == reference
        if not full_trajectory_match:
            raise ValueError("a validation trajectory differs from the reference")
    result = {
        "weights_sha256": manifest["weights_sha256"],
        "dataset_manifest_sha256": dataset_hash,
        "validation_state_metrics": states,
        "validation_rollout_metrics": rollouts["summary"],
        "training_metrics_match": True,
        "state_metrics_match_exactly": states == expected_states,
        "loss_comparison_tolerance": {"relative": 1e-6, "absolute": 1e-7},
        "full_trajectory_match": full_trajectory_match,
        "test_status": "reserved; test arrays and episodes not loaded",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "verification.json", result)
    write_json(args.output_dir / "validation_rollouts.json", rollouts)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
