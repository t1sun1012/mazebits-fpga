"""Calibrate, export, reload, and evaluate W8A8/W4A4 post-training quantization."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from mazebits.calibration import calibrate
from mazebits.evaluation import evaluate_rollouts
from mazebits.fixed_point import from_float
from mazebits.model_io import load_model, sha256, write_json
from mazebits.quantized_evaluation import evaluate_integer, float_scores
from mazebits.quantized_io import load_quantized, save_quantized
from mazebits.training import configure_cpu, load_split, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("model/cnn-fp32"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/maze-v1"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/quantization-ptq"))
    parser.add_argument("--calibration-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("quantization output must be new or empty")
    configure_cpu(args.seed, 1)
    model, fp_manifest = load_model(args.model_dir)
    dataset_hash = sha256(args.data_dir / "manifest.json")
    if dataset_hash != fp_manifest["provenance"]["dataset_manifest_sha256"]:
        raise ValueError("calibration dataset does not match the trained model")
    train_inputs, _, _ = load_split(args.data_dir, "train")
    started = time.perf_counter()
    calibration = calibrate(model, train_inputs.numpy(),
                            sample_count=args.calibration_samples, seed=args.seed)
    del train_inputs
    args.run_dir.mkdir(parents=True, exist_ok=True)
    calibration["source_weights_sha256"] = fp_manifest["weights_sha256"]
    calibration["dataset_manifest_sha256"] = dataset_hash
    write_json(args.run_dir / "calibration.json", calibration)
    calibration_hash = sha256(args.run_dir / "calibration.json")
    provenance = {
        "method": "post-training quantization; no fine-tuning",
        "source_weights_sha256": fp_manifest["weights_sha256"],
        "dataset_manifest_sha256": dataset_hash,
        "calibration_sha256": calibration_hash,
        "calibration_partition": "train",
        "calibration_samples": calibration["sampling"]["actual_samples"],
        "calibration_seed": args.seed,
        "test_status": "reserved; no test arrays or episodes loaded",
        "source_sha256": source_hashes(),
    }
    inputs, targets, metadata = load_split(args.data_dir, "val")
    reference_scores = float_scores(model, inputs)
    float_actions = reference_scores.argmax(axis=1)
    float_good = targets.numpy()[np.arange(len(inputs)), float_actions]
    fp_rollouts = evaluate_rollouts(model, metadata)
    if fp_rollouts["summary"] != fp_manifest["provenance"]["metrics"]["validation_rollout_metrics"]:
        raise ValueError("floating-point baseline replay mismatch")
    results = {
        "provenance": provenance,
        "float32": {
            "states": len(inputs),
            "optimal_action_accuracy": float(float_good.mean()),
            "optimal_action_correct": int(float_good.sum()),
            "rollouts": fp_rollouts["summary"],
        },
    }
    predictions = {"float32_actions": float_actions.astype(np.uint8)}
    print(json.dumps({"calibration_complete": {
        bits: {key: value[key] for key in ("weight_fracs", "activation_fracs")}
        for bits, value in calibration["schemes"].items()
    }}), flush=True)
    for bits in (8, 4):
        tag = f"w{bits}a{bits}"
        policy = from_float(model, bits, calibration, provenance=provenance)
        exported = save_quantized(policy, args.run_dir / tag)
        reloaded = load_quantized(args.run_dir / tag)
        metrics, rollouts, scores = evaluate_integer(
            reloaded, inputs, targets, metadata, reference_scores)
        metrics["logical_parameter_bytes"] = exported["logical_storage"]["total_bytes"]
        metrics["model_manifest_sha256"] = sha256(args.run_dir / tag / "manifest.json")
        results[tag] = metrics
        predictions[tag + "_scores"] = scores
        write_json(args.run_dir / (tag + "_rollouts.json"), rollouts)
        write_json(args.run_dir / (tag + "_metrics.json"), metrics)
        print(json.dumps({tag: {"optimal_action_accuracy": metrics["optimal_action_accuracy"],
                               "rollouts": metrics["rollouts"]}}, allow_nan=False), flush=True)
    results["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.run_dir / "comparison.json", results)
    np.savez_compressed(args.run_dir / "validation_predictions.npz", **predictions)
    print(json.dumps({"complete": True, "seconds": results["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
