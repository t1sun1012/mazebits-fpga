"""Evaluate an exported integer bundle on a named validation distribution.

No calibration or training occurs here, and reserved-test examples are not read.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from compare_policies import rollout_metrics_grouped
from mazebits.model_io import load_model, sha256, write_json
from mazebits.quantized_io import load_quantized
from mazebits.quantized_evaluation import evaluate_integer, float_scores
from mazebits.training import configure_cpu, load_split


def evaluate_bundle(model_dir, float_model_dir, data_dir, output_dir):
    model_dir, float_model_dir, data_dir, output_dir = map(Path, (model_dir, float_model_dir, data_dir, output_dir))
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("evaluation output must be new or empty")
    policy = load_quantized(model_dir)
    model, fp_manifest = load_model(float_model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["provenance"]["source_weights_sha256"] != fp_manifest["weights_sha256"]:
        raise ValueError("integer bundle and floating-point source checkpoint do not match")
    paths = [*(model_dir / name for name in ("manifest.json", "weights.mem", "biases.mem", "quantization.vh")),
             *(float_model_dir / name for name in ("manifest.json", "weights.npz")),
             *(data_dir / name for name in ("manifest.json", "val.npz", "val.json"))]
    watched = {path: sha256(path) for path in paths}
    started = time.perf_counter()
    inputs, targets, metadata = load_split(data_dir, "val")
    scores = float_scores(model, inputs)
    metrics, rollouts, integer_scores = evaluate_integer(policy, inputs, targets, metadata, scores)
    metrics["rollouts"] = rollout_metrics_grouped(rollouts, metadata)
    metrics["logical_parameter_bytes"] = manifest["logical_storage"]["total_bytes"]
    metrics["model_manifest_sha256"] = watched[model_dir / "manifest.json"]
    if any(sha256(path) != digest for path, digest in watched.items()):
        raise RuntimeError("model or dataset changed during evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "rollouts.json", rollouts)
    np.savez_compressed(output_dir / "state_predictions.npz", scores=integer_scores,
                        float_actions=scores.argmax(axis=1).astype(np.uint8))
    root = Path(__file__).absolute().parents[1]
    source_paths = [Path(__file__).absolute(), root / "python/compare_policies.py", *[root / "python/mazebits" / n for n in
                    ("model.py", "model_io.py", "quantized_io.py", "fixed_point.py", "quantized_evaluation.py", "training.py", "evaluation.py")]]
    record = {"format": "mazebits-integer-evaluation-v1", "precision": metrics["precision"],
              "model_directory": model_dir.as_posix(), "float_source_directory": float_model_dir.as_posix(),
              "dataset_directory": data_dir.as_posix(), "split": "val", "calibration_changed": False,
              "test_status": "reserved; final-test arrays and episodes not loaded",
              "input_sha256": {p.as_posix(): value for p, value in watched.items()},
              "source_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in source_paths},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "threads": torch.get_num_threads()},
              "elapsed_seconds": time.perf_counter() - started,
              "outputs": {p.name: sha256(p) for p in output_dir.iterdir() if p.is_file()}}
    write_json(output_dir / "verification.json", record)
    print(json.dumps({"precision": metrics["precision"], "dataset": data_dir.as_posix(),
                      "success_count": metrics["rollouts"]["success_count"], "episodes": metrics["rollouts"]["episodes"],
                      "success_rate": metrics["rollouts"]["success_rate"], "outcomes": metrics["rollouts"]["outcomes"]}), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "float-model-dir", "data-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    configure_cpu(312, 1)
    evaluate_bundle(**vars(args))
