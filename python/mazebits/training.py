"""Deterministic CPU imitation training and validation-only model selection."""

import copy
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from .evaluation import evaluate_rollouts
from .model import ModelConfig, PolicyCNN, augment_batch, optimal_set_loss
from .model_io import load_model, save_model, sha256, write_json


def configure_cpu(seed: int, threads: int = 1) -> None:
    if seed < 0 or threads < 1:
        raise ValueError("seed must be nonnegative and threads positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_split(directory: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load only train or validation; final-test data is deliberately excluded."""
    if split not in ("train", "val"):
        raise ValueError("Step 2 loads only train and val; test is reserved")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for filename in (f"{split}.npz", f"{split}.json"):
        if sha256(directory / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"dataset hash mismatch: {filename}")
    with np.load(directory / f"{split}.npz", allow_pickle=False) as arrays:
        inputs, targets = arrays["inputs"], arrays["targets"]
        if (inputs.dtype != np.uint8 or inputs.ndim != 4
                or tuple(inputs.shape[1:]) != (3, 8, 8)
                or targets.dtype != np.uint8 or targets.shape != (len(inputs), 4)):
            raise ValueError("dataset shape or dtype violates the input contract")
        if (not np.isin(inputs, [0, 1]).all() or not np.isin(targets, [0, 1]).all()
                or not targets.any(axis=1).all()
                or not (inputs[:, 1].sum(axis=(1, 2)) == 1).all()
                or not (inputs[:, 2].sum(axis=(1, 2)) == 1).all()):
            raise ValueError("invalid binary inputs or empty optimal-action labels")
        x = torch.from_numpy(inputs.astype(np.float32))
        y = torch.from_numpy(targets.copy())
    metadata = json.loads((directory / f"{split}.json").read_text(encoding="utf-8"))
    if metadata["split"] != split:
        raise ValueError("dataset metadata split mismatch")
    return x, y, metadata


def state_metrics(model, inputs, targets, *, batch_size: int = 512) -> dict:
    previous = model.training
    model.eval()
    total_loss, correct = 0.0, 0
    try:
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                x, y = inputs[start:start + batch_size], targets[start:start + batch_size]
                logits = model(x)
                if not torch.isfinite(logits).all():
                    raise ValueError("model produced non-finite logits")
                total_loss += float(optimal_set_loss(logits, y)) * len(x)
                predicted = logits.argmax(dim=1)
                correct += int(y.gather(1, predicted[:, None]).sum())
    finally:
        model.train(previous)
    return {"states": len(inputs), "optimal_action_correct": correct,
            "optimal_action_accuracy": correct / len(inputs),
            "optimal_set_loss": total_loss / len(inputs)}


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "python").rglob("*.py")), root / "requirements.txt"]
    return {path.relative_to(root).as_posix(): sha256(path) for path in paths}


def initialize_policy(config: ModelConfig, initial_model: Path | None = None):
    """Load checked portable weights for fine-tuning, or initialize a new CNN."""
    if initial_model is None:
        return PolicyCNN(config), {"method": "random initialization"}
    model, manifest = load_model(Path(initial_model))
    if model.config != config:
        raise ValueError("initial model architecture does not match requested channels/depth")
    return model, {
        "method": "fine-tune existing floating-point weights; new Adam optimizer",
        "weights_sha256": manifest["weights_sha256"],
        "manifest_sha256": sha256(Path(initial_model) / "manifest.json"),
        "source_dataset_manifest_sha256": manifest.get("provenance", {}).get("dataset_manifest_sha256"),
    }


def train(
    data_dir: Path, run_dir: Path, *, seed: int = 311, channels: int = 16,
    depth: int = 8, epochs: int = 60, patience: int = 12, batch_size: int = 256,
    learning_rate: float = 0.001, threads: int = 1, augment: bool = True,
    initial_model: Path | None = None
) -> dict:
    if epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs/patience/batch size/learning rate must be positive")
    model_config = ModelConfig(channels=channels, depth=depth)
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run output must be empty: {run_dir}")
    configure_cpu(seed, threads)
    model, initialization = initialize_policy(model_config, initial_model)
    train_x, train_y, _ = load_split(data_dir, "train")
    val_x, val_y, val_metadata = load_split(data_dir, "val")
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffle_rng = torch.Generator().manual_seed(seed)
    augmentation_rng = torch.Generator().manual_seed(seed + 1)
    config = {
        "seed": seed, "channels": channels, "depth": depth, "epochs": epochs,
        "patience": patience, "batch_size": batch_size, "learning_rate": learning_rate,
        "optimizer": "Adam", "threads": threads, "augment_d4": augment,
        "initialization": initialization,
        "loss": "negative log probability mass of all optimal actions",
        "selection": "highest validation rollout success count; then lowest validation loss",
        "rollout": "raw argmax, no action mask, stop on goal/blocked/repeated cell/64 actions",
        "test_status": "reserved; test arrays and episodes not loaded",
        "validation_success_target": 0.90,
    }
    provenance = {
        "training_config": config, "model": model.description(),
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
        "source_sha256": source_hashes(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "platform": platform.platform(),
                        "device": "cpu"},
    }
    write_json(run_dir / "run.json", provenance)
    best_key, best_state, best_epoch, stale = (-1, -float("inf")), None, 0, 0
    history = []
    start_time = time.perf_counter()
    if initial_model is not None:
        # Keep the starting weights eligible: fine-tuning must beat their
        # validation rollout count/loss to replace the selected checkpoint.
        initial_validation = state_metrics(model, val_x, val_y)
        initial_rollout = evaluate_rollouts(model, val_metadata)["summary"]
        best_key = (initial_rollout["success_count"], -initial_validation["optimal_set_loss"])
        best_state = copy.deepcopy(model.state_dict())
        torch.save({"state_dict": best_state, "epoch": 0}, run_dir / "best.pt")
        write_json(run_dir / "initial_metrics.json", {
            "state_metrics": initial_validation, "rollout_metrics": initial_rollout,
            "initialization": initialization,
        })
        print(json.dumps({"initial": {"success_count": initial_rollout["success_count"],
                                      "episodes": initial_rollout["episodes"],
                                      "loss": initial_validation["optimal_set_loss"]}}), flush=True)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(train_x), generator=shuffle_rng)
        training_loss = 0.0
        for offset in range(0, len(train_x), batch_size):
            indices = permutation[offset:offset + batch_size]
            x, y = train_x[indices], train_y[indices]
            if augment:
                transform = int(torch.randint(8, (), generator=augmentation_rng))
                x, y = augment_batch(x, y, transform % 4, transform >= 4)
            optimizer.zero_grad(set_to_none=True)
            loss = optimal_set_loss(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(indices)
        validation = state_metrics(model, val_x, val_y)
        rollout = evaluate_rollouts(model, val_metadata)["summary"]
        key = (rollout["success_count"], -validation["optimal_set_loss"])
        improved = key > best_key
        if improved:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_state, "epoch": epoch},
                       run_dir / "best.pt")
        else:
            stale += 1
        row = {
            "epoch": epoch, "train_loss": training_loss / len(train_x),
            "val_loss": validation["optimal_set_loss"],
            "val_action_accuracy": validation["optimal_action_accuracy"],
            "val_success_count": rollout["success_count"],
            "val_episodes": rollout["episodes"],
            "val_success_rate": rollout["success_rate"],
            "val_mean_success_path_ratio": rollout["successful_mean_path_ratio"],
            "selected": improved,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row, allow_nan=False), flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    validation = state_metrics(model, val_x, val_y)
    rollouts = evaluate_rollouts(model, val_metadata)
    metrics = {
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - start_time,
        "validation_state_metrics": validation,
        "validation_rollout_metrics": rollouts["summary"],
        "validation_success_target": config["validation_success_target"],
        "validation_target_met": rollouts["summary"]["success_rate"] >= 0.90,
        "test_status": config["test_status"],
    }
    provenance["selected_epoch"] = best_epoch
    provenance["metrics"] = metrics
    save_model(model, run_dir / "model", provenance=provenance)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "validation_rollouts.json", rollouts)
    print(json.dumps({"complete": metrics}, allow_nan=False), flush=True)
    return metrics
