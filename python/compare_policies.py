"""Compare FP32 policies on shared validation layouts and separate known failures.

Both models receive exactly the same states and rollout definitions. The reserved
final-test arrays and episode files are never opened. Diagnostic photographs are
known development cases, not an independent estimate of generalization.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from mazebits.encoding import encode_state
from mazebits.evaluation import evaluate_rollouts, _summary
from mazebits.maze import Action, Maze, step
from mazebits.model_io import load_model, sha256, write_json
from mazebits.reference import shortest_path
from mazebits.training import configure_cpu, load_split

ROOT = Path(__file__).absolute().parents[1]
DIAGNOSTIC_MAZES = {
    "precision-reversal-IMG_7179": "S#......\n.#......\n.#......\n.#.....G\n.#......\n.#......\n........\n........",
    "long-detour-IMG_7180": "S#G.....\n.######.\n.#......\n.#......\n.#......\n.#......\n.#......\n........",
    "precision-loss-latest-photo": "S.......\n...####.\n...#....\n...#....\n...#....\n...#....\n...#....\n.......G",
}


def distance_bucket(distance: int) -> str:
    if distance < 0:
        raise ValueError("distance must be nonnegative")
    return "0-9" if distance < 10 else "10-19" if distance < 20 else "20+"


def _path_label(path: Path) -> str:
    try:
        return path.absolute().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.absolute())


def _source_hashes() -> dict:
    paths = [Path(__file__).absolute(), *[ROOT / "python/mazebits" / name for name in (
        "model.py", "model_io.py", "training.py", "evaluation.py", "maze.py",
        "encoding.py", "reference.py")]]
    return {_path_label(path): sha256(path) for path in paths}


def _state_summary(correct: np.ndarray, losses: np.ndarray) -> dict:
    return {
        "states": len(correct),
        "optimal_action_correct": int(correct.sum()),
        "optimal_action_accuracy": float(correct.mean()) if len(correct) else None,
        "optimal_set_loss": float(losses.mean()) if len(losses) else None,
    }


def state_metrics_grouped(model, inputs, targets, layout_indices, distances,
                          metadata, *, batch_size=512) -> dict:
    """One batched inference pass; aggregate the same decisions by each group."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    layout_indices, distances = np.asarray(layout_indices), np.asarray(distances)
    if (layout_indices.shape != (len(inputs),)
            or distances.shape != (len(inputs),)
            or not np.issubdtype(layout_indices.dtype, np.integer)
            or not np.issubdtype(distances.dtype, np.integer)
            or np.any(layout_indices < 0)
            or np.any(layout_indices >= len(metadata["layouts"]))
            or np.any(distances < 0)):
        raise ValueError("invalid per-state layout indices or distances")
    if targets.shape != (len(inputs), 4) or not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("invalid optimal-action labels")
    if not targets.any(dim=1).all():
        raise ValueError("empty optimal-action labels")
    mode_states = [(module, module.training) for module in model.modules()]
    correct_parts, loss_parts = [], []
    try:
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(inputs), batch_size):
                x, y = inputs[offset:offset+batch_size], targets[offset:offset+batch_size]
                scores = model(x)
                if (not isinstance(scores, torch.Tensor)
                        or scores.shape != (len(x), 4)
                        or not torch.isfinite(scores).all()):
                    raise ValueError("model must return finite [batch, 4] scores")
                actions = scores.argmax(dim=1)
                correct_parts.append(y.gather(1, actions[:, None]).squeeze(1).bool().cpu().numpy())
                losses = (torch.logsumexp(scores, dim=1)
                          - torch.logsumexp(scores.masked_fill(~y.bool(), -torch.inf), dim=1))
                loss_parts.append(losses.cpu().numpy().astype(np.float64))
    finally:
        for module, mode in mode_states:
            module.training = mode
    correct = np.concatenate(correct_parts) if correct_parts else np.array([], dtype=bool)
    losses = np.concatenate(loss_parts) if loss_parts else np.array([], dtype=np.float64)
    result = _state_summary(correct, losses)
    families = np.asarray([metadata["layouts"][int(index)].get("family", "unknown")
                           for index in layout_indices], dtype=str)
    buckets = np.asarray([distance_bucket(int(distance)) for distance in distances], dtype=str)
    for name, labels in (("by_family", families), ("by_bfs_distance", buckets)):
        result[name] = {label: _state_summary(correct[labels == label], losses[labels == label])
                        for label in sorted(set(labels.tolist()))}
    return result


def rollout_metrics_grouped(rollouts, metadata) -> dict:
    result = dict(rollouts["summary"])
    families, distances = defaultdict(list), defaultdict(list)
    for row in rollouts["episodes"]:
        family = metadata["layouts"][row["layout_index"]].get("family", "unknown")
        families[family].append(row)
        distances[distance_bucket(row["optimal_steps"])].append(row)
    result["by_family"] = {key: _summary(rows) for key, rows in sorted(families.items())}
    result["by_bfs_distance"] = {key: _summary(rows) for key, rows in sorted(distances.items())}
    return result


def paired_outcomes(baseline_rows, candidate_rows) -> dict:
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("paired comparison requires identical episode definitions")
    counts = {"both_success": 0, "both_failed": 0,
              "candidate_gained_success": 0, "candidate_lost_success": 0}
    identity = ("layout_index", "goal_index", "start", "goal", "optimal_steps")
    for before, after in zip(baseline_rows, candidate_rows):
        if any(before[key] != after[key] for key in identity):
            raise ValueError("paired comparison requires identical episode definitions")
        a, b = before["outcome"] == "success", after["outcome"] == "success"
        key = ("both_success" if a and b else "both_failed" if not a and not b
               else "candidate_gained_success" if b else "candidate_lost_success")
        counts[key] += 1
    counts["episodes"] = len(baseline_rows)
    counts["net_success_count_change"] = (counts["candidate_gained_success"]
                                           - counts["candidate_lost_success"])
    return counts


def diagnostic_trace(model, maze_text: str, *, max_steps=64) -> dict:
    """Run raw model choices first; compute BFS only afterward for reporting."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    maze = Maze.from_ascii(maze_text)
    position, visited = maze.start, {maze.start}
    events, path = [], [list(position)]
    outcome = "success" if position == maze.goal else "timeout"
    mode_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            for index in range(max_steps):
                if position == maze.goal:
                    break
                inputs = torch.tensor(encode_state(maze, position), dtype=torch.float32).view(1, 3, 8, 8)
                scores = model(inputs)
                if not isinstance(scores, torch.Tensor) or scores.shape != (1, 4) or not torch.isfinite(scores).all():
                    raise ValueError("model must return finite [batch, 4] scores")
                action = int(scores[0].argmax())
                moved = step(maze, position, action)
                events.append({"attempt": index+1, "from": list(position),
                               "scores_up_right_down_left": scores[0].tolist(),
                               "action": Action(action).name, "to": list(moved.position)})
                position = moved.position
                path.append(list(position))
                if moved.done:
                    outcome = "success"
                    break
                if moved.blocked:
                    outcome = "blocked"
                    break
                if position in visited:
                    outcome = "loop"
                    break
                visited.add(position)
    finally:
        for module, mode in mode_states:
            module.training = mode
    reference = shortest_path(maze)
    return {"maze_ascii": maze.to_ascii(), "outcome": outcome,
            "attempts": len(events), "path": path, "events": events,
            "optimal_steps": len(reference)-1 if reference is not None else None}


def compare(baseline: Path, candidate: Path, data_dirs: list[Path], output_dir: Path,
            *, batch_size=512, seed=311, threads=1) -> dict:
    if not data_dirs:
        raise ValueError("at least one validation dataset is required")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("comparison output must be new or empty")
    # No model, source, or dataset writes occur during comparison.
    source_hashes = _source_hashes()
    models, model_records = {}, {}
    watched = {}
    for name, directory in (("baseline", Path(baseline)), ("candidate", Path(candidate))):
        models[name], manifest = load_model(directory)
        model_records[name] = {
            "directory": _path_label(directory),
            "weights_sha256": manifest["weights_sha256"],
            "manifest_sha256": sha256(directory / "manifest.json"),
            "config": manifest["config"],
            "training_dataset_manifest_sha256": manifest.get("provenance", {}).get("dataset_manifest_sha256"),
        }
        for filename in ("weights.npz", "manifest.json"):
            watched[directory / filename] = sha256(directory / filename)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    result = {
        "format": "mazebits-fp32-comparison-v1",
        "models": model_records,
        "source_sha256": source_hashes,
        "configuration": {"batch_size": batch_size, "seed": seed, "threads": threads,
                          "max_steps": 64, "action_selection": "raw argmax; no action masking or route repair",
                          "distance_groups": ["0-9", "10-19", "20+"]},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "device": "cpu"},
        "test_status": "reserved; final-test arrays and episode files not loaded",
        "diagnostic_status": "known development cases; excluded from aggregate validation metrics",
        "datasets": [],
    }
    for index, directory in enumerate(map(Path, data_dirs)):
        for filename in ("manifest.json", "val.npz", "val.json"):
            watched[directory / filename] = sha256(directory / filename)
        inputs, targets, metadata = load_split(directory, "val")
        with np.load(directory / "val.npz", allow_pickle=False) as arrays:
            layout_indices, distances = arrays["layout_index"].copy(), arrays["distances"].copy()
        record = {"directory": _path_label(directory), "split": "val",
                  "manifest_sha256": watched[directory / "manifest.json"],
                  "val_npz_sha256": watched[directory / "val.npz"],
                  "val_metadata_sha256": watched[directory / "val.json"],
                  "models": {}}
        traces = {}
        for name, model in models.items():
            states = state_metrics_grouped(model, inputs, targets, layout_indices, distances,
                                           metadata, batch_size=batch_size)
            rollouts = evaluate_rollouts(model, metadata, batch_size=batch_size)
            grouped = rollout_metrics_grouped(rollouts, metadata)
            traces[name] = rollouts["episodes"]
            filename = f"dataset-{index+1}-{name}-rollouts.json"
            write_json(output_dir / filename, {"summary": grouped, "episodes": traces[name]})
            record["models"][name] = {"states": states, "rollouts": grouped,
                                       "trajectory_file": filename,
                                       "trajectory_sha256": sha256(output_dir / filename)}
            print(json.dumps({"dataset": _path_label(directory), "model": name,
                              "success": grouped["success_count"], "episodes": grouped["episodes"],
                              "action_accuracy": states["optimal_action_accuracy"]}), flush=True)
        record["paired_outcomes"] = paired_outcomes(traces["baseline"], traces["candidate"])
        result["datasets"].append(record)
    diagnostics = {name: {model_name: diagnostic_trace(model, maze_text)
                         for model_name, model in models.items()}
                   for name, maze_text in DIAGNOSTIC_MAZES.items()}
    write_json(output_dir / "diagnostics.json", diagnostics)
    result["diagnostics_file"] = "diagnostics.json"
    result["diagnostics_sha256"] = sha256(output_dir / "diagnostics.json")
    result["elapsed_seconds"] = time.perf_counter() - start_time
    if any(sha256(path) != digest for path, digest in watched.items()) or _source_hashes() != source_hashes:
        raise RuntimeError("model, dataset, or evaluation source changed during comparison")
    result["inputs_unchanged_during_evaluation"] = True
    write_json(output_dir / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("model/cnn-fp32"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, action="append", required=True,
                        help="Repeat for each validation dataset; final-test split is never used")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()
    configure_cpu(args.seed, args.threads)
    compare(args.baseline, args.candidate, args.data_dir, args.output_dir,
            batch_size=args.batch_size, seed=args.seed, threads=args.threads)
    print(f"Saved comparison: {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
