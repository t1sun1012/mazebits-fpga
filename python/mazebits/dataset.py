"""Reproducible, layout-separated BFS imitation-learning examples."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np

from .encoding import encode_state
from .maze import Action, Cell, Maze, SIZE, neighbor
from .reference import bfs_distances


@dataclass(frozen=True)
class DatasetConfig:
    train_layouts: int = 800
    val_layouts: int = 150
    test_layouts: int = 150
    seed: int = 311
    goals_per_layout: int = 3
    episodes_per_goal: int = 8
    min_component: int = 24

    def __post_init__(self) -> None:
        for name in ("train_layouts", "val_layouts", "test_layouts",
                     "goals_per_layout", "episodes_per_goal"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.min_component) is not int or not 2 <= self.min_component <= 64:
            raise ValueError("min_component must be an integer between 2 and 64")
        if self.goals_per_layout > self.min_component:
            raise ValueError("goals_per_layout cannot exceed min_component")
        if self.episodes_per_goal >= self.min_component:
            raise ValueError("episodes_per_goal must be smaller than min_component")


def walls_from_bits(bits: str) -> frozenset[Cell]:
    """Decode 64 characters in y*8+x order; 1 means wall."""
    if len(bits) != SIZE * SIZE or any(bit not in "01" for bit in bits):
        raise ValueError("walls must be exactly 64 binary characters")
    return frozenset((index % SIZE, index // SIZE)
                     for index, bit in enumerate(bits) if bit == "1")


def _wall_bits(walls: frozenset[Cell]) -> str:
    return "".join("1" if (x, y) in walls else "0"
                   for y in range(SIZE) for x in range(SIZE))


def canonical_layout_id(walls: frozenset[Cell]) -> str:
    """One ID for every rotation/reflection of a wall layout (the D4 group)."""
    if any(not isinstance(cell, tuple) or len(cell) != 2
           or any(type(value) is not int or not 0 <= value < SIZE for value in cell)
           for cell in walls):
        raise ValueError("wall coordinates must be in-bounds integer pairs")
    variants = []
    for reflected in (False, True):
        transformed = frozenset((SIZE - 1 - x if reflected else x, y)
                                for x, y in walls)
        for _ in range(4):
            variants.append(_wall_bits(transformed))
            transformed = frozenset((SIZE - 1 - y, x) for x, y in transformed)
    return f"{int(min(variants), 2):016x}"


def _largest_component(walls: frozenset[Cell]) -> tuple[Cell, ...]:
    remaining = {(x, y) for y in range(SIZE) for x in range(SIZE)} - walls
    largest: tuple[Cell, ...] = ()
    while remaining:
        anchor = min(remaining, key=lambda cell: (cell[1], cell[0]))
        component = tuple(sorted(bfs_distances(Maze(walls, anchor, anchor)),
                                 key=lambda cell: (cell[1], cell[0])))
        remaining.difference_update(component)
        if len(component) > len(largest):
            largest = component
    return largest


def _episode_starts(
    distances: dict[Cell, int], count: int, rng: random.Random
) -> list[Cell]:
    """One random start per equal-size rank bin, ordered by BFS distance."""
    candidates = sorted((cell for cell, distance in distances.items() if distance > 0),
                        key=lambda cell: (distances[cell], cell[1], cell[0]))
    count = min(count, len(candidates))
    return [rng.choice(candidates[index * len(candidates) // count:
                                  (index + 1) * len(candidates) // count])
            for index in range(count)]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Use fixed ZIP metadata so repeated builds have identical artifact bytes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3  # Keep ZIP metadata identical on Windows and Linux.
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=6)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_dataset(
    output_dir: str | Path, *, train_layouts: int = 800, val_layouts: int = 150,
    test_layouts: int = 150, seed: int = 311, goals_per_layout: int = 3,
    episodes_per_goal: int = 8, min_component: int = 24
) -> dict:
    """Generate all splits without overwriting existing data.

    goal_index is local to its layout. Test files are generated but should remain
    unopened by model-selection/training code until the final evaluation.
    """
    config = DatasetConfig(train_layouts, val_layouts, test_layouts, seed,
                           goals_per_layout, episodes_per_goal, min_component)
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen: set[str] = set()
    audit = {"candidate_layouts": 0, "rejected_small_component": 0,
             "rejected_duplicate_or_d4_equivalent": 0}
    manifest = {
        "schema_version": 1, "config": asdict(config),
        "distribution": {
            "family": "independent Bernoulli obstacle cells",
            "requested_wall_probabilities": [0.1, 0.2, 0.3],
            "probability_selection": "uniform",
            "goal_selection": "distinct cells sampled from largest free component",
            "example_selection": "all reachable nonterminal cells for each goal",
            "episode_selection": "one random start per equal-size BFS-distance rank bin",
        },
        "input_contract": {"shape": [3, 8, 8], "dtype": "uint8", "values": [0, 1],
                           "channels": ["walls", "agent", "goal"]},
        "target_contract": {"shape": [4], "dtype": "uint8",
                            "actions": ["UP", "RIGHT", "DOWN", "LEFT"],
                            "meaning": "1 for every shortest-path action",
                            "goal_index": "zero-based within layout"},
        "split_policy": "globally unique wall layouts including D4 equivalents; all goals and states stay in their layout split",
        "splits": {}, "files": {}, "test_status": "generated, reserved for final evaluation",
    }
    for split, count in (("train", train_layouts), ("val", val_layouts),
                         ("test", test_layouts)):
        arrays_lists: dict[str, list] = {key: [] for key in (
            "inputs", "targets", "layout_index", "goal_index", "positions", "distances")}
        metadata = {"schema_version": 1, "split": split, "layouts": [], "episodes": []}
        while len(metadata["layouts"]) < count:
            audit["candidate_layouts"] += 1
            if audit["candidate_layouts"] > 1000 * (train_layouts + val_layouts + test_layouts):
                raise RuntimeError("unable to sample enough unique eligible layouts")
            probability = rng.choice((0.1, 0.2, 0.3))
            walls = frozenset((x, y) for y in range(SIZE) for x in range(SIZE)
                              if rng.random() < probability)
            layout_id = canonical_layout_id(walls)
            if layout_id in seen:
                audit["rejected_duplicate_or_d4_equivalent"] += 1
                continue
            component = _largest_component(walls)
            if len(component) < min_component:
                audit["rejected_small_component"] += 1
                continue
            seen.add(layout_id)
            layout_index = len(metadata["layouts"])
            goals = rng.sample(component, goals_per_layout)
            metadata["layouts"].append({
                "id": layout_id, "walls": _wall_bits(walls), "family": "bernoulli",
                "requested_density": probability, "wall_count": len(walls),
                "largest_component_size": len(component),
                "goals": [list(goal) for goal in goals],
            })
            for goal_index, goal in enumerate(goals):
                maze = Maze(walls, goal, goal)
                distances = bfs_distances(maze)
                for position in sorted(distances, key=lambda cell: (cell[1], cell[0])):
                    distance = distances[position]
                    if distance == 0:
                        continue
                    target = [int(distances.get(neighbor(position, action)) == distance - 1)
                              for action in Action]
                    if not any(target):
                        raise RuntimeError("reachable nonterminal cell has no optimal action")
                    arrays_lists["inputs"].append(encode_state(maze, position))
                    arrays_lists["targets"].append(target)
                    arrays_lists["layout_index"].append(layout_index)
                    arrays_lists["goal_index"].append(goal_index)
                    arrays_lists["positions"].append(position)
                    arrays_lists["distances"].append(distance)
                for start in _episode_starts(distances, episodes_per_goal, rng):
                    metadata["episodes"].append({
                        "layout_index": layout_index, "goal_index": goal_index,
                        "start": list(start), "goal": list(goal),
                        "optimal_steps": distances[start],
                    })
        arrays = {key: np.asarray(values, dtype=np.int32 if key in
                                  ("layout_index", "goal_index") else np.uint8)
                  for key, values in arrays_lists.items()}
        arrays["inputs"] = arrays["inputs"].reshape(-1, 3, SIZE, SIZE)
        npz_path, json_path = destination / f"{split}.npz", destination / f"{split}.json"
        _write_npz(npz_path, arrays)
        _write_json(json_path, metadata)
        multiplicities = Counter(int(value) for value in arrays["targets"].sum(axis=1))
        densities = [layout["wall_count"] / 64 for layout in metadata["layouts"]]
        manifest["splits"][split] = {
            "layouts": count, "goals": count * goals_per_layout,
            "samples": len(arrays["inputs"]), "episodes": len(metadata["episodes"]),
            "layout_ids": [layout["id"] for layout in metadata["layouts"]],
            "action_positive_counts": arrays["targets"].sum(axis=0).tolist(),
            "optimal_action_count_histogram": {str(key): value for key, value
                                               in sorted(multiplicities.items())},
            "distance_min": int(arrays["distances"].min()),
            "distance_max": int(arrays["distances"].max()),
            "distance_mean": float(arrays["distances"].mean()),
            "actual_wall_density_mean": float(sum(densities) / len(densities)),
            "npz_file": npz_path.name, "metadata_file": json_path.name,
        }
        for path in (npz_path, json_path):
            manifest["files"][path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest["audit"] = {**audit, "accepted_unique_layouts": len(seen),
                         "cross_split_exact_or_d4_overlap": 0}
    _write_json(destination / "manifest.json", manifest)
    return manifest
