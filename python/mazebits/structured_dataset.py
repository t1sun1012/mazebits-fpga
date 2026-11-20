"""Structured BFS-imitation data with whole-layout exclusion and deterministic files.

Version 2 retains the original binary input/target contract. Its generators add
connected barriers and long winding routes; no generated shortest path is a CNN
input. The legacy v1 generator is intentionally unchanged.
"""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from .dataset import (DatasetConfig, _episode_starts, _largest_component,
                      _sha256, _wall_bits, _write_json, _write_npz,
                      canonical_layout_id)
from .encoding import encode_state
from .examples import EXAMPLES
from .maze import Action, Maze, SIZE, neighbor
from .reference import bfs_distances

# These known board demonstrations are diagnostic holdouts, never generated
# training/validation/test layouts. They are already known, not unseen tests.
DIAGNOSTIC_MAPS = {
    "IMG_7179": "S#......\n.#......\n.#......\n.#.....G\n.#......\n.#......\n........\n........",
    "IMG_7180": "S#G.....\n.######.\n.#......\n.#......\n.#......\n.#......\n.#......\n........",
    "quantization-loop-photo": "S.......\n...####.\n...#....\n...#....\n...#....\n...#....\n...#....\n.......G",
}
FAMILY_CYCLE = ("bernoulli", "barrier", "corridor", "bernoulli", "corridor")


def diagnostic_layout_ids():
    return {canonical_layout_id(m.walls) for m in EXAMPLES.values()} | {
        canonical_layout_id(Maze.from_ascii(text).walls)
        for text in DIAGNOSTIC_MAPS.values()}


def _transform(walls, rng):
    if rng.randrange(2):
        walls = {(7 - x, y) for x, y in walls}
    for _ in range(rng.randrange(4)):
        walls = {(7 - y, x) for x, y in walls}
    return frozenset(walls)


def _proposal(family, rng):
    if family == "bernoulli":
        density = rng.choice((0.1, 0.2, 0.3))
        return frozenset((x, y) for y in range(8) for x in range(8)
                         if rng.random() < density), density
    if family == "barrier":
        shape = rng.choice(("divider", "elbow", "u"))
        if shape == "divider":
            x = rng.randrange(1, 7)
            walls = {(x, y) for y in range(rng.randrange(4, 8))}
        elif shape == "elbow":
            x = rng.randrange(1, 5)
            corner = rng.randrange(1, 5)
            walls = {(x, y) for y in range(rng.randrange(5, 8))}
            walls |= {(xx, corner) for xx in range(x, rng.randrange(x + 2, 9))}
        else:
            left = rng.randrange(0, 4)
            right = rng.randrange(left + 3, 8)
            top = rng.randrange(0, 3)
            bottom = rng.randrange(top + 3, 8)
            walls = {(x, top) for x in range(left, right + 1)}
            walls |= {(x, y) for x in (left, right) for y in range(top, bottom + 1)}
        noise = rng.choice((0.02, 0.05, 0.09))
    elif family == "corridor":
        # Alternating gaps force travel along each separator. Isolated extra
        # obstacles diversify topology without introducing route shortcuts.
        columns = rng.choice(((1, 3, 5), (2, 4, 6), (1, 3, 6), (1, 4, 6),
                              (1, 5), (2, 6)))
        high_first = bool(rng.randrange(2))
        walls = set()
        for index, x in enumerate(columns):
            high = high_first != bool(index % 2)
            gap = rng.choice((6, 7) if high else (0, 1))
            walls.update((x, y) for y in range(8) if y != gap)
        noise = rng.choice((0.03, 0.06, 0.10))
    else:
        raise ValueError(f"unknown maze family: {family}")
    walls |= {(x, y) for y in range(8) for x in range(8)
              if (x, y) not in walls and rng.random() < noise}
    return _transform(walls, rng), None


def _farthest_goals(walls, component, count, rng):
    """Include two sweep endpoints, then random distinct goals.

    Two BFS sweeps choose far-apart endpoints; they do not claim an exact graph
    diameter. The first two returned goals each have the reported separation.
    """
    anchor = rng.choice(component)
    distances = bfs_distances(Maze(walls, anchor, anchor))
    maximum = max(distances.values())
    first = rng.choice(sorted(cell for cell, distance in distances.items()
                              if distance == maximum))
    distances = bfs_distances(Maze(walls, first, first))
    maximum = max(distances.values())
    second = rng.choice(sorted(cell for cell, distance in distances.items()
                               if distance == maximum))
    goals = [first]
    if count > 1:
        goals.append(second)
    if count > 2:
        goals.extend(rng.sample([cell for cell in component if cell not in goals], count - 2))
    return goals, maximum


def _validate_id(value):
    if (not isinstance(value, str) or len(value) != 16
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("excluded layout IDs must be 16 lowercase hexadecimal characters")
    return value


def _exclusions(manifests, maze_files, supplied_ids):
    excluded = diagnostic_layout_ids()
    sources = []
    for source in manifests:
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("splits"):
            raise ValueError(f"exclusion manifest has no splits: {path}")
        ids = {_validate_id(value) for split in data["splits"].values()
               for value in split["layout_ids"]}
        excluded.update(ids)
        sources.append({"kind": "dataset_manifest", "name": path.name,
                        "sha256": _sha256(path), "layout_count": len(ids)})
    for source in maze_files:
        path = Path(source)
        maze = Maze.from_ascii(path.read_text(encoding="utf-8"))
        identity = canonical_layout_id(maze.walls)
        excluded.add(identity)
        sources.append({"kind": "maze_file", "name": path.name,
                        "sha256": _sha256(path), "layout_id": identity})
    excluded.update(_validate_id(value) for value in supplied_ids)
    return excluded, sources


def _save_split(destination, split, layouts, rng, config):
    arrays_lists = {key: [] for key in
                   ("inputs", "targets", "layout_index", "goal_index", "positions", "distances")}
    metadata = {"schema_version": 2, "split": split, "layouts": layouts, "episodes": []}
    family_distances = {}
    for li, layout in enumerate(layouts):
        walls = frozenset((i % 8, i // 8) for i, bit in enumerate(layout["walls"]) if bit == "1")
        stats = family_distances.setdefault(layout["family"], [])
        for gi, goal_list in enumerate(layout["goals"]):
            goal = tuple(goal_list)
            maze = Maze(walls, goal, goal)
            distances = bfs_distances(maze)
            for position in sorted(distances, key=lambda c: (c[1], c[0])):
                distance = distances[position]
                if distance == 0:
                    continue
                target = [int(distances.get(neighbor(position, a)) == distance - 1) for a in Action]
                if not any(target):
                    raise RuntimeError("reachable nonterminal state lacks a BFS action")
                arrays_lists["inputs"].append(encode_state(maze, position))
                arrays_lists["targets"].append(target)
                arrays_lists["layout_index"].append(li)
                arrays_lists["goal_index"].append(gi)
                arrays_lists["positions"].append(position)
                arrays_lists["distances"].append(distance)
                stats.append(distance)
            starts = _episode_starts(distances, config.episodes_per_goal, rng)
            # Include the hardest starting distance, rather than merely a random
            # sample from its rank bin. Earlier starts occupy disjoint bins.
            farthest = sorted(cell for cell, d in distances.items() if d == max(distances.values()))
            replacement = rng.choice([cell for cell in farthest if cell not in starts[:-1]])
            starts[-1] = replacement
            for start in starts:
                metadata["episodes"].append({"layout_index": li, "goal_index": gi,
                    "start": list(start), "goal": list(goal), "optimal_steps": distances[start]})
    arrays = {name: np.asarray(values, dtype=np.int32 if name in
                              ("layout_index", "goal_index") else np.uint8)
              for name, values in arrays_lists.items()}
    arrays["inputs"] = arrays["inputs"].reshape(-1, 3, SIZE, SIZE)
    npz_path, json_path = destination / f"{split}.npz", destination / f"{split}.json"
    _write_npz(npz_path, arrays)
    _write_json(json_path, metadata)
    multiplicities = Counter(map(int, arrays["targets"].sum(axis=1)))
    families = {}
    for family, distances in sorted(family_distances.items()):
        count = sum(layout["family"] == family for layout in layouts)
        families[family] = {"layouts": count, "goals": count * config.goals_per_layout,
                           "samples": len(distances), "episodes": count * config.goals_per_layout * config.episodes_per_goal,
                           "distance_min": min(distances), "distance_max": max(distances),
                           "distance_mean": sum(distances) / len(distances),
                           "states_distance_at_least_26": sum(d >= 26 for d in distances)}
    return {"layouts": len(layouts), "goals": len(layouts) * config.goals_per_layout,
            "samples": len(arrays["inputs"]), "episodes": len(metadata["episodes"]),
            "layout_ids": [layout["id"] for layout in layouts],
            "action_positive_counts": arrays["targets"].sum(axis=0).tolist(),
            "optimal_action_count_histogram": {str(k): v for k, v in sorted(multiplicities.items())},
            "distance_min": int(arrays["distances"].min()), "distance_max": int(arrays["distances"].max()),
            "distance_mean": float(arrays["distances"].mean()),
            "actual_wall_density_mean": sum(layout["wall_count"] / 64 for layout in layouts) / len(layouts),
            "npz_file": npz_path.name, "metadata_file": json_path.name,
            "by_family": families}


def generate_structured_dataset(output_dir, *, train_layouts=1200, val_layouts=200,
                                test_layouts=200, seed=312, goals_per_layout=3,
                                episodes_per_goal=8, min_component=24,
                                exclude_manifests=(), exclude_maze_files=(),
                                excluded_layout_ids=(), max_attempts_per_layout=1000):
    """Create deterministic v2 data without overwrites or whole-layout leakage.

    Pass the v1 audit manifest in exclude_manifests (the CLI default) to exclude
    every old train/validation/reserved-test layout. Only its IDs/hashes are
    read; old or new reserved-test examples are never used for selection.
    """
    config = DatasetConfig(train_layouts, val_layouts, test_layouts, seed,
                           goals_per_layout, episodes_per_goal, min_component)
    if type(max_attempts_per_layout) is not int or max_attempts_per_layout < 1:
        raise ValueError("max_attempts_per_layout must be a positive integer")
    excluded, sources = _exclusions(exclude_manifests, exclude_maze_files, excluded_layout_ids)
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen = set(excluded)
    audit = Counter()
    manifest = {
        "schema_version": 2, "config": {**asdict(config), "max_attempts_per_layout": max_attempts_per_layout},
        "distribution": {"family": "structured mixture",
            "family_cycle": list(FAMILY_CYCLE),
            "bernoulli_wall_probabilities": [0.1, 0.2, 0.3],
            "barrier_shapes": ["divider", "elbow", "u"],
            "corridor": "two or three separators with alternating end gaps; sampled endpoint distance at least 26",
            "goal_selection": "two far-apart BFS-sweep endpoints then distinct random goals",
            "example_selection": "all reachable nonterminal cells for every selected goal",
            "episode_selection": "BFS-distance rank bins, final bin replaced by an actual farthest start"},
        "input_contract": {"shape": [3, 8, 8], "dtype": "uint8", "values": [0, 1], "channels": ["walls", "agent", "goal"]},
        "target_contract": {"shape": [4], "dtype": "uint8", "actions": ["UP", "RIGHT", "DOWN", "LEFT"],
                            "meaning": "1 for every shortest-path action", "goal_index": "zero-based within layout"},
        "split_policy": "globally unique wall layouts including D4 equivalents, excluding all supplied prior split IDs and known diagnostics",
        "exclusions": {"sources": sources, "layout_ids": sorted(excluded),
                       "known_diagnostic_ids": sorted(diagnostic_layout_ids()),
                       "known_diagnostic_status": "excluded but already observed; not unbiased unseen test evidence"},
        "splits": {}, "files": {}, "test_status": "generated, reserved for final evaluation"}
    for split, count in (("train", train_layouts), ("val", val_layouts), ("test", test_layouts)):
        layouts = []
        for index in range(count):
            family = FAMILY_CYCLE[index % len(FAMILY_CYCLE)]
            for attempt in range(max_attempts_per_layout):
                audit["candidate_layouts"] += 1
                walls, requested_density = _proposal(family, rng)
                identity = canonical_layout_id(walls)
                if identity in seen:
                    audit["rejected_excluded_or_duplicate_d4"] += 1
                    continue
                component = _largest_component(walls)
                if len(component) < min_component:
                    audit["rejected_small_component"] += 1
                    continue
                goals, separation = _farthest_goals(walls, component, goals_per_layout, rng)
                if family == "corridor" and separation < 26:
                    audit["rejected_short_corridor"] += 1
                    continue
                seen.add(identity)
                layouts.append({"id": identity, "walls": _wall_bits(walls), "family": family,
                                "requested_density": requested_density, "wall_count": len(walls),
                                "largest_component_size": len(component), "sampled_endpoint_distance": separation,
                                "goals": [list(goal) for goal in goals]})
                break
            else:
                raise RuntimeError(f"unable to sample {split} layout {index} ({family}) within {max_attempts_per_layout} attempts")
        manifest["splits"][split] = _save_split(destination, split, layouts, rng, config)
        for path in (destination / f"{split}.npz", destination / f"{split}.json"):
            manifest["files"][path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest["audit"] = {**dict(audit), "accepted_unique_layouts": len(seen) - len(excluded),
                         "excluded_layouts": len(excluded), "cross_split_exact_or_d4_overlap": 0,
                         "overlap_with_exclusions": 0}
    _write_json(destination / "manifest.json", manifest)
    return manifest
