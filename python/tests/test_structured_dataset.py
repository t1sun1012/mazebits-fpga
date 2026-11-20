"""Structured data: BFS correctness, family coverage, exclusions and reproducibility."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mazebits.dataset import canonical_layout_id, walls_from_bits
from mazebits.maze import Maze
from mazebits.reference import bfs_distances, optimal_actions
from mazebits.structured_dataset import (FAMILY_CYCLE, diagnostic_layout_ids,
                                        generate_structured_dataset)


class StructuredDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prior = cls.root / "prior.json"
        # A real canonical excluded layout, including D4 equivalents.
        cls.excluded = canonical_layout_id(frozenset({(1, 1), (2, 2), (4, 5)}))
        cls.prior.write_text(json.dumps({"splits": {"train": {"layout_ids": [cls.excluded]},
                                                   "val": {"layout_ids": []},
                                                   "test": {"layout_ids": []}}}))
        cls.config = dict(train_layouts=5, val_layouts=3, test_layouts=3, seed=312,
                          goals_per_layout=3, episodes_per_goal=3,
                          exclude_manifests=[cls.prior])
        cls.output = cls.root / "one"
        cls.manifest = generate_structured_dataset(cls.output, **cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_bfs_labels_encoding_and_episode_distances(self):
        for split in ("train", "val", "test"):
            meta = json.loads((self.output / f"{split}.json").read_text())
            with np.load(self.output / f"{split}.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["inputs"].shape[1:], (3, 8, 8))
                self.assertEqual(arrays["inputs"].dtype, np.uint8)
                self.assertEqual(arrays["layout_index"].dtype, np.int32)
                counts = {}
                for i in range(len(arrays["inputs"])):
                    li, gi = int(arrays["layout_index"][i]), int(arrays["goal_index"][i])
                    layout = meta["layouts"][li]
                    pos = tuple(map(int, arrays["positions"][i]))
                    maze = Maze(walls_from_bits(layout["walls"]), pos, tuple(layout["goals"][gi]))
                    self.assertEqual(set(np.flatnonzero(arrays["targets"][i])),
                                     set(map(int, optimal_actions(maze, pos))))
                    self.assertEqual(int(arrays["distances"][i]), bfs_distances(maze)[pos])
                    self.assertEqual(int(arrays["inputs"][i, 1].sum()), 1)
                    self.assertEqual(int(arrays["inputs"][i, 1, pos[1], pos[0]]), 1)
                    self.assertEqual(set(zip(*np.where(arrays["inputs"][i, 0]))),
                                     {(y, x) for x, y in maze.walls})
                    counts[(li, gi)] = counts.get((li, gi), 0) + 1
                for li, layout in enumerate(meta["layouts"]):
                    for gi, goal in enumerate(layout["goals"]):
                        maze = Maze(walls_from_bits(layout["walls"]), tuple(goal), tuple(goal))
                        self.assertEqual(counts[(li, gi)], len(bfs_distances(maze)) - 1)
            seen = set()
            for ep in meta["episodes"]:
                layout = meta["layouts"][ep["layout_index"]]
                maze = Maze(walls_from_bits(layout["walls"]), tuple(ep["start"]), tuple(ep["goal"]))
                self.assertEqual(bfs_distances(maze)[maze.start], ep["optimal_steps"])
                identity = ep["layout_index"], ep["goal_index"], maze.start
                self.assertNotIn(identity, seen)
                seen.add(identity)

    def test_family_coverage_and_actual_longest_start(self):
        for split in ("train", "val", "test"):
            meta = json.loads((self.output / f"{split}.json").read_text())
            self.assertEqual([layout["family"] for layout in meta["layouts"]],
                             [FAMILY_CYCLE[i % 5] for i in range(len(meta["layouts"]))])
            self.assertGreaterEqual(self.manifest["splits"][split]["distance_max"], 26)
            self.assertGreater(self.manifest["splits"][split]["by_family"]["corridor"]
                               ["states_distance_at_least_26"], 0)
            for li, layout in enumerate(meta["layouts"]):
                if layout["family"] == "corridor":
                    self.assertGreaterEqual(layout["sampled_endpoint_distance"], 26)
                for gi, goal in enumerate(layout["goals"]):
                    maze = Maze(walls_from_bits(layout["walls"]), tuple(goal), tuple(goal))
                    longest = max(bfs_distances(maze).values())
                    episodes = [e for e in meta["episodes"] if e["layout_index"] == li and e["goal_index"] == gi]
                    self.assertEqual(max(e["optimal_steps"] for e in episodes), longest)

    def test_prior_diagnostics_and_all_split_d4_ids_are_disjoint(self):
        seen = diagnostic_layout_ids() | {self.excluded}
        self.assertIn(self.excluded, self.manifest["exclusions"]["layout_ids"])
        for split in ("train", "val", "test"):
            meta = json.loads((self.output / f"{split}.json").read_text())
            for layout in meta["layouts"]:
                walls = walls_from_bits(layout["walls"])
                identity = canonical_layout_id(walls)
                self.assertNotIn(identity, seen)
                self.assertEqual(identity, layout["id"])
                self.assertEqual(identity, canonical_layout_id(frozenset((7 - y, x) for x, y in walls)))
                seen.add(identity)
        self.assertEqual(self.manifest["audit"]["accepted_unique_layouts"], 11)
        # Force an excluded proposal: it must be rejected, not assigned a split.
        walls = frozenset({(1, 1), (2, 2), (4, 5)})
        with patch("mazebits.structured_dataset._proposal", return_value=(walls, .1)):
            with self.assertRaisesRegex(RuntimeError, "within 2 attempts"):
                generate_structured_dataset(self.root / "forced", train_layouts=1, val_layouts=1,
                                            test_layouts=1, exclude_manifests=[self.prior],
                                            max_attempts_per_layout=2)

    def test_rebuild_is_byte_identical_and_existing_output_safe(self):
        second = self.root / "two"
        manifest = generate_structured_dataset(second, **self.config)
        self.assertEqual(manifest, self.manifest)
        for p in self.output.iterdir():
            self.assertEqual(p.read_bytes(), (second / p.name).read_bytes())
        with self.assertRaises(FileExistsError):
            generate_structured_dataset(self.output, **self.config)

    def test_invalid_configuration_and_exclusions_fail_before_output(self):
        cases = ({"train_layouts": 0}, {"seed": -1}, {"min_component": 65},
                 {"goals_per_layout": 25}, {"max_attempts_per_layout": 0},
                 {"max_attempts_per_layout": True}, {"excluded_layout_ids": ["bad"]})
        for i, config in enumerate(cases):
            output = self.root / f"invalid{i}"
            with self.subTest(config=config), self.assertRaises(ValueError):
                generate_structured_dataset(output, **config)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
