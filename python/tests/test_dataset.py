"""Dataset correctness, whole-layout separation, and artifact reproducibility."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.dataset import DatasetConfig, canonical_layout_id, generate_dataset, walls_from_bits
from mazebits.encoding import encode_state
from mazebits.maze import Maze
from mazebits.reference import bfs_distances, optimal_actions


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name) / "first"
        cls.config = dict(train_layouts=2, val_layouts=2, test_layouts=2,
                          seed=311, goals_per_layout=2, episodes_per_goal=3)
        cls.manifest = generate_dataset(cls.folder, **cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_shapes_encoding_and_all_teacher_targets(self):
        for split in ("train", "val", "test"):
            metadata = json.loads((self.folder / f"{split}.json").read_text())
            with np.load(self.folder / f"{split}.npz", allow_pickle=False) as arrays:
                count = len(arrays["inputs"])
                self.assertEqual(arrays["inputs"].shape, (count, 3, 8, 8))
                self.assertEqual(arrays["targets"].shape, (count, 4))
                self.assertEqual(arrays["inputs"].dtype, np.uint8)
                self.assertEqual(arrays["layout_index"].dtype, np.int32)
                self.assertTrue(np.isin(arrays["targets"], [0, 1]).all())
                observed = set()
                for index in range(count):
                    li, gi = int(arrays["layout_index"][index]), int(arrays["goal_index"][index])
                    layout = metadata["layouts"][li]
                    position = tuple(int(value) for value in arrays["positions"][index])
                    goal = tuple(layout["goals"][gi])
                    maze = Maze(walls_from_bits(layout["walls"]), position, goal)
                    self.assertNotEqual(position, goal)
                    np.testing.assert_array_equal(
                        arrays["inputs"][index].reshape(-1), encode_state(maze, position))
                    expected = set(map(int, optimal_actions(maze, position)))
                    self.assertEqual(set(np.flatnonzero(arrays["targets"][index])), expected)
                    self.assertEqual(int(arrays["distances"][index]),
                                     bfs_distances(maze)[position])
                    self.assertNotIn((li, gi, position), observed)
                    observed.add((li, gi, position))
                expected_count = sum(
                    len(bfs_distances(Maze(walls_from_bits(layout["walls"]),
                                          tuple(goal), tuple(goal)))) - 1
                    for layout in metadata["layouts"] for goal in layout["goals"])
                self.assertEqual(count, expected_count)

    def test_layouts_and_d4_equivalents_are_disjoint_across_splits(self):
        seen = set()
        for split in ("train", "val", "test"):
            metadata = json.loads((self.folder / f"{split}.json").read_text())
            for layout in metadata["layouts"]:
                walls = walls_from_bits(layout["walls"])
                identity = canonical_layout_id(walls)
                self.assertEqual(identity, layout["id"])
                self.assertNotIn(identity, seen)
                seen.add(identity)
                reflected = frozenset((7 - x, y) for x, y in walls)
                rotated = frozenset((7 - y, x) for x, y in walls)
                self.assertEqual(canonical_layout_id(reflected), identity)
                self.assertEqual(canonical_layout_id(rotated), identity)
        self.assertEqual(len(seen), 6)

    def test_episodes_are_unique_reachable_and_correctly_labeled(self):
        for split in ("train", "val", "test"):
            metadata = json.loads((self.folder / f"{split}.json").read_text())
            seen = set()
            self.assertEqual(len(metadata["episodes"]), 12)
            for episode in metadata["episodes"]:
                li, gi = episode["layout_index"], episode["goal_index"]
                layout = metadata["layouts"][li]
                self.assertEqual(episode["goal"], layout["goals"][gi])
                start, goal = tuple(episode["start"]), tuple(episode["goal"])
                maze = Maze(walls_from_bits(layout["walls"]), start, goal)
                self.assertEqual(bfs_distances(maze)[start], episode["optimal_steps"])
                self.assertNotIn((li, gi, start), seen)
                self.assertNotEqual(start, goal)
                seen.add((li, gi, start))

    def test_repeated_generation_is_byte_identical(self):
        second = Path(self.temporary.name) / "second"
        other = generate_dataset(second, **self.config)
        self.assertEqual(self.manifest, other)
        for path in self.folder.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())

    def test_existing_output_is_not_overwritten(self):
        original = (self.folder / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            generate_dataset(self.folder, **self.config)
        self.assertEqual((self.folder / "manifest.json").read_bytes(), original)

    def test_invalid_configuration_and_wall_encoding(self):
        for config in (dict(train_layouts=0), dict(seed=-1), dict(min_component=65),
                       dict(goals_per_layout=25), dict(episodes_per_goal=24),
                       dict(train_layouts=True)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                DatasetConfig(**config)
        for bits in ("0" * 63, "0" * 63 + "x"):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                walls_from_bits(bits)
        self.assertEqual(walls_from_bits("0" * 10 + "1" + "0" * 53),
                         frozenset({(2, 1)}))


if __name__ == "__main__":
    unittest.main()
