"""The exact binary tensor contract that future training and RTL will share."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.encoding import CHANNELS, INPUT_VALUES, encode_state
from mazebits.examples import EXAMPLES
from mazebits.maze import Maze


class EncodingTests(unittest.TestCase):
    def test_exact_channel_and_coordinate_offsets(self):
        maze = Maze({(2, 1)}, (3, 4), (7, 6))
        encoded = encode_state(maze, maze.start)
        self.assertEqual(CHANNELS, 3)
        self.assertEqual(INPUT_VALUES, 192)
        self.assertIsInstance(encoded, tuple)
        self.assertEqual(len(encoded), 192)
        self.assertEqual(set(encoded), {0, 1})
        self.assertTrue(all(type(value) is int for value in encoded))
        self.assertEqual([i for i, value in enumerate(encoded) if value],
                         [10, 99, 183])

    def test_moving_agent_changes_only_its_plane(self):
        maze = EXAMPLES["detour"]
        initial = encode_state(maze, maze.start)
        moved = encode_state(maze, (0, 1))
        self.assertEqual(initial[:64], moved[:64])
        self.assertEqual(initial[128:], moved[128:])
        self.assertEqual(sum(initial[:64]), len(maze.walls))
        self.assertEqual(sum(initial[64:128]), 1)
        self.assertEqual(sum(moved[64:128]), 1)
        self.assertEqual(sum(initial[128:]), 1)
        self.assertEqual([i for i in range(192) if initial[i] != moved[i]],
                         [64, 72])

    def test_agent_and_goal_can_overlap_without_losing_a_channel(self):
        maze = EXAMPLES["open"]
        encoded = encode_state(maze, maze.goal)
        self.assertEqual(encoded[127], 1)
        self.assertEqual(encoded[191], 1)
        self.assertEqual(sum(encoded[64:128]), 1)
        self.assertEqual(sum(encoded[128:]), 1)

    def test_encoding_rejects_wall_and_out_of_bounds_positions(self):
        maze = EXAMPLES["detour"]
        for position in ((1, 0), (-1, 0), (8, 7), (False, 0)):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    encode_state(maze, position)


if __name__ == "__main__":
    unittest.main()
