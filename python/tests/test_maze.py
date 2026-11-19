"""Contract checks for the environment, independent of model training or RTL."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.examples import EXAMPLES
from mazebits.maze import Action, Maze, legal_actions, neighbor, render_ascii, step


class MazeContractTests(unittest.TestCase):
    def test_action_codes_and_coordinates(self):
        self.assertEqual(
            [(a.name, int(a)) for a in Action],
            [("UP", 0), ("RIGHT", 1), ("DOWN", 2), ("LEFT", 3)],
        )
        self.assertEqual(
            [neighbor((3, 4), a) for a in Action],
            [(3, 3), (4, 4), (3, 5), (2, 4)],
        )

    def test_example_ascii_round_trip(self):
        for name, maze in EXAMPLES.items():
            with self.subTest(name=name):
                self.assertEqual(Maze.from_ascii(maze.to_ascii()), maze)
                crlf_text = "\r\n" + maze.to_ascii().replace("\n", "\r\n") + "\r\n"
                self.assertEqual(Maze.from_ascii(crlf_text), maze)

    def test_parser_rejects_malformed_maps(self):
        valid = EXAMPLES["open"].to_ascii()
        invalid = [
            "\n".join(valid.splitlines()[:-1]),
            valid.replace("S.......", "S......"),
            valid.replace("S", "."),
            valid.replace("G", "."),
            valid.replace("S.......", "SS......"),
            valid.replace(".......G", "......GG"),
            valid.replace("S", "B"),  # B plus a second goal.
            valid.replace("S", "?"),
        ]
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    Maze.from_ascii(text)

    def test_constructor_rejects_invalid_coordinates_and_walls(self):
        invalid = [
            ({(8, 0)}, (0, 0), (7, 7)),
            ({(-1, 0)}, (0, 0), (7, 7)),
            ({(1.0, 1)}, (0, 0), (7, 7)),
            (set(), (True, 0), (7, 7)),
            (set(), (0, 0), (7, 8)),
            (set(), [0, 0], (7, 7)),
            ({(0, 0)}, (0, 0), (7, 7)),
            ({(7, 7)}, (0, 0), (7, 7)),
        ]
        for walls, start, goal in invalid:
            with self.subTest(walls=walls, start=start, goal=goal):
                with self.assertRaises(ValueError):
                    Maze(walls, start, goal)

    def test_maze_copies_mutable_walls_and_is_frozen(self):
        caller_walls = {(2, 2)}
        maze = Maze(caller_walls, (0, 0), (7, 7))
        caller_walls.clear()
        caller_walls.add((3, 3))
        self.assertEqual(maze.walls, frozenset({(2, 2)}))
        with self.assertRaises(FrozenInstanceError):
            maze.start = (1, 1)

    def test_each_outer_boundary_blocks_without_wrapping(self):
        maze = Maze(frozenset(), (0, 0), (4, 4))
        cases = [
            ((3, 0), Action.UP),
            ((7, 3), Action.RIGHT),
            ((3, 7), Action.DOWN),
            ((0, 3), Action.LEFT),
        ]
        for position, action in cases:
            with self.subTest(position=position, action=action):
                result = step(maze, position, action)
                self.assertEqual(result.position, position)
                self.assertTrue(result.blocked)
                self.assertFalse(result.done)
                self.assertNotIn(action, legal_actions(maze, position))

    def test_wall_collision_and_legal_move(self):
        maze = Maze({(2, 1)}, (1, 1), (7, 7))
        blocked = step(maze, maze.start, Action.RIGHT)
        self.assertEqual((blocked.position, blocked.blocked, blocked.done),
                         ((1, 1), True, False))
        moved = step(maze, maze.start, Action.DOWN)
        self.assertEqual((moved.position, moved.blocked, moved.done),
                         ((1, 2), False, False))
        self.assertEqual(legal_actions(maze, maze.start),
                         (Action.UP, Action.DOWN, Action.LEFT))

    def test_reaching_goal_and_terminal_actions(self):
        maze = Maze(frozenset(), (1, 1), (2, 1))
        reached = step(maze, maze.start, Action.RIGHT)
        self.assertEqual((reached.position, reached.blocked, reached.done),
                         (maze.goal, False, True))
        self.assertEqual(legal_actions(maze, maze.goal), ())
        for action in Action:
            with self.subTest(action=action):
                terminal = step(maze, maze.goal, action)
                self.assertEqual((terminal.position, terminal.blocked, terminal.done),
                                 (maze.goal, False, True))

    def test_invalid_actions_are_errors_even_at_goal(self):
        maze = EXAMPLES["open"]
        for position in (maze.start, maze.goal):
            for action in (-1, 4, 1.0, "1", True, None):
                with self.subTest(position=position, action=action):
                    with self.assertRaises(ValueError):
                        step(maze, position, action)

    def test_invalid_current_positions_are_rejected(self):
        maze = EXAMPLES["detour"]
        for position in ((-1, 0), (8, 0), (1, 0), (0.0, 0)):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    step(maze, position, Action.DOWN)
                with self.assertRaises(ValueError):
                    legal_actions(maze, position)

    def test_rendering_preserves_maze_and_marks_agent_and_path(self):
        maze = EXAMPLES["open"]
        original = maze.to_ascii()
        rendered = render_ascii(maze, position=(2, 0),
                                path=((0, 0), (1, 0), (2, 0)))
        self.assertEqual(rendered.splitlines()[0], "S*A.....")
        self.assertEqual(rendered.splitlines()[7], ".......G")
        self.assertEqual(maze.to_ascii(), original)
        self.assertEqual(render_ascii(maze, position=maze.goal).splitlines()[7],
                         ".......X")


if __name__ == "__main__":
    unittest.main()
