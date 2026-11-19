"""Shortest-path teacher checks, including an independent small-graph oracle."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazebits.examples import EXAMPLES
from mazebits.maze import Action, Maze, step
from mazebits.reference import bfs_distances, optimal_actions, shortest_path


class ReferenceTests(unittest.TestCase):
    def test_empty_grid_manhattan_distance_and_tied_actions(self):
        maze = EXAMPLES["open"]
        distances = bfs_distances(maze)
        self.assertEqual(len(distances), 64)
        self.assertEqual(distances[maze.start], 14)
        self.assertEqual(optimal_actions(maze, maze.start),
                         (Action.RIGHT, Action.DOWN))
        expected = tuple([(x, 0) for x in range(8)]
                         + [(7, y) for y in range(1, 8)])
        self.assertEqual(shortest_path(maze), expected)

    def test_wall_requires_hand_computed_detour(self):
        maze = EXAMPLES["detour"]
        expected = tuple(
            [(0, y) for y in range(8)]
            + [(1, 7), (2, 7)]
            + [(2, y) for y in range(6, -1, -1)]
            + [(x, 0) for x in range(3, 8)]
        )
        self.assertEqual(bfs_distances(maze)[maze.start], 21)
        self.assertEqual(shortest_path(maze), expected)
        self.assertEqual(optimal_actions(maze, maze.start), (Action.DOWN,))

    def test_disconnected_goal_has_no_route_or_training_target(self):
        maze = EXAMPLES["blocked"]
        distances = bfs_distances(maze)
        self.assertNotIn(maze.start, distances)
        self.assertNotIn((1, 0), distances)  # A wall is also absent.
        self.assertEqual(distances[maze.goal], 0)
        self.assertIsNone(shortest_path(maze))
        self.assertEqual(optimal_actions(maze, maze.start), ())

    def test_start_at_goal_is_zero_move_route(self):
        maze = EXAMPLES["at-goal"]
        self.assertEqual(bfs_distances(maze)[maze.start], 0)
        self.assertEqual(shortest_path(maze), (maze.goal,))
        self.assertEqual(optimal_actions(maze, maze.goal), ())
        self.assertTrue(step(maze, maze.start, Action.RIGHT).done)

    def test_reference_rejects_wall_and_out_of_bounds_current_cells(self):
        maze = EXAMPLES["detour"]
        for position in ((1, 0), (-1, 0), (8, 7)):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    shortest_path(maze, position)
                with self.assertRaises(ValueError):
                    optimal_actions(maze, position)

    def test_all_128_three_by_three_wall_patterns_against_relaxation_oracle(self):
        """Compare BFS to synchronous distance relaxation, without its helpers.

        Start/goal are fixed free corners. Every subset of the other seven cells
        is an obstacle configuration; all cells outside this 3x3 patch are walls.
        """
        patch = {(x, y) for y in range(3) for x in range(3)}
        outside = {(x, y) for y in range(8) for x in range(8)} - patch
        start, goal = (0, 0), (2, 2)
        candidates = sorted(patch - {start, goal})
        action_deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))

        for mask in range(1 << len(candidates)):
            with self.subTest(obstacle_mask=mask):
                inner_walls = {
                    cell for bit, cell in enumerate(candidates) if mask & (1 << bit)
                }
                free = patch - inner_walls
                maze = Maze(outside | inner_walls, start, goal)

                # A finite simple route contains at most len(free)-1 edges.
                # Nine is therefore an unreachable sentinel in this 3x3 graph.
                oracle = {cell: (0 if cell == goal else 9) for cell in free}
                for _ in range(len(free) - 1):
                    old = oracle.copy()
                    oracle = {
                        cell: min(
                            old[cell],
                            min(
                                [old[other] + 1 for other in free
                                 if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1]
                                + [9]
                            ),
                        )
                        for cell in free
                    }
                expected = {cell: distance for cell, distance in oracle.items()
                            if distance < 9}
                self.assertEqual(bfs_distances(maze), expected)

                for position in free:
                    route = shortest_path(maze, position)
                    distance = expected.get(position)
                    if distance is None:
                        self.assertIsNone(route)
                        self.assertEqual(optimal_actions(maze, position), ())
                        continue

                    self.assertEqual(route[0], position)
                    self.assertEqual(route[-1], goal)
                    self.assertEqual(len(route) - 1, distance)
                    self.assertEqual(len(set(route)), len(route))
                    self.assertTrue(set(route) <= free)
                    for first, second in zip(route, route[1:]):
                        self.assertEqual(
                            abs(first[0] - second[0]) + abs(first[1] - second[1]), 1
                        )
                    expected_actions = tuple(
                        Action(code) for code, (dx, dy) in enumerate(action_deltas)
                        if distance > 0
                        and expected.get((position[0] + dx, position[1] + dy))
                        == distance - 1
                    )
                    actions = optimal_actions(maze, position)
                    self.assertEqual(actions, expected_actions)
                    for action in actions:
                        moved = step(maze, position, action)
                        self.assertFalse(moved.blocked)
                        self.assertEqual(expected[moved.position], distance - 1)


if __name__ == "__main__":
    unittest.main()
