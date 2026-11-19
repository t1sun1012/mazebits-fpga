"""BFS teacher and evaluation reference, separate from learned-policy execution."""

from collections import deque
from collections.abc import Mapping

from .maze import Action, Cell, Maze, neighbor


def bfs_distances(maze: Maze) -> dict[Cell, int]:
    """Shortest distance to the goal for every reachable free cell.

    Missing cells are walls or unreachable. The goal always has distance zero.
    """
    distances = {maze.goal: 0}
    queue = deque([maze.goal])
    while queue:
        cell = queue.popleft()
        for action in Action:
            candidate = neighbor(cell, action)
            if maze.is_open(candidate) and candidate not in distances:
                distances[candidate] = distances[cell] + 1
                queue.append(candidate)
    return distances


def _optimal_actions(
    position: Cell, distances: Mapping[Cell, int]
) -> tuple[Action, ...]:
    distance = distances.get(position)
    if distance is None or distance == 0:
        return ()
    return tuple(
        action for action in Action
        if distances.get(neighbor(position, action)) == distance - 1
    )


def optimal_actions(maze: Maze, position: Cell) -> tuple[Action, ...]:
    """Return all equally optimal teacher labels; none for terminal/unreachable."""
    maze.require_open(position)
    return _optimal_actions(position, bfs_distances(maze))


def shortest_path(maze: Maze, position: Cell | None = None) -> tuple[Cell, ...] | None:
    """Return a BFS reference route including both endpoints, or None.

    Ties use UP, RIGHT, DOWN, LEFT order. This is a teacher/reference, not ML.
    """
    current = maze.start if position is None else position
    maze.require_open(current)
    distances = bfs_distances(maze)
    if current not in distances:
        return None
    route = [current]
    while current != maze.goal:
        current = neighbor(current, _optimal_actions(current, distances)[0])
        route.append(current)
    return tuple(route)
