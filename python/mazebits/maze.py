"""The fixed 8x8 environment contract shared by training and future RTL."""

from dataclasses import dataclass
from enum import IntEnum
from textwrap import dedent
from typing import Iterable

SIZE = 8
Cell = tuple[int, int]  # (x, y); row-major storage uses index y * SIZE + x.


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _check_cell(cell: Cell, name: str) -> None:
    if (
        not isinstance(cell, tuple)
        or len(cell) != 2
        or any(type(value) is not int for value in cell)
        or not all(0 <= value < SIZE for value in cell)
    ):
        raise ValueError(f"{name} must be an (x, y) integer tuple inside the 8x8 grid")


def _action(value: Action | int) -> Action:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("action must be UP=0, RIGHT=1, DOWN=2, or LEFT=3")
    try:
        return Action(value)
    except ValueError as error:
        raise ValueError("action must be UP=0, RIGHT=1, DOWN=2, or LEFT=3") from error


def neighbor(cell: Cell, action: Action | int) -> Cell:
    """Return a candidate neighbor; it may lie outside the board."""
    _check_cell(cell, "cell")
    dx, dy = _DELTAS[_action(action)]
    return cell[0] + dx, cell[1] + dy


@dataclass(frozen=True)
class Maze:
    walls: frozenset[Cell]
    start: Cell
    goal: Cell

    def __post_init__(self) -> None:
        # Copy mutable caller-owned sets, so subsequent edits cannot change a run.
        object.__setattr__(self, "walls", frozenset(self.walls))
        for wall in self.walls:
            _check_cell(wall, "wall")
        self.require_open(self.start, "start")
        self.require_open(self.goal, "goal")

    def is_open(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < SIZE and 0 <= y < SIZE and cell not in self.walls

    def require_open(self, cell: Cell, name: str = "position") -> None:
        _check_cell(cell, name)
        if cell in self.walls:
            raise ValueError(f"{name} cannot be inside a wall")

    @classmethod
    def from_ascii(cls, text: str) -> "Maze":
        """Read exactly 8x8 cells: # wall, . free, S start, G goal, B both."""
        rows = dedent(text).strip("\r\n").splitlines()
        if len(rows) != SIZE or any(len(row) != SIZE for row in rows):
            raise ValueError("maze must contain exactly eight rows of eight characters")
        walls: set[Cell] = set()
        starts: list[Cell] = []
        goals: list[Cell] = []
        for y, row in enumerate(rows):
            for x, symbol in enumerate(row):
                if symbol not in "#.SGB":
                    raise ValueError(f"unsupported maze character {symbol!r} at {(x, y)}")
                if symbol == "#":
                    walls.add((x, y))
                if symbol in "SB":
                    starts.append((x, y))
                if symbol in "GB":
                    goals.append((x, y))
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("maze needs exactly one start and one goal, or a single B")
        return cls(frozenset(walls), starts[0], goals[0])

    def to_ascii(self) -> str:
        """Return the parseable maze definition without agent/path overlays."""
        rows = [["#" if (x, y) in self.walls else "." for x in range(SIZE)]
                for y in range(SIZE)]
        sx, sy = self.start
        gx, gy = self.goal
        rows[sy][sx] = "S"
        rows[gy][gx] = "B" if self.start == self.goal else "G"
        return "\n".join("".join(row) for row in rows)


@dataclass(frozen=True)
class MoveResult:
    position: Cell
    blocked: bool
    done: bool


def step(maze: Maze, position: Cell, action: Action | int) -> MoveResult:
    """Apply one action. This checks collisions only; it never chooses a route."""
    maze.require_open(position)
    chosen = _action(action)
    if position == maze.goal:
        return MoveResult(position, blocked=False, done=True)
    candidate = neighbor(position, chosen)
    if not maze.is_open(candidate):
        return MoveResult(position, blocked=True, done=False)
    return MoveResult(candidate, blocked=False, done=candidate == maze.goal)


def legal_actions(maze: Maze, position: Cell) -> tuple[Action, ...]:
    """Legal moves in the frozen action order; a terminal state has none."""
    maze.require_open(position)
    if position == maze.goal:
        return ()
    return tuple(action for action in Action if maze.is_open(neighbor(position, action)))


def render_ascii(
    maze: Maze, *, position: Cell | None = None, path: Iterable[Cell] = ()
) -> str:
    """Display overlays: * path, A agent, X agent on goal. Not an input format."""
    rows = [list(row) for row in maze.to_ascii().splitlines()]
    for cell in path:
        maze.require_open(cell, "path cell")
        x, y = cell
        if rows[y][x] == ".":
            rows[y][x] = "*"
    if position is not None:
        maze.require_open(position)
        x, y = position
        rows[y][x] = "X" if position == maze.goal else "A"
    return "\n".join("".join(row) for row in rows)
