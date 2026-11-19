"""Exact binary input layout for the future learned policy."""

from .maze import Cell, Maze, SIZE

CHANNELS = 3
INPUT_VALUES = CHANNELS * SIZE * SIZE


def encode_state(maze: Maze, position: Cell) -> tuple[int, ...]:
    """192 binary values in CHW order: walls, current agent, goal.

    Offset = channel * 64 + y * 8 + x. Values remain exact 0/1; no /255 scaling.
    This is a logical tensor layout, not yet a UART or packed-memory format.
    """
    maze.require_open(position)
    cells = [(x, y) for y in range(SIZE) for x in range(SIZE)]
    return tuple(
        [int(cell in maze.walls) for cell in cells]
        + [int(cell == position) for cell in cells]
        + [int(cell == maze.goal) for cell in cells]
    )
