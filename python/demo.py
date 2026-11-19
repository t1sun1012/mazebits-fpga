"""Run a hand-checkable BFS reference demo. No model is trained in Step 1."""

import argparse

from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.maze import Action, neighbor, render_ascii, step
from mazebits.reference import optimal_actions, shortest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze", choices=tuple(EXAMPLES), default="detour")
    parser.add_argument("--trace", action="store_true", help="print each reference move")
    args = parser.parse_args()
    maze = EXAMPLES[args.maze]
    route = shortest_path(maze)
    encoded = encode_state(maze, maze.start)
    print(f"MazeBits | Step 1 | BFS reference | {args.maze}")
    print("No trained neural policy or FPGA inference is running.")
    print(render_ascii(maze))
    print(f"Input: {len(encoded)} binary values (walls / agent / goal)")
    print("Start actions:", ", ".join(a.name for a in optimal_actions(maze, maze.start)) or "none")
    if route is None:
        print("BFS result: no route exists from this start to this goal.")
        return
    print(f"BFS shortest distance: {len(route) - 1} moves")
    position = maze.start
    for index, destination in enumerate(route[1:], start=1):
        action = next(a for a in Action if neighbor(position, a) == destination)
        result = step(maze, position, action)
        if result.blocked or result.position != destination:
            raise RuntimeError("movement rules disagree with the BFS reference")
        position = result.position
        if args.trace:
            print(f"\nMove {index}: {action.name} -> {position}; goal={result.done}")
            print(render_ascii(maze, position=position, path=route[:index + 1]))
    print("Final reference path (*); X = agent at goal:")
    print(render_ascii(maze, position=position, path=route))
    print("Result: goal reached")


if __name__ == "__main__":
    main()
