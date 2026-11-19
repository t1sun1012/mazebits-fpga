"""Watch the trained CNN navigate an example or a custom 8x8 ASCII maze."""

import argparse
from pathlib import Path

import torch

from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.maze import Action, Maze, render_ascii, step
from mazebits.model_io import load_model
from mazebits.reference import shortest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("model/cnn-fp32"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--maze", choices=tuple(EXAMPLES), default="detour")
    selection.add_argument("--maze-file", type=Path, help="eight ASCII rows using # . S G B")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    torch.set_num_threads(1)
    model, _ = load_model(args.model_dir)
    maze = (Maze.from_ascii(args.maze_file.read_text(encoding="utf-8"))
            if args.maze_file else EXAMPLES[args.maze])
    position, visited, path = maze.start, {maze.start}, [maze.start]
    actions = []
    outcome = "success" if position == maze.goal else "timeout"
    print("MazeBits | Trained floating-point CNN | CPU inference")
    print(render_ascii(maze, position=position))
    with torch.inference_mode():
        for index in range(args.max_steps):
            if position == maze.goal:
                break
            inputs = torch.tensor(encode_state(maze, position),
                                  dtype=torch.float32).view(1, 3, 8, 8)
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("model produced non-finite scores")
            action = Action(int(logits.argmax(dim=1)))
            moved = step(maze, position, action)
            position = moved.position
            actions.append(action)
            path.append(position)
            if args.trace:
                print(f"\nAction {index + 1}: {action.name} -> {position}")
                print(render_ascii(maze, position=position, path=path))
            if moved.done:
                outcome = "success"
                break
            if moved.blocked:
                outcome = "blocked"
                break
            if position in visited:
                outcome = "loop"
                break
            visited.add(position)
    # Compute a teacher route only after inference, solely for comparison.
    reference = shortest_path(maze)
    print("\nCNN path (*), A = current agent, X = reached goal:")
    print(render_ascii(maze, position=position, path=path))
    print(f"CNN outcome: {outcome}; attempted actions: {len(actions)}")
    if reference is None:
        print("BFS comparison: this start and goal are disconnected.")
    else:
        print(f"BFS comparison: shortest route is {len(reference) - 1} moves.")
    print("The CNN used raw argmax actions without BFS assistance or action repair.")


if __name__ == "__main__":
    main()
