"""Watch a memory-file-loaded W8A8 or W4A4 policy navigate a maze on the CPU."""

import argparse
from pathlib import Path

import torch

from mazebits.encoding import encode_state
from mazebits.examples import EXAMPLES
from mazebits.fixed_point import IntegerPolicy
from mazebits.maze import Action, Maze, render_ascii, step
from mazebits.quantized_io import load_quantized
from mazebits.reference import shortest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--model-dir", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--maze", choices=tuple(EXAMPLES), default="detour")
    selection.add_argument("--maze-file", type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    torch.set_num_threads(1)
    directory = args.model_dir or Path(f"model/cnn-w{args.bits}a{args.bits}")
    policy = load_quantized(directory)
    if policy.bits != args.bits:
        raise ValueError("--bits does not match the loaded memory bundle")
    model = IntegerPolicy(policy).eval()
    maze = (Maze.from_ascii(args.maze_file.read_text(encoding="utf-8"))
            if args.maze_file else EXAMPLES[args.maze])
    position, visited, path = maze.start, {maze.start}, [maze.start]
    actions = []
    outcome = "success" if position == maze.goal else "timeout"
    print(f"MazeBits | W{args.bits}A{args.bits} integer reference | CPU")
    print(render_ascii(maze, position=position))
    with torch.inference_mode():
        for index in range(args.max_steps):
            if position == maze.goal:
                break
            inputs = torch.tensor(encode_state(maze, position),
                                  dtype=torch.float32).view(1, 3, 8, 8)
            scores = model(inputs)
            action = Action(int(scores.argmax(dim=1)))
            moved = step(maze, position, action)
            position = moved.position
            actions.append(action)
            path.append(position)
            if args.trace:
                print(f"\nAction {index + 1}: {action.name} -> {position}; scores={scores[0].tolist()}")
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
    reference = shortest_path(maze)  # Reporting only, after every policy decision.
    print("\nInteger-policy path (*), A = agent, X = reached goal:")
    print(render_ascii(maze, position=position, path=path))
    print(f"CNN outcome: {outcome}; attempted actions: {len(actions)}")
    print("BFS comparison: " + ("no route exists." if reference is None
                               else f"shortest route is {len(reference) - 1} moves."))
    print("Raw integer argmax; no route assistance or illegal-action repair.")


if __name__ == "__main__":
    main()
