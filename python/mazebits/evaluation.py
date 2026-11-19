"""Closed-loop policy evaluation with raw actions and no route assistance."""

from collections import Counter, defaultdict
from statistics import mean

import torch

from .maze import Maze, SIZE, step


def _summary(rows: list[dict]) -> dict:
    outcomes = Counter(row["outcome"] for row in rows)
    successful = [row for row in rows if row["outcome"] == "success"]
    ratios = [
        row["steps"] / row["optimal_steps"]
        for row in successful if row["optimal_steps"] > 0
    ]
    attempts = sum(row["steps"] for row in rows)
    blocked = outcomes["blocked"]  # A blocked proposal terminates the episode.
    return {
        "episodes": len(rows),
        "success_count": len(successful),
        "success_rate": len(successful) / len(rows) if rows else None,
        "outcomes": {
            outcome: outcomes[outcome] for outcome in ("success", "blocked", "loop", "timeout")
        },
        "attempted_actions": attempts,
        "blocked_proposals": blocked,
        "blocked_proposal_rate": blocked / attempts if attempts else None,
        "successful_mean_steps": mean(row["steps"] for row in successful) if successful else None,
        "successful_mean_path_ratio": mean(ratios) if ratios else None,
        "successful_path_ratio_count": len(ratios),
        "successful_zero_step_count": sum(row["optimal_steps"] == 0 for row in successful),
        "successful_optimal_rate": (
            sum(row["steps"] == row["optimal_steps"] for row in successful) / len(successful)
            if successful else None
        ),
    }


def evaluate_rollouts(
    model: torch.nn.Module, metadata: dict, *, max_steps: int = 64, batch_size: int = 512
) -> dict:
    """Evaluate fixed, solvable episodes using only raw model argmax actions.

    The caller supplies shortest distances solely for reporting path efficiency.
    They are never model inputs or action selectors. Outcomes distinguish reaching
    the goal, a blocked proposal, a repeated position, and the step budget.
    This assumes a deterministic, stateless policy in eval mode.

    Steps counts attempted actions; path includes the initial position
    and the position after every attempt, including a final blocked attempt.
    Zero-distance successes have no defined path ratio and are excluded from
    that mean. Empty denominators produce None rather than misleading zeros.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    wall_sets = []
    wall_planes = []
    for layout in metadata["layouts"]:
        bits = layout["walls"]
        if not isinstance(bits, str) or len(bits) != SIZE * SIZE or set(bits) - {"0", "1"}:
            raise ValueError("each layout needs 64 row-major binary wall characters")
        wall_sets.append(frozenset((i % SIZE, i // SIZE) for i, bit in enumerate(bits) if bit == "1"))
        wall_planes.append(torch.tensor([int(bit) for bit in bits], dtype=torch.float32).view(SIZE, SIZE))

    definitions = metadata["episodes"]
    base_inputs = torch.zeros((len(definitions), 3, SIZE, SIZE), dtype=torch.float32)
    mazes, rows, positions, visited, active = [], [], [], [], []
    for index, episode in enumerate(definitions):
        layout_index = episode["layout_index"]
        if type(layout_index) is not int or not 0 <= layout_index < len(wall_sets):
            raise ValueError("episode layout_index is invalid")
        start, goal = tuple(episode["start"]), tuple(episode["goal"])
        maze = Maze(wall_sets[layout_index], start, goal)
        optimal_steps = episode["optimal_steps"]
        if (
            type(optimal_steps) is not int or optimal_steps < 0
            or (optimal_steps == 0) != (start == goal)
        ):
            raise ValueError("optimal_steps must be nonnegative and zero exactly at the goal")
        goal_index = episode["goal_index"]
        goals = metadata["layouts"][layout_index].get("goals")
        if type(goal_index) is not int or goal_index < 0:
            raise ValueError("episode goal_index is invalid")
        if goals is not None and (
            goal_index >= len(goals) or tuple(goals[goal_index]) != goal
        ):
            raise ValueError("episode goal disagrees with its layout goal_index")
        base_inputs[index, 0] = wall_planes[layout_index]
        base_inputs[index, 2, goal[1], goal[0]] = 1
        positions.append(start)
        mazes.append(maze)
        visited.append({start})
        rows.append({
            "layout_index": layout_index,
            "goal_index": goal_index,
            "start": list(start),
            "goal": list(goal),
            "optimal_steps": optimal_steps,
            "requested_density": metadata["layouts"][layout_index].get("requested_density"),
            "outcome": "success" if start == goal else None,
            "steps": 0,
            "actions": [],
            "path": [list(start)],
        })
        if start != goal:
            active.append(index)

    # Preserve mixed train/eval states too, rather than resetting every child
    # to the parent's initial state after evaluation.
    training_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            while active:
                remaining = []
                for offset in range(0, len(active), batch_size):
                    indices = active[offset:offset + batch_size]
                    inputs = base_inputs[indices]
                    for local_index, episode_index in enumerate(indices):
                        x, y = positions[episode_index]
                        inputs[local_index, 1, y, x] = 1
                    logits = model(inputs)
                    if not isinstance(logits, torch.Tensor) or logits.shape != (len(indices), 4):
                        raise ValueError("policy must return a tensor of shape [batch, 4]")
                    if not torch.isfinite(logits).all().item():
                        raise ValueError("policy produced non-finite action scores")
                    actions = logits.argmax(dim=1).tolist()  # Lowest action ID breaks ties.
                    for episode_index, action in zip(indices, actions):
                        row = rows[episode_index]
                        moved = step(mazes[episode_index], positions[episode_index], action)
                        positions[episode_index] = moved.position
                        row["steps"] += 1
                        row["actions"].append(action)
                        row["path"].append(list(moved.position))
                        if moved.done:
                            row["outcome"] = "success"
                        elif moved.blocked:
                            row["outcome"] = "blocked"
                        elif moved.position in visited[episode_index]:
                            row["outcome"] = "loop"
                        elif row["steps"] >= max_steps:
                            row["outcome"] = "timeout"
                        else:
                            visited[episode_index].add(moved.position)
                            remaining.append(episode_index)
                active = remaining
    finally:
        for module, training in training_states:
            module.training = training

    summary = _summary(rows)
    groups = defaultdict(list)
    for row in rows:
        density = row["requested_density"]
        groups["unknown" if density is None else str(density)].append(row)
    summary["by_requested_density"] = {
        density: _summary(group) for density, group in sorted(groups.items())
    }
    return {"summary": summary, "episodes": rows}
