# Step 1: Maze Environment and BFS Reference

This step establishes the movement rules and trustworthy shortest-path labels needed for imitation learning. It uses only the Python standard library. The demo is a BFS reference demonstration; a trained policy and FPGA implementation come in later steps.

## Run the reference

From the repository root with Python 3.12 or newer:

~~~powershell
python python/demo.py --maze open
python python/demo.py --maze detour --trace
python python/demo.py --maze blocked
python python/demo.py --maze at-goal
python -m unittest discover -s python/tests -v
~~~

The --trace option shows the reference route as a sequence of movement states. The blocked case exercises an unreachable destination; at-goal exercises a terminal initial state.

## Maze and coordinates

Every map is exactly 8 columns by 8 rows. A position is (x, y): the top-left cell is (0, 0), x increases rightward, and y increases downward. Row-major storage indexes a cell as y * 8 + x.

Walls are stored as an immutable frozenset of coordinate pairs. Start, goal, and current positions must be in bounds and on free cells. Start and goal may coincide.

ASCII input accepts these symbols:

| Symbol | Meaning |
|---|---|
| # | Wall |
| . | Free cell |
| S | Start |
| G | Goal |
| B | Start and goal together |

The display may additionally use A for the current agent, X for an agent on its goal, and * for a reference path. Display output is intended for inspection and is not promised to be parseable as an input map.

## Movement contract

| Action | Integer ID | Change in position |
|---|---|---|
| UP | 0 | (0, -1) |
| RIGHT | 1 | (+1, 0) |
| DOWN | 2 | (0, +1) |
| LEFT | 3 | (-1, 0) |

Moves are orthogonal and span one cell. A valid direction into a wall or outside the grid leaves the agent in place and reports a blocked move. An invalid action value is an input error.

The goal is terminal: once reached, further actions cannot move the agent away from it. A terminal no-op is distinct from a blocked attempt. Movement checks enforce legality; they do not choose a route.

## BFS reference and training labels

The reference module computes distances outward from the goal:

- bfs_distances returns a dictionary mapping reachable free cells to minimum movement counts. Walls and unreachable cells are absent.
- optimal_actions returns every action leading to distance d - 1, in the fixed action order above. Goal and unreachable states return no actions.
- shortest_path returns a deterministic shortest route including both endpoints, or None when disconnected. If start equals goal, the path contains one position.

Several actions can be equally good. Keeping all optimal actions avoids teaching that one arbitrary tie-breaking choice is uniquely correct.

For an empty grid from (0, 0) to (7, 7), the minimum is 14 moves, with RIGHT and DOWN both optimal initially. A path's movement count is its number of positions minus one.

## Model input contract

The encoding module returns a tuple of 192 binary integers, ordered by channel, then row, then column:

1. Walls: 64 values, with 1 at each obstacle.
2. Agent: 64 values, with exactly one 1.
3. Goal: 64 values, with exactly one 1.

The flat index is channel * 64 + y * 8 + x. Values are exact 0 or 1; no image normalization is applied. At a terminal state, the agent and goal planes identify the same cell.

## Files and completion checks

The implementation is in python/mazebits/maze.py, reference.py, and encoding.py. The CLI is python/demo.py, with tests under python/tests/.

Before advancing, verify legal movement, blocked moves at walls and all boundaries, terminal behavior, malformed inputs, unreachable goals, shortest-path ties, and exact encoding order. Known shortest-path examples must agree with their expected distances.

The next step generates examples from many layouts and trains a candidate policy. Step 1's BFS remains the label generator and evaluation reference.

## Verified checkpoint

On 2026-11-19, all 21 tests passed locally with Python 3.12.14. The tests include all 128 obstacle patterns in an embedded 3x3 grid, checked against an independent distance-relaxation oracle. This validates the shortest-path teacher as well as the hand-written examples.

| Example | Expected shortest movement count |
|---|---|
| open | 14 |
| detour | 21 |
| blocked | No route |
| at-goal | 0 |

This evidence covers the environment, reference solver, and encoding. Neural-policy accuracy and hardware results will be measured in later milestones.
