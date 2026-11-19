# MazeBits: Learning to Navigate with Fewer Bits

## Purpose

MazeBits is an interactive FPGA testbed for studying how neural-network precision affects learned navigation. Its central question is:

> How does reducing a neural network from 8-bit to 4-bit weights and hidden activations affect navigation success, path quality, and FPGA resource usage?

A motivating application is a small mobile robot navigating around obstacles, such as a warehouse robot moving between shelves. The project uses a simplified grid world to study a learned controller under memory and compute constraints. Success will be demonstrated by measured navigation behavior and reproducible hardware results; the project does not assume that a neural policy outperforms a conventional shortest-path algorithm.

## Interactive demonstration

The target is the Nexys A7-100T, which provides switches, buttons, and VGA output suitable for the interface. See the [Digilent board manual](https://digilent.com/reference/_media/reference/programmable-logic/nexys-a7/nexys-a7_rm.pdf).

The minimum demonstration uses an 8x8 maze:

1. Directional buttons move an editing cursor.
2. Switches select wall editing, start placement, goal placement, or run mode.
3. The center button applies an edit or starts a comparison.
4. Two VGA panels show the 8-bit and 4-bit agents on identical map/start/goal cases.
5. Each panel displays the path, movement count, and run outcome.

The map is frozen during a comparison. Edits reset both agents. We will define deterministic termination for reaching the goal, repeated states, and a step limit. A loop or timeout does not prove that a maze has no solution.

The FPGA performs neural inference using previously exported weights. Training and dataset generation run on a laptop CPU. Animation speed will be controlled independently of measured inference latency.

## Learning approach

The initial method is BFS-guided imitation learning. We generate obstacle layouts and use breadth-first search to obtain shortest-path distances to selected goals. For reachable states, every legal action reducing distance by one is an acceptable target. Examples include states throughout the reachable region, rather than only positions along one demonstration path.

The policy input has three 8x8 binary planes: walls, current agent position, and goal. It outputs four action scores for up, right, down, and left. A collision checker enforces walls and boundaries without computing a route for the policy.

A compact CNN is the first candidate. We will evaluate complete rollouts on unseen layouts before freezing the architecture for RTL. Layouts, including their associated states and goals, stay within one training, validation, or test partition. Test layouts remain reserved for final evaluation.

The first software gate is useful floating-point navigation on validation layouts. If the candidate fails, we will revise the dataset or model before implementing that network in hardware. Reinforcement learning is an optional later experiment; it is not part of the minimum version.

## Precision comparison

The two configurations are:

- W8A8: 8-bit weights and hidden activations.
- W4A4: 4-bit weights and hidden activations.

Binary input cells retain exact values. Biases and accumulators use wider representations where range analysis requires them. Rounding, scaling, saturation, and signed arithmetic will be specified explicitly and checked against an integer Python reference. See [Jacob et al. on integer-only inference and quantization-aware training](https://arxiv.org/abs/1712.05877).

Both configurations begin from the same floating-point checkpoint and use identical evaluation scenarios. If quantization-aware fine-tuning is needed, its procedure and results will be reported separately.

A paired runtime demonstration compares behavior. Separate fixed-width synthesis builds under equivalent constraints compare hardware resources and timing. Reduced precision does not guarantee lower cycle count, fewer DSP blocks, or proportional BRAM savings.

| Measurement | Interpretation |
|---|---|
| Success rate on solvable held-out cases | Navigation reliability |
| Path length relative to BFS, for successful runs | Route efficiency |
| Blocked proposals, loops, and timeouts | Failure behavior |
| Cycles per decision, excluding animation | Inference latency |
| LUTs, registers, DSPs, BRAM, and timing | Hardware implementation cost |

## Shared milestones

All three members collaborate across dataset creation, training, hardware development, integration, and evaluation. Each milestone has a joint review and a reproducible check; Git history and project notes document actual contributions.

| Step | Deliverable | Acceptance gate |
|---|---|---|
| 1 | Maze environment and BFS reference | Movement, paths, labels, and encoding pass tests |
| 2 | Dataset and floating-point CNN | Full-rollout validation on unseen layouts |
| 3 | 8-bit and 4-bit integer references | Documented arithmetic and measured navigation changes |
| 4 | Verilog inference and testbenches | Exact agreement with integer reference vectors |
| 5 | Switch/button editor and VGA | Editable map displayed correctly on the board |
| 6 | Integrated comparison | Both precisions run on identical edited scenarios |
| 7 | Evaluation and release | Reproducible reports, demo, and repository instructions |

The minimum scope is one working 8x8 system. Larger maps, physical robots, online training, and elaborate menus are extensions. The main early uncertainty is generalization to unseen maps, so software evaluation comes before the final hardware architecture.
