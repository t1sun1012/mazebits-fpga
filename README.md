# MazeBits

Interactive FPGA maze navigation with a learned policy, VGA visualization, and an 8-bit versus 4-bit quantization comparison.

MazeBits is an EC311 project for the Nexys A7-100T. Users will edit an 8x8 maze with switches and buttons, then compare two quantized neural policies navigating the same scenario on a VGA display. Training will run on a laptop CPU; inference will run on the FPGA.

## Current milestone

**Step 1 complete: maze environment and BFS reference.** The Python foundation defines movement, terminal states, shortest-path reference labels, and the binary input format for a future model. The command-line demonstration runs BFS, not a trained policy.

Training, quantization, Verilog inference, and board integration are later milestones. All three team members collaborate through shared implementation, reviews, and testing.

## Quick start

Use Python 3.12 or newer. Step 1 uses only the Python standard library; no package installation is required. Run these commands from the repository root:

~~~powershell
python python/demo.py --maze detour --trace
python python/demo.py --maze blocked
python -m unittest discover -s python/tests -v
~~~

The demo also accepts --maze open and --maze at-goal. See the walkthrough for coordinates, action IDs, ASCII symbols, and expected behavior.

## Documentation

- [Project proposal](docs/proposal.md): purpose, demonstration, evaluation, and shared milestones.
- [Step 1 walkthrough](docs/step-01.md): maze rules, BFS labels, encoding, and verification.

## Verification

All 21 local tests pass on Python 3.12.14, including an independent shortest-path check across all 128 obstacle patterns in a 3x3 region. The known examples require 14 moves (open), 21 moves (detour), no route (blocked), and zero moves (at-goal). GitHub Actions runs the tests and a reference demo on pushes and pull requests.

## Next checkpoint

Generate training examples from many layouts, split complete layouts between training/validation/test, and evaluate a small CNN through full navigation rollouts. The model architecture will be finalized only after the floating-point policy demonstrates useful navigation on unseen maps.

Work progresses through one reviewed milestone per branch, beginning with step-01-maze-environment. The repository is [t1sun1012/mazebits-fpga](https://github.com/t1sun1012/mazebits-fpga).
