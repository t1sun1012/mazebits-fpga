# MazeBits

Interactive FPGA maze navigation with a learned policy, VGA visualization, and an 8-bit versus 4-bit quantization comparison.

MazeBits is an EC311 project for the Nexys A7-100T. The planned board demonstration lets users edit an 8x8 maze with switches and buttons and compare two quantized neural policies on a VGA display. All three team members collaborate through shared implementation, reviews, and testing.

## Current status

**Step 5 implemented: switch/button maze editor and VGA display, ready for a board test.** The editor and display pass HDL simulation and routed 100 MHz timing checks. A bitstream is generated locally; physical programming and monitor testing remain to be done. Step 6 will connect both verified CNNs to the edited scenarios.

## Maze editor

![Actual HDL-rendered editor frame](results/editor/baseline_frame.png)

Arrows move the cursor; center applies an edit. SW1:SW0 selects 00=WALL, 01=START, 10=GOAL, or 11=VIEW. Reset restores an empty map. Both panels mirror the edited map; VIEW freezes edits. Neural navigation is the next integration step.

The VGA output is 640x480 at 59.5238 Hz, using a 25 MHz pixel enable in one 100 MHz clock domain. The routed editor/display top uses 557 LUTs, 432 registers, one RAMB18, and zero DSPs; setup and hold timing pass with zero DRC violations.

~~~powershell
python scripts/run_editor.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
python scripts/run_editor_vivado.py --stage bitstream --vivado C:/Xilinx/Vivado/2024.1/bin/vivado.bat
~~~

Read [Step 5 and board-testing instructions](docs/step-05.md), [VGA timing/layout](docs/vga-layout.md), and [editor verification evidence](results/editor/README.md). The image above is an example captured from the actual simulated RGB outputs.

## Learned policy baseline

Both quantized models retain the same Step 3 checkpoint and train-only calibration.

| Validation result | FP32 | W8A8 | W4A4 |
|---|---:|---:|---:|
| Goal-reaching runs | 3,598 / 3,600 | 3,598 / 3,600 | 3,517 / 3,600 |
| Navigation success | 99.9444% | 99.9444% | 97.6944% |
| Logical parameter bytes | 67,024 | 17,152 | 8,840 |

W4A4 reduces the packed parameter payload by 48.46% relative to W8A8, with a 2.25 percentage-point reduction in navigation success. Routed resource and timing measurements for the minimal board test are below. The test partition is reserved.

![Precision comparison](results/quantization/comparison.png)

## Verilog inference

| Routed board-test metric | W8A8 | W4A4 |
|---|---:|---:|
| LUTs | 496 | 366 |
| Registers | 273 | 258 |
| 36 Kb BRAM tile equivalents | 9 | 5 |
| DSP blocks | 0 | 0 |
| Setup slack (ns) | 1.497 | 1.757 |
| Hold slack (ns) | 0.122 | 0.073 |
| DRC violations | 0 | 0 |
| Cycles per decision | 3,217,153 | 3,217,153 |
| Latency at 100 MHz | 32.17153 ms | 32.17153 ms |

These are measurements for the button/LED wrapper with two fixed fixtures, not the final editable/VGA system. Both builds have zero DRC violations. The serial schedule gives both widths the same latency; this is not a CPU speed comparison.

~~~powershell
python scripts/run_rtl.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin --bits 8 4
python scripts/run_vivado.py --bits 8 4 --stage bitstream --vivado C:/Xilinx/Vivado/2024.1/bin/vivado.bat
~~~

The HDL runners need only standard-library Python. Icarus Verilog is also supported with --sim iverilog. Read [Step 4](docs/step-04.md), [board controls and constraints](docs/board-constraints.md), and [implementation evidence](results/rtl/README.md).

## Try the precision comparison

Use Python 3.12 and run from the repository root:

~~~powershell
python -m pip install -r requirements.txt
python python/predict_fixed.py --bits 8 --maze detour --trace
python python/predict_fixed.py --bits 4 --maze detour --trace
~~~

The committed memory bundles run immediately; dataset generation and retraining are unnecessary for this demo. On the hand-written detour, W8A8 reaches the goal in 21 moves, while W4A4 hits a wall on attempt four. These are CPU integer references; the board demonstration is a later milestone.

The floating-point baseline remains available:

~~~powershell
python python/predict.py --maze detour --trace
~~~

All policies select their raw argmax action. No BFS route or illegal-action repair assists inference.

## Model and data

The CNN has 16,756 parameters: eight 3x3 convolution/ReLU layers with 16 channels, followed by a 1x1 four-action head. Binary input planes represent walls, current agent position, and goal. The agent's cell selects the four action scores.

Training uses BFS-guided imitation learning. The dataset contains 800 training layouts with 117,171 labeled states, 150 validation layouts with 22,029 states, and 150 reserved test layouts. Rotated and reflected copies cannot cross partitions. All equally optimal shortest-path actions remain valid targets.

W8A8/W4A4 means signed 8/4-bit weights and unsigned 8/4-bit hidden activations. Biases, accumulators, and final scores remain signed 32-bit. Power-of-two scales, ties-away rounding, saturation, and every partial-sum bound are specified in [Step 3](docs/step-03.md).

## Verification

All 98 software tests passed. For each precision:

- 133,160 intermediate/output words matched independent scalar arithmetic across eight fixtures.
- Fresh-process replay matched all 88,116 validation score words and 3,600 trajectories.
- Exported memory files, headers, tensor layouts, and hashes were checked.

~~~powershell
python -m unittest discover -s python/tests -v
python python/verify_fixed_vectors.py --model-dir model/cnn-w8a8 --vectors-dir test_vectors/cnn-w8a8
python python/verify_fixed_vectors.py --model-dir model/cnn-w4a4 --vectors-dir test_vectors/cnn-w4a4
~~~

GitHub Actions runs the unit tests, both full Python vector replays, both full RTL-vector suites in Icarus Verilog, the board-control tests, and the editor/VGA suites. A full validation replay additionally needs the generated dataset:

~~~powershell
python python/prepare_data.py --output-dir data/maze-v1
python python/verify_fixed_point.py --output-dir runs/recheck-fixed
~~~

Reuse an existing dataset. Generated output directories must be new or empty.

## Documentation

- [Project proposal](docs/proposal.md): application, demonstration, and shared milestones.
- [Step 1](docs/step-01.md): maze movement, BFS, and input encoding.
- [Step 2](docs/step-02.md): dataset, CNN training, and floating-point results.
- [Step 3](docs/step-03.md): integer arithmetic, calibration, exports, and comparison.
- [Step 4](docs/step-04.md): Verilog datapath, interfaces, cycle counts, HDL verification, and routed builds.
- [Step 5](docs/step-05.md): board controls, editable map, VGA display, and physical test instructions.
- [Dataset audit](results/dataset/README.md) and [training evidence](results/cnn-baseline/README.md).
- [Quantization evidence](results/quantization/README.md).
- [W8A8 bundle](model/cnn-w8a8/README.md) and [W4A4 bundle](model/cnn-w4a4/README.md).
- [W8A8 RTL vectors](test_vectors/cnn-w8a8/README.md) and [W4A4 RTL vectors](test_vectors/cnn-w4a4/README.md).

## Next checkpoint

Step 6 integrates both navigation policies on identical edited scenarios and displays their paths, movement counts, and outcomes. Each completed milestone is committed and pushed on its own branch.
