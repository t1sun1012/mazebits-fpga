# MazeBits

Interactive FPGA maze navigation with a learned policy, VGA visualization, and an 8-bit versus 4-bit quantization comparison.

MazeBits is an EC311 project for the Nexys A7-100T. The planned board demonstration lets users edit an 8x8 maze with switches and buttons and compare two quantized neural policies on a VGA display. All three team members collaborate through shared implementation, reviews, and testing.

## Current status

**Step 3 complete: verified W8A8 and W4A4 integer CNNs.** Both variants were calibrated from the same floating-point checkpoint, using training states only and no fine-tuning. FPGA memory files and exact intermediate reference vectors are committed.

| Validation result | FP32 | W8A8 | W4A4 |
|---|---:|---:|---:|
| Goal-reaching runs | 3,598 / 3,600 | 3,598 / 3,600 | 3,517 / 3,600 |
| Navigation success | 99.9444% | 99.9444% | 97.6944% |
| Logical parameter bytes | 67,024 | 17,152 | 8,840 |

W4A4 reduces the packed parameter payload by 48.46% relative to W8A8, with a 2.25 percentage-point reduction in navigation success. Actual FPGA resources and timing remain to be measured. The test partition is reserved.

![Precision comparison](results/quantization/comparison.png)

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

All 88 software tests passed. For each precision:

- 133,160 intermediate/output words matched independent scalar arithmetic across eight fixtures.
- Fresh-process replay matched all 88,116 validation score words and 3,600 trajectories.
- Exported memory files, headers, tensor layouts, and hashes were checked.

~~~powershell
python -m unittest discover -s python/tests -v
python python/verify_fixed_vectors.py --model-dir model/cnn-w8a8 --vectors-dir test_vectors/cnn-w8a8
python python/verify_fixed_vectors.py --model-dir model/cnn-w4a4 --vectors-dir test_vectors/cnn-w4a4
~~~

GitHub Actions runs the unit tests and both full vector replays. A full validation replay additionally needs the generated dataset:

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
- [Dataset audit](results/dataset/README.md) and [training evidence](results/cnn-baseline/README.md).
- [Quantization evidence](results/quantization/README.md).
- [W8A8 bundle](model/cnn-w8a8/README.md) and [W4A4 bundle](model/cnn-w4a4/README.md).
- [W8A8 RTL vectors](test_vectors/cnn-w8a8/README.md) and [W4A4 RTL vectors](test_vectors/cnn-w4a4/README.md).

## Next checkpoint

Step 4 will implement the Verilog inference datapath and controller, verify intermediate values against these vectors, and measure synthesis resources and timing. Each completed milestone is committed and pushed on its own branch.
