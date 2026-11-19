# MazeBits

Interactive FPGA maze navigation with a learned policy, VGA visualization, and an 8-bit versus 4-bit quantization comparison.

MazeBits is an EC311 project for the Nexys A7-100T. The planned board demonstration lets users edit an 8x8 maze with switches and buttons and compare two quantized neural policies on a VGA display. All three team members collaborate through shared implementation, reviews, and testing.

## Current status

**Step 2 complete: generated maze dataset and trained floating-point CNN.** Training ran on the laptop CPU. The selected model reached the goal in **3,598 of 3,600 validation runs (99.9444%)** on 150 layouts excluded from training. Two loops remain. The final-test partition is reserved.

The model has **16,756 parameters**: eight 3x3 convolution/ReLU layers, 16 channels, and a 1x1 four-action head. It reads binary walls, agent position, and goal, then selects the action scores at the agent's cell. This is BFS-guided imitation learning.

~~~text
Generated mazes -> BFS optimal-action labels -> CPU CNN training
                 -> exported FP32 model -> verified validation replay
~~~

Fixed-point quantization, Verilog inference, and board integration are the next milestones.

## Try the trained policy

Use Python 3.12 and run from the repository root:

~~~powershell
python -m pip install -r requirements.txt
python python/predict.py --maze detour --trace
~~~

The committed model runs immediately; no dataset generation or retraining is needed for this demo. The CNN solves the hand-written detour in 21 moves. It chooses raw actions without a BFS route or an illegal-action mask.

A retained failure case can be inspected with:

~~~powershell
python python/predict.py --maze-file examples/validation-loop.txt --trace
~~~

## Dataset and validation

| Partition | Layouts | Labeled decision states | Use |
|---|---:|---:|---|
| Training | 800 | 117,171 | Learn the policy |
| Validation | 150 | 22,029 | Select and evaluate checkpoints |
| Test | 150 | 22,308 | Reserved for final evaluation |

Layouts are separated even under rotation and reflection. Every reachable nonterminal state for each selected goal receives all equally optimal BFS action labels. The first dataset uses random obstacle probabilities of 10%, 20%, and 30%; performance on arbitrary edited mazes is not guaranteed.

Training took about 11.2 minutes on one CPU thread. Epoch 41 was selected from 53 completed epochs using validation navigation success, with validation loss as the tie-break.

![Training history](results/cnn-baseline/training.png)

## Verify or retrain

~~~powershell
python -m unittest discover -s python/tests -v
python python/prepare_data.py --output-dir data/maze-v1
python python/evaluate.py --model-dir model/cnn-fp32 --data-dir data/maze-v1 --output-dir runs/recheck-cnn --reference-rollouts results/cnn-baseline/validation_rollouts.json
python python/train.py --data-dir data/maze-v1 --run-dir runs/my-cnn-run
~~~

Reuse the dataset when it already exists. Generated output directories must be new or empty. The 52 software tests passed locally. Fresh-process model loading reproduced all validation state metrics and every trajectory exactly in the recorded CPU environment. GitHub Actions runs the tests and demonstration checks.

## Documentation

- [Project proposal](docs/proposal.md): application, demonstration, and shared milestones.
- [Step 1](docs/step-01.md): movement, BFS reference, and binary input encoding.
- [Step 2](docs/step-02.md): dataset, architecture, training, evaluation, and reproduction.
- [Dataset manifest](results/dataset/README.md): split audit and artifact hashes.
- [Trained model](model/cnn-fp32/README.md): portable FP32 weights and provenance.
- [Training results](results/cnn-baseline/README.md): measurements, failure traces, and verification.

## Next checkpoint

Step 3 will specify integer scaling, rounding, saturation, and accumulator widths, then measure W8A8 and W4A4 navigation against this floating-point baseline. Each completed milestone is committed and pushed on its own branch.
