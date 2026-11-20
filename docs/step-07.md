# Step 7: Improve the learned policy and measure the tradeoff

The board exposed a policy limitation: the original FP32 model loops on a solvable 26-move detour, so quantization alone cannot explain every failure. This step broadens the training distribution and measures the resulting model on shared validation routes before replacing FPGA weights.

The CNN remains eight 3x3 convolution/ReLU layers with 16 channels and a 1x1 four-action head (16,756 parameters). Inputs, action order, and raw-argmax termination rules remain identical. No BFS route search, action masking, or visited-cell repair is added to inference.

## Reproduce

From the repository root with the CPU Python environment:

```powershell
python python/prepare_structured_data.py --output-dir data/maze-v2
python python/train.py --data-dir data/maze-v2 --run-dir runs/cnn-structured-v2 --initial-model model/cnn-fp32 --seed 312 --epochs 45 --patience 12 --learning-rate 0.0003 --threads 4
python python/compare_policies.py --baseline model/cnn-fp32 --candidate runs/cnn-structured-v2/model --data-dir data/maze-v1 --data-dir data/maze-v2 --output-dir runs/comparison-structured-v2 --threads 4
```

Generated dataset and run directories must be new or empty. Reuse a generated dataset when repeating training into a different run directory. The v1 dataset can be regenerated with `python python/prepare_data.py --output-dir data/maze-v1` if absent. The original `model/cnn-fp32` bundle stays available as the baseline.

## Dataset and model selection

The [v2 dataset audit](../results/dataset-v2/README.md) describes 1,200 training layouts, 200 validation layouts, and 200 reserved-test layouts. The training mixture is 40% random obstacles, 20% connected barriers/U-shaped obstacles, and 40% winding corridors; maximum training BFS distance is 36. Prior layout IDs and known board demonstrations are excluded including rotations/reflections. Goals and distance-stratified starts deliberately exercise long routes.

Fine-tuning initializes from hash-verified original FP32 weights and uses a new Adam optimizer. Checkpoint zero is eligible, so a regressing run cannot silently displace the starting model on the selected validation objective. Each epoch is selected by v2 complete-route success count, with validation loss breaking ties. The original v1 validation set is evaluated afterward to reveal regressions. Training does not use either reserved-test split.

The photographed mazes are already-known development diagnostics. Even though their exact layouts and symmetries are excluded from training, success on them is not an unbiased estimate of generalization. The paired aggregate comparison uses separately generated validation layouts and reports gains as well as losses, with maze-family and BFS-distance groups.

## Runtime, memory, and quality

Use the [benchmarking guide](benchmarking.md) for measurement definitions and the original board's evidence. The original W8A8/W4A4 engines each take 3,217,153 cycles per decision; at 100 MHz this is 32.17153 ms. Lower precision saves storage and routed resources in that build, without a per-decision speedup. An early failed route is not a faster successful route.

A new checkpoint must be calibrated from training data, exported, reloaded and evaluated before FPGA use. Existing Step 6 hardware timing/resource measurements belong to its original weights and bitstream. New trained models are kept separately until their integer behavior and hardware implementation are verified.

## Measured results and new bundles

[The complete comparison](../results/policy-improvement/README.md) reports improvements on both distributions. Original random-layout route success rises to 100% FP32, 100% W8A8, and 98.81% W4A4. On structured validation, the corresponding values are 99.44%, 99.29%, and 84.38%. The original models scored 76.56%, 76.46%, and 65.75% on those same structured routes.

The selected model is published in `model/cnn-structured-fp32`; integer candidates are in `model/cnn-structured-w8a8` and `model/cnn-structured-w4a4`. Existing default bundles and board bitstreams still use the original checkpoint. The Python/scalar evidence belongs to this training step. [Step 8](step-08.md) adds new-weight RTL verification and a separate implementation build; physical testing is tracked there.

To reproduce the training-only calibration and v2 integer evaluation:

```powershell
python python/quantize.py --model-dir model/cnn-structured-fp32 --data-dir data/maze-v2 --run-dir runs/quantization-structured-v2 --calibration-samples 4096 --seed 312
```

To compare any frozen integer bundle against its FP32 source on another validation distribution without recalibrating:

```powershell
python python/evaluate_integer_bundle.py --model-dir model/cnn-structured-w8a8 --float-model-dir model/cnn-structured-fp32 --data-dir data/maze-v1 --output-dir runs/recheck-structured-w8-v1
python python/evaluate_integer_bundle.py --model-dir model/cnn-structured-w4a4 --float-model-dir model/cnn-structured-fp32 --data-dir data/maze-v1 --output-dir runs/recheck-structured-w4-v1
python python/evaluate_integer_bundle.py --model-dir model/cnn-w8a8 --float-model-dir model/cnn-fp32 --data-dir data/maze-v2 --output-dir runs/recheck-original-w8-v2
python python/evaluate_integer_bundle.py --model-dir model/cnn-w4a4 --float-model-dir model/cnn-fp32 --data-dir data/maze-v2 --output-dir runs/recheck-original-w4-v2
```

All output directories must be new or empty. The evaluator validates source-checkpoint identity, checks unchanged model/data/source hashes, records grouped route outcomes, and opens only the validation split. Check [remaining failures and paired losses](../results/policy-improvement/README.md) as well as aggregate success.
