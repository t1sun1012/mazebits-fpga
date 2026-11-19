# MazeBits floating-point CNN

This is the selected CPU-trained policy from Step 2. The bundle stores float32 weights in NPZ format and a JSON manifest; loading requires no pickle checkpoint.

Architecture: eight 3x3 convolution/ReLU layers with 16 channels, followed by a 1x1 four-action head. The input is 3x8x8 binary walls/agent/goal. The agent position selects four output scores in UP, RIGHT, DOWN, LEFT order.

There are 16,756 parameters. First-layer padding represents outside-wall cells as obstacles; all hidden padding is zero. See [the walkthrough](../../docs/step-02.md) for the exact contract.

~~~powershell
python python/predict.py --model-dir model/cnn-fp32 --maze detour --trace
python python/predict.py --model-dir model/cnn-fp32 --maze-file examples/validation-loop.txt --trace
~~~

The loader verifies the weight-file SHA-256, names, shapes, float32 dtypes, and finite values. The manifest includes the selected epoch, training settings, source hashes, environment, dataset identity, and validation metrics.

The selected checkpoint reaches the goal in 3,598 of 3,600 validation episodes. Two loops remain. This is a floating-point baseline; 8-bit and 4-bit inference will be measured in Step 3.
