# Improved W4A4 CNN candidate

This bundle is the Step 7 structured-maze checkpoint exported with 4-bit weights and hidden activations. Biases and accumulators remain 32-bit. It shares the same dimensions and input/action contract as the original network. Calibration used 4,096 training states from maze-v2, seed 312.

The [improvement evidence](../../results/policy-improvement/README.md) reports validation performance on both v1 and v2 and records remaining failures. The [manifest](manifest.json) binds every tensor, scale, memory word, range proof, and source checkpoint. [Full intermediate traces](../../test_vectors/cnn-structured-w4a4/manifest.json) match the independent scalar arithmetic reference.

This is a software-verified candidate. Existing Step 6 FPGA bitstreams and resource/timing reports use the original model bundles; these new weights have not yet been rebuilt or tested on the board.

```powershell
python python/predict_fixed.py --bits 4 --model-dir model/cnn-structured-w4a4 --maze-file results/navigation/counterexamples/long-detour.txt --trace
python python/verify_fixed_vectors.py --model-dir model/cnn-structured-w4a4 --vectors-dir test_vectors/cnn-structured-w4a4
```
