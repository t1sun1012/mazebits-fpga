# Improved W8A8 CNN

This bundle is the Step 7 structured-maze checkpoint exported with 8-bit weights and hidden activations. Biases and accumulators remain 32-bit. It shares the same dimensions and input/action contract as the original network. Calibration used 4,096 training states from maze-v2, seed 312.

The [improvement evidence](../../results/policy-improvement/README.md) reports validation performance on both v1 and v2 and records remaining failures. The [manifest](manifest.json) binds every tensor, scale, memory word, range proof, and source checkpoint. [Full intermediate traces](../../test_vectors/cnn-structured-w8a8/manifest.json) match the independent scalar arithmetic reference.

[Step 8](../../docs/step-08.md) verifies this bundle against the Verilog core and complete dual-policy navigation scenarios, and builds a separate improved-weight bitstream. See its [hardware evidence](../../results/navigation-structured/README.md) for implementation measurements and physical-test status. Existing Step 6 bitstreams and reports retain the original model bundles.

```powershell
python python/predict_fixed.py --bits 8 --model-dir model/cnn-structured-w8a8 --maze-file results/navigation/counterexamples/long-detour.txt --trace
python python/verify_fixed_vectors.py --model-dir model/cnn-structured-w8a8 --vectors-dir test_vectors/cnn-structured-w8a8
```
