# Improved W8A8 reference vectors

These eight cases contain inputs, all layer accumulators/activations, scores, and actions for the Step 7 model. Export verified 133,160 intermediate/output words with independent scalar loops; zero mismatches. They are Python arithmetic evidence, not an RTL pass record.

```powershell
python python/export_fixed_vectors.py --model-dir model/cnn-structured-w8a8 --training-metadata data/maze-v2/train.json --output-dir runs/reexport-structured-w8a8
python python/verify_fixed_vectors.py --model-dir model/cnn-structured-w8a8 --vectors-dir test_vectors/cnn-structured-w8a8
```

The export output directory must be new or empty. Reserved-test examples are not used.
