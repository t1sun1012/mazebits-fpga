# Improved floating-point CNN

Step 7 fine-tunes the original CNN on 1,200 additional structured-maze layouts with barriers and winding corridors. The unchanged eight-layer, 16-channel CNN has 16,756 parameters. The selected checkpoint is epoch 35 of 45, using v2 validation route success and validation loss for selection.

It succeeds on 3,600/3,600 original validation routes and 4,773/4,800 structured validation routes. The original model scored 3,598/3,600 and 3,675/4,800 on these same sets. There are still 27 structured-validation loops. Both reserved-test partitions remain unevaluated.

The [manifest](manifest.json) and NPZ weights are hash-verified by the loader. [Step 7](../../docs/step-07.md) and [complete evidence](../../results/policy-improvement/README.md) describe training, paired gains/losses, and separate diagnostic cases.

```powershell
python python/predict.py --model-dir model/cnn-structured-fp32 --maze-file results/navigation/counterexamples/long-detour.txt --trace
```

The board continues using the original Step 6 weights until a separate hardware verification/build is performed for this checkpoint.
