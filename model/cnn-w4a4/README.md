# MazeBits W4A4 integer CNN

This bundle was calibrated from the unchanged Step 2 FP32 checkpoint. No fine-tuning was performed. Weights are signed 4-bit words; hidden activations are unsigned 4-bit codes. Biases, accumulators, and final scores use signed 32 bits.

## Files

- weights.mem: 16,624 signed words, layer-major OIHW.
- biases.mem: 132 signed 32-bit words.
- quantization.vh: generated layer offsets, dimensions, widths, and scale exponents; include inside a Verilog module.
- manifest.json: the authoritative arithmetic contract, per-channel accumulator bounds, storage accounting, provenance, and hashes.

Memory files contain one hexadecimal word per line, with negative values encoded in two's complement. Hidden activations must be zero-extended for signed multiplication. Rounding uses nearest with ties away from zero. Requantization may shift left; the W8A8 first layer requires that case. Head scores share one scale and remain wide for argmax.

Every bias-first partial sum is bounded inside [-3,690, 3,391], comfortably within signed 32 bits. The proof uses all legal input codes, not just observed samples.

Logical packed parameter payload: 8,840 bytes. This excludes FPGA block-allocation overhead and is not a measured BRAM/DSP result.

Validation navigation success: 3,517 / 3,600 (97.6944%). Full results and the reserved-test policy are in [Step 3](../../docs/step-03.md).

~~~powershell
python python/predict_fixed.py --bits 4 --maze detour --trace
python python/verify_fixed_vectors.py --model-dir model/cnn-w4a4 --vectors-dir test_vectors/cnn-w4a4
~~~

The reference runs on the CPU. FPGA RTL is the next milestone.
