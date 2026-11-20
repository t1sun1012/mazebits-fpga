# Measuring quantization: runtime, storage, and navigation quality

The current Step 6 build demonstrates a resource/accuracy tradeoff: W4A4 uses less FPGA memory and logic, while W8A8 succeeds on more of the fixed validation routes. Both engines take the same number of cycles per decision. W8A8 means 8-bit weights and hidden activations; W4A4 means 4-bit weights and hidden activations. Biases and accumulators remain 32-bit.

The numbers below belong to the original trained checkpoint and the routed Step 6 bitstream. A retrained checkpoint needs its own quantization/evaluation records. Its hardware counts must be measured again after implementation, even when its layer dimensions stay the same.

## Recorded comparison

| Metric | W8A8 | W4A4 | Evidence |
|---|---:|---:|---|
| CNN cycles per decision | 3,217,153 | 3,217,153 | [Real dual-CNN simulation](../results/navigation/verification.json) |
| Inference latency at 100 MHz | 32.17153 ms | 32.17153 ms | Cycles / routed clock frequency |
| Packed weights | 16,624 bytes | 8,312 bytes | Model manifests |
| Packed 32-bit biases | 528 bytes | 528 bytes | Model manifests |
| Total packed parameters | 17,152 bytes | 8,840 bytes | Model manifests |
| Logical feature-bank storage | 2,048 bytes | 1,024 bytes | Two 1,024-word banks in the CNN RTL |
| Routed CNN BRAM, 36 Kb equivalents | 9 | 5 | [Combined routed hierarchy](../results/navigation/utilization_hierarchy.rpt) |
| Routed CNN LUTs | 656 | 516 | Same combined build |
| Routed CNN registers | 347 | 332 | Same combined build |
| Routed CNN DSP blocks | 0 | 0 | Same combined build |
| Optimal-action accuracy | 99.9092% | 99.4008% | Same 22,029 validation states |
| Successful routes | 3,598 / 3,600 | 3,517 / 3,600 | Same fixed validation episodes |
| Route success rate | 99.9444% | 97.6944% | [Quantization comparison](../results/quantization/comparison.json) |
| Blocked / loop / timeout | 0 / 2 / 0 | 10 / 73 / 0 | Same validation episodes |

These results mean 48.46% less packed parameter storage, 44.44% fewer CNN BRAM equivalents, and 21.34% fewer CNN LUTs for W4A4, together with a 2.25 percentage-point decrease in validation route success. There is no measured per-decision speedup in this implementation. Zero DSP usage means synthesis implemented the arithmetic in logic; it does not mean that no multiplication occurs.

The input dataset contains 150 validation wall layouts with three goals and eight fixed starts per goal. Its randomly scattered obstacle distribution does not establish success on every hand-edited maze. The reserved final-test partition has not been evaluated. See the [photographed counterexamples](../results/navigation/counterexamples/README.md) for failures outside the original validation distribution, including a map where only W4A4 succeeds.

## 1. Runtime: compare equal work

Inference latency is:

```text
inference time = completed CNN cycles / clock frequency
              = 3,217,153 / 100,000,000
              = 0.03217153 seconds per decision
```

The cycle count comes from the CNN's start-to-completion counter, verified by the RTL testbench. The clock frequency comes from the successful routed 10 ns constraint; this is not an independent measurement of oscillator frequency. The physical board photographs corroborate the displayed counter values.

The navigation display reports cumulative inference cycles as `CYC HEX`. For example, `02AF420E` is 45,040,142 decimal cycles, or 450.40142 ms of inference for 14 decisions at 100 MHz. Divide the cumulative cycles by `TRIES` for a per-decision count when `TRIES > 0`; start equal to goal has no inference calls and no meaningful per-decision average.

The deliberate 20,000,000-cycle animation pause after each nonterminal action adds 200 ms to visible movement. That pause, button debounce, controller handshakes, VGA refresh, and any external transfer overhead are outside the reported CNN cycle total. A stopwatch on the monitor measures a different quantity. Define the start and finish events explicitly if reporting total user-visible route time.

An agent that fails after four attempts can show fewer total cycles than an agent that reaches the goal after fourteen. That is not faster successful navigation. Compare latency on identical input states, then compare route duration on the paired subset where both models reach the same goal. Report failed episodes separately. Do not compare the average length of each model's successful runs without acknowledging that the successful subsets can differ.

Both precisions currently use the same serial multiply/accumulate schedule. Narrower operands save resources, but the controller executes the same scheduled operations. Demonstrating a speed benefit would require a separately measured architectural change, such as processing more products per cycle, or a new clock target that passes timing. Positive timing slack alone is not evidence that an untested higher frequency works.

## 2. Storage: separate payload from allocated hardware

The parameter calculation is:

```text
parameter bytes = (weight count * weight bits + bias count * bias bits) / 8
W8A8            = (16,624 * 8 + 132 * 32) / 8 = 17,152
W4A4            = (16,624 * 4 + 132 * 32) / 8 = 8,840
```

The two 1,024-word feature banks require another 2,048 bytes at 8 bits or 1,024 bytes at 4 bits. These are logical storage requirements; they exclude controller registers, accumulated scores, latched inputs, and physical memory allocation overhead. The [8-bit](../model/cnn-w8a8/manifest.json) and [4-bit](../model/cnn-w4a4/manifest.json) manifests define the exact word widths and payload counts.

Vivado reports actual allocated memory blocks. In the same routed dual-engine design, `cnn8` contains eight RAMB36 blocks and two RAMB18 blocks: nine 36 Kb equivalents. `cnn4` contains four RAMB36 and two RAMB18 blocks: five equivalents. Capacity is allocated in fixed block sizes and constrained by depth, width, and ports, so payload size and block allocation are different measurements. Nominal BRAM capacity also includes parity bits that this design may not use as data. The separate VGA and controller resources are not charged to either CNN in this table, and cross-hierarchy optimization can affect attribution.

Do not use the ASCII `.mem` file size as model memory usage: each hexadecimal character and newline occupies a file byte. Do not use full `.bit` file size either: it configures the device, including routing and the rest of the design. Disk packaging, packed parameters, logical working memory, and routed FPGA allocation answer different questions.

## 3. Performance: score navigation, not just a screenshot

For every comparison, freeze the same layout/start/goal list and the movement/termination rules. Run the actual exported integer models. Let each model choose its own raw action; do not repair illegal actions with BFS or mask failures for only one precision.

Report these measures together:

- **Optimal-action accuracy:** fraction of states where the chosen action is one of the BFS shortest-path actions. Multiple correct actions can exist.
- **Route success:** fraction of episodes reaching the goal. One wrong decision can fail a route even when state accuracy is high.
- **Failure counts:** blocked moves, loops, and step-limit timeouts separately.
- **Path efficiency:** successful path length divided by the BFS optimum; 1 is optimal. Exclude zero-length routes from this ratio and state the successful sample count.
- **Paired outcomes:** both succeed, only W8A8 succeeds, only W4A4 succeeds, or both fail, on identical scenarios. Compare family and path-distance groups as well as the aggregate.

The original validation set has 3,517 paired successes, 81 successes only at W8A8, none only at W4A4, and two failures at both precisions. This is evidence for a reduction on that validation distribution, not a guarantee that 8-bit wins on every maze. A photograph showing 8-bit GOAL and 4-bit LOOP illustrates a quantization failure; the paired dataset measures how frequently it happens. Multiple starts on the same layout are correlated, so use layout-level resampling if adding uncertainty intervals.

The software reference must match the FPGA arithmetic first. Reproduce the exported integer scores and trajectory tests, then use the board for reproducible examples and controls. This separates a learned-policy error from an RTL arithmetic error.

## 4. Reproduce the current report

From the repository root:

```powershell
python scripts/summarize_benchmarks.py --output runs/benchmark-report/summary.json
```

The output file must be new. The script calculates the table from the saved evidence, records source hashes, and rejects mismatched model, RTL, or routed-report identities. It does not retrain, run new navigation evaluations, or measure the board. It intentionally remains bound to the original Step 6 artifacts rather than silently combining a new checkpoint with old hardware results.

To reproduce the underlying numerical and hardware checks, follow [Step 3](step-03.md) and [Step 6](step-06.md). The saved measurement sources are:

- [Quantization metrics](../results/quantization/comparison.json) and [paired outcomes](../results/quantization/paired_comparison.json).
- [Routed implementation identity, clock, and total resources](../results/navigation/implementation.json).
- [Per-CNN routed resources](../results/navigation/utilization_hierarchy.rpt) and [published report hashes](../results/navigation/published_report_hashes.json).
- [Real-CNN simulation](../results/navigation/verification.json) and [physical board-test log](../results/navigation/board-test.json).

## 5. If comparing with the laptop CPU

A CPU comparison is a separate benchmark. Record the CPU model, library/backend versions, thread count, and numerical precision. Use the same input states and batch size 1 for single-decision latency, load models before timing, warm up the backend, repeat measurements, and report median and tail latency. Keep preprocessing/encoding and any host-to-board transfer time either outside both computation measurements or explicitly reported as separate end-to-end terms. Do not include training time or animation delays in inference latency.

Comparing a slow scalar Python correctness oracle against the FPGA does not establish a speedup over optimized CPU inference. Compare against an optimized implementation too, disclose when CPU and FPGA arithmetic differ, and verify action/score behavior. No FPGA-versus-CPU speedup is claimed by the current report.
