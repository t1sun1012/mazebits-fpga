# Step 4 evidence

Both fixed-precision CNNs pass the frozen Step 3 reference vectors in Vivado XSim 2024.1. Each compares 133,160 intermediate/output words over eight fixtures, plus an additional repeated inference. Thirteen arithmetic boundary checks, three signed/tie argmax checks, and the start/reset/retention protocol checks pass for each precision.

- [Numerical and protocol verification](verification.json); [8-bit log](w8a8_simulation.txt), [4-bit log](w4a4_simulation.txt).
- [Board controls](board_controls.json): reset, debounce, busy handling, fixture capture, and LEDs with a stub CNN.
- [8-bit implementation](w8a8_implementation.json), [4-bit implementation](w4a4_implementation.json): source/model hashes, resources, timing, and bitstream identity.
- Routed reports: [8-bit timing](w8a8/timing_route.rpt), [4-bit timing](w4a4/timing_route.rpt), [8-bit resources](w8a8/utilization_route.rpt), [4-bit resources](w4a4/utilization_route.rpt).

| Routed board-test metric | W8A8 | W4A4 |
|---|---:|---:|
| LUTs | 496 | 366 |
| Registers | 273 | 258 |
| 36 Kb BRAM tile equivalents | 9 | 5 |
| DSP blocks | 0 | 0 |
| Setup slack (ns) | 1.497 | 1.757 |
| Hold slack (ns) | 0.122 | 0.073 |
| DRC violations | 0 | 0 |
| Cycles per decision | 3,217,153 | 3,217,153 |
| Latency at 100 MHz | 32.17153 ms | 32.17153 ms |

Both builds target xc7a100tcsg324-1 with a 10 ns clock. Setup and hold timing pass, no DRC violations remain, and both bitstreams were generated. Bitstreams remain local in runs/vivado; hardware programming and physical board testing have not been performed.

Resource counts describe mazebits_smoke_top with its two selectable fixed fixtures and connected action/status outputs. They include wrapper logic and synthesis pruning. They are not counts for the future full editable/VGA system, which will need new implementation measurements. The core accepts programmable map inputs and is verified across eight varied reference fixtures. Both widths have the same cycle count because they share the serial schedule.

The BRAM figures are 36 Kb tile equivalents (one RAMB18 counts as half). Physical BRAM granularity explains why these figures differ from Step 3's logical parameter-byte totals. Both builds infer the small multiplier in LUT logic, so both use zero DSP blocks.

Source hashes connect the final RTL simulation and implementation to the same core. Committed reports omit the workstation Host line and replace any repository path with <repo>; adjacent published_report_hashes.json records these public bytes. The implementation JSON retains the raw local report hashes.

Reproduce with the commands in [Step 4](../../docs/step-04.md). No training data or retraining is required for HDL simulation or implementation. The reserved test partition is untouched.
