# Step 4: Verilog inference

Step 4 implements the frozen Step 3 integer CNN in Verilog and checks it against the committed Python reference vectors. Both builds use the same controller and network shape; the selected `quantization.vh` fixes the weight and activation width at compile time.

## Scope

The inference core accepts an 8x8 wall bitmap, an agent cell, and a goal cell. It returns four signed action scores and their argmax. It does not edit maps, move the agent, compute BFS, reject illegal actions, or draw VGA pixels. The caller handles navigation termination and movement. In particular, an at-goal input can still produce scores; the future navigation controller must stop before requesting another move.

The accompanying board top is a small button/LED inference test. The full switch/button editor, VGA display, and paired navigation are later milestones.

## Input and output contract

- `clk`: 100 MHz target clock; `rst`: synchronous active-high reset at the core interface.
- `walls[63:0]`: bit `row*8+column` is 1 for a wall. Bit 0 is the upper-left cell.
- `agent[5:0]`, `goal[5:0]`: cell indices in the same order. Both cells must be free. They may coincide.
- `start`: sampled when idle. The core snapshots all maze inputs at acceptance; changes and additional starts while busy have no effect.
- `busy`: inference is in progress. `done`: a one-clock completion pulse.
- `score_up`, `score_right`, `score_down`, `score_left`: signed 32-bit scores in one shared scale.
- `action`: 0=up, 1=right, 2=down, 3=left. Equal scores prefer the lower action index.
- `cycles`: the measured inference clock count. This excludes button debounce and display animation.

Reset aborts an in-flight inference. Result outputs remain available after completion. The hardware assumes the caller supplies a valid map/agent/goal encoding; it does not guarantee a successful route on arbitrary edited mazes.

## Arithmetic and memory

The core implements eight padded 3x3 convolution/ReLU layers with 16 output channels, followed by the four-channel 1x1 head. Input planes are generated from the wall bitmap and two cell indices. The first convolution pads the wall plane with 1 and the agent/goal planes with 0; later layers pad with 0.

The serial datapath reuses a multiplier and accumulator, with synchronous-read weight storage and two feature-map buffers that alternate between layers. Each multiply-accumulate uses three cycles: issue the reads, register the product, then accumulate it. A balanced signed comparison tree selects the final action. Weights use the committed OIHW order and feature maps use CHW order. Feature maps do not need a bulk reset: every output cell is written before the next layer reads it.

Weights are signed 8 or 4 bits. Hidden activations are unsigned and are zero-extended before multiplication. Biases and accumulators remain signed 32-bit. Requantization uses a widened intermediate, rounds half ties away from zero, applies ReLU, and saturates to the activation range. The W8A8 first layer requires a left shift by one; a design supporting only right shifts would be incorrect. The final head has no ReLU or output requantization.

The exact scales and offsets come from the selected committed model header. See [Step 3](step-03.md) for rounding examples and the proof that all partial sums fit signed 32 bits.

## Debug interface

`trace_valid` identifies a completed output value. `trace_layer` is 0 through 7 for hidden layers and 8 for the head. `trace_index` is the flattened CHW address within that layer. `trace_acc` exposes the raw signed accumulator; `trace_act` exposes the saturated hidden activation, zero-extended in the 4-bit build. The head uses `trace_act=0`.

All head positions are computed in this first implementation, including positions other than the current agent. This makes every committed intermediate reference value observable and keeps the verification simple. Only the agent's four head values determine the returned action.

## Reproducible checks

The HDL runner compiles a separate design for each precision, stages the matching model memory files, and compares every hidden accumulator, hidden activation, head value, selected score, and action against the eight committed fixtures. Protocol tests additionally exercise reset, input latching, ignored starts while busy, and repeated inference.

Run the complete reference suite without retraining or regenerating the dataset:

~~~powershell
python scripts/run_rtl.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin --bits 8 4
python scripts/run_board_smoke.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
~~~

With Icarus Verilog on PATH, use `--sim iverilog`. These runners use only Python's standard library. `--smoke` on the core runner checks the first fixture plus reset/repeat/protocol checks; the default checks all eight. GitHub Actions runs both full precision suites plus the separate board-control tests.

Build and record FPGA implementation results:

~~~powershell
python scripts/run_vivado.py --bits 8 4 --stage bitstream --vivado C:/Xilinx/Vivado/2024.1/bin/vivado.bat
~~~

`--stage route` omits bitstream generation; `--stage synth` stops after synthesis. The runner checks tool exit status, rejects critical warnings, records source hashes, and saves resources and timing in `runs/vivado/w8a8/implementation.json` and `runs/vivado/w4a4/implementation.json`. No command programs a connected board.

Read [board constraints and controls](board-constraints.md) for the supplied XDC provenance, pin table, reset/debounce behavior, and test fixtures.

## Cycle count

There are 1,063,936 multiply-accumulates and 8,448 output values per decision. Each product takes three cycles, each output takes two bias-read cycles and one store cycle, and completion takes one final cycle:

~~~text
cycles = 3 * 1,063,936 + 3 * 8,448 + 1 = 3,217,153
latency at 100 MHz = 32.17153 ms
~~~

Both widths use this schedule, so 4-bit inference has the same cycle count. Reduced precision changes memory use and logic, not the number of loops. This deliberately serial first implementation makes verification straightforward; no claim is made that it is faster than laptop CPU inference. The cycle-based latency requires routed timing closure and excludes button debounce and future VGA animation.

## Implementation results

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

Both builds pass routed setup and hold timing at 100 MHz, have zero DRC violations, and generate bitstreams in Vivado 2024.1. Each matches all 133,160 reference words across eight fixtures, plus repeat/protocol/arithmetic checks. The 93 Python tests (including five pin/reference audits) also pass.

These resources describe the minimal button/LED board test, including its wrapper and pruning of unused outputs/fixed-input logic. They are not final editable-maze/VGA system totals. Both use zero DSP blocks because Vivado implements these small multipliers in LUT logic. Board programming and physical testing remain to be done.

The generated bitstreams are local at:

- runs/vivado/w8a8/mazebits_smoke_w8a8.bit
- runs/vivado/w4a4/mazebits_smoke_w4a4.bit

See [the committed evidence](../results/rtl/README.md) for hashes, complete resources, timing, and simulation reports. The source branch contains reproducible build inputs; generated Vivado projects/checkpoints and bitstreams are ignored.

## Next milestone

Step 5 adds the switch/button maze editor and VGA renderer. Step 6 connects navigation controllers to the inference core and presents both precisions on the same edited scenario. The current smoke test performs one decision on a selected fixture, without moving an agent.
