# RTL verification

The CNN testbench checks the frozen Step 3 integer vectors, including every
convolution accumulator, every hidden activation, all head outputs, the four
selected scores, and the chosen action. Each precision covers eight fixtures
(133,160 compared words), followed by a repeated inference on fixture zero.
The runner verifies model/vector SHA-256 hashes before rearranging the frozen
CHW traces into the testbench's case-major files; it does not regenerate expected
values from the RTL.

Run either simulator from any working directory:

~~~powershell
python scripts/run_rtl.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin --bits 8 4
~~~

~~~sh
python scripts/run_rtl.py --sim iverilog --bits 8 4
~~~

Use `--smoke` for one fixture plus repeat, reset, and interface checks. Build
products and JSON/log evidence are written under `runs/rtl/`. Only the Python
standard library is needed. The XSim smoke option uses a flag rather than an
equals-containing argument because Windows batch wrappers split unquoted `=`.

The bench also checks:

- 13 rounding, ReLU, saturation, and widened-left-shift boundary cases.
- Three signed argmax and lowest-index tie cases.
- Reset during an active inference, followed by a clean restart.
- Captured inputs despite changes to the live map, agent, and goal.
- Ignored start requests while busy, including the final busy edge.
- Stable public scores until completion, a one-cycle done pulse, and idle retention.
- Identical measured cycle counts across fixtures and repeated invocations.

The arithmetic checks deliberately call the core's arithmetic helper and force
its pending score registers before reset. No force is active during inference.
All golden trajectory inputs and memory files remain unchanged.

The separate board-control test uses a stub inference engine to exercise reset
synchronization, button debouncing, request handling, and result/status LEDs:

~~~powershell
python scripts/run_board_smoke.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
~~~

Cycle simulation demonstrates functional agreement. Timing closure, device
resources, and the physical board are checked separately by the Vivado build
and subsequent board testing.

## Editor and VGA verification

Step 5 adds four testbenches, run with:

~~~powershell
python scripts/run_editor.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
~~~

Use --sim iverilog for Icarus. The editor covers all 64 cells and endpoint safety, the debouncers cover bounce and held inputs, the integrated top checks actual button/switch wiring, and the VGA test checks three complete frames. RGB frame captures are produced directly by the HDL. See [Step 5](../docs/step-05.md) and [VGA layout](../docs/vga-layout.md) for precise coverage and physical-board test instructions.
