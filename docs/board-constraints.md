# Board constraints and the Step 4 smoke test

The supplied `Nexys-4-DDR-Master.xdc` is the pin reference for this project. Its header identifies Nexys4 DDR Rev. C. Digilent's [Nexys A7-100T master constraints](https://github.com/Digilent/digilent-xdc/blob/master/Nexys-A7-100T-Master.xdc) state that the Nexys 4 DDR was rebranded as the Nexys A7 without substantial changes and that the file also works with Nexys 4 DDR. The selected FPGA part is `xc7a100tcsg324-1`.

The original user file is archived byte for byte at [constraints/reference/Nexys-4-DDR-Master.xdc](../constraints/reference/Nexys-4-DDR-Master.xdc). Its SHA-256 is:

~~~text
36b0dd2332991699f478f430f0f01add968ff375c65c6d76a301e864ffb70572
~~~

This archived file is a reference, not an active project constraint file. Only [constraints/mazebits_smoke.xdc](../constraints/mazebits_smoke.xdc) is loaded by the Step 4 build. The active constraints match the actual smoke-test top-level ports. Unused VGA, seven-segment, memory, and other interfaces remain unconnected until later milestones. In particular, the reference specifies **LVCMOS18 for SW8 and SW9**; a future editor must preserve their individual voltage standards instead of setting every switch to LVCMOS33.

The active XDC also sets `CFGBVS=VCCO` and `CONFIG_VOLTAGE=3.3`, matching the configuration and power connections in the [official Digilent Nexys 4 DDR schematic](https://digilent.com/reference/_media/reference/programmable-logic/nexys-4-ddr/nexys-4-ddr_sch.pdf). These configuration-bank properties are separate from each individual I/O pin voltage standard.

## Active ports

| Top-level port | Package pin | Purpose |
|---|---|---|
| CLK100MHZ | E3 | 100 MHz oscillator; 10 ns constraint |
| CPU_RESETN | C12 | Active-low reset button |
| BTNC | N17 | Start one inference after debounce |
| SW0 | J15 | Select open (0) or detour (1) fixture |
| LED0, LED1 | H17, K15 | Action code, least significant bit first |
| LED2 | J13 | Inference busy |
| LED3 | N14 | Completed result available |
| LED4 | R18 | Fixture selection captured at start |
| LED5 | V17 | Precision: 0 means W8A8, 1 means W4A4 |
| LED6, LED7 | U17, U16 | Off |

The action codes are 0=up, 1=right, 2=down, 3=left. Read LED1:LED0 only after LED3 lights. The action output keeps its previous completed value during a new inference; LED3 clears when that inference starts.

Press CPU_RESETN to reset. Set SW0, allow it to settle, then press and release BTNC. BTNC is synchronized and must remain stable for 1,000,000 clock cycles (10 ms) to register a press or release. Holding the button does not retrigger inference. Presses while busy are ignored. SW0 is synchronized, and its fixture is captured when inference begins; changing it during a run does not alter that run.

CPU_RESETN is sampled through two synchronizer registers and one final synchronous reset register. The active-high internal reset follows an asserted or released button after three clock edges; the controller and CNN consume it on the next edge. Initial register values hold reset at FPGA startup. No register has an asynchronous reset or preset, which prevents asynchronous controls from reaching inferred block RAMs after implementation optimizations. The XDC excludes only external asynchronous paths into the first register of the button, switch, and reset synchronizers. The register-to-register paths inside the synchronizers and CNN remain subject to the 10 ns clock constraint. LED outputs have no clocked external receiver.

## Fixed input fixtures

Both examples start at row 0, column 0. SW0=0 selects an empty 8x8 grid with goal at row 7, column 7. SW0=1 selects a goal at row 0, column 7 and a wall in column 1 of rows 0 through 6:

~~~text
S#.....G
.#......
.#......
.#......
.#......
.#......
.#......
........
~~~

The wrapper computes **one policy decision** from that state. It does not yet move an agent, edit a map, or drive VGA. The initial action can coincide between models even when their later trajectories differ.

## Reproducible Vivado builds

From the repository root, use Vivado 2024.1 or a compatible version. On this computer:

~~~powershell
& 'C:\Xilinx\Vivado\2024.1\bin\vivado.bat' -mode batch -source scripts/build_vivado.tcl -tclargs 8 route
& 'C:\Xilinx\Vivado\2024.1\bin\vivado.bat' -mode batch -source scripts/build_vivado.tcl -tclargs 4 route
~~~

The first argument is 8 or 4. The second argument is:

- `synth`: synthesis reports and checkpoint.
- `route`: synthesis, placement, routing, timing, and DRC reports.
- `bitstream`: the same full build plus a bitstream, provided timing and DRC checks pass.

The script selects the matching exported `quantization.vh`, copies that model's weights and biases into the build directory, and uses the same part, wrapper, and 100 MHz constraints for both precisions. Reports go into `runs/vivado/w8a8` or `runs/vivado/w4a4`. Do not compare resource usage from different wrapper configurations or claim that 4-bit arithmetic must use fewer DSPs or clock cycles. Only the routed reports establish whether the 100 MHz target is met. The build records all DRC counts, rejects critical/error violations, and also rejects REQP-1839 asynchronous RAM-control warnings without changing their severity.

The wrapper exposes action and status only, so synthesis can remove unused debug, score, and cycle-counter outputs. Resource numbers from this wrapper describe that build. Board operation requires programming and a physical check; successful simulation and implementation alone do not establish that a board has been tested.

The automated pin audit in [python/tests/test_board_constraints.py](../python/tests/test_board_constraints.py) checks the source-file hash, pin/voltage matches, complete port coverage, unique pin assignments, and the 100 MHz clock constraint.

## Board-control simulation

The separate [board-control testbench](../tb/tb_mazebits_smoke_top.sv) replaces the CNN with a fixed-latency stub to isolate button debouncing, synchronization, request acceptance, reset abort/restart, fixture capture, and LED status. It does not test CNN arithmetic; the main RTL-vector testbench does that. Both precision configurations passed this control test in Vivado XSim 2024.1.

~~~powershell
& '..\.venv\Scripts\python.exe' scripts/run_board_smoke.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
~~~

The same runner supports Icarus Verilog via `--sim iverilog`. It writes a compact result and source hashes to `runs/board-controls/verification.json`. Do not compile this testbench's stub together with the real `mazebits_cnn.v` module.
