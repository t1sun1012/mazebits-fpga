# Step 8: Improved policies on the FPGA

This step selects the Step 7 structured-maze weights explicitly for the existing dual-CNN navigation design. The original bundles and their bitstream remain separately available. The arithmetic core, navigation controller, VGA renderer, pins, and buttons are unchanged.

## Build and verify the selected checkpoint

Run from the repository root with the project Python environment:

```powershell
python scripts/run_rtl.py --model-set structured --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin --bits 8 4
python scripts/run_navigation.py --model-set structured --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
python scripts/run_navigation_vivado.py --model-set structured --stage bitstream --vivado C:/Xilinx/Vivado/2024.1/bin/vivado.bat
```

Each selector defaults to `original` when omitted. Improved outputs use `runs/rtl-structured`, `runs/navigation-structured`, and `runs/vivado/navigation-structured`. The new bitstream is `mazebits_navigation_structured.bit`; `runs/vivado/navigation/mazebits_navigation.bit` remains the original Step 6 build.

Staging verifies both integer manifests/memory files, the fixed architecture, and their shared floating-point checkpoint, dataset, and calibration identities. The seven closed-loop reference cases include the three photographed development mazes. They contain 14 episodes, 157 decisions, and 628 signed score words. Every reference decision was independently scalar-checked before HDL simulation. No BFS planner or action repair runs on the FPGA.

## Recorded verification

The final software suite passes 153 tests. Each improved RTL core passes eight full fixtures (133,160 intermediate/output words), a repeated inference, 13 rounding boundaries, three signed/tie argmax cases, and reset/busy/input-snapshot checks. Complete dual-CNN simulation passes all seven cases: 14 episodes, 157 decisions, and 628 exact signed-score comparisons. Controller tests pass 202 checks, integrated controls pass 34, and VGA simulation checks three full frames. The original four-case regression also passes, retaining its 60 decisions and 240 scores.

[Published navigation evidence](../results/navigation-structured/README.md) and [core evidence](../results/rtl-structured/README.md) keep these scopes separate. The dual-navigation simulation records refer to source commit `c4968ff0316dca7a4f0b64a68abb8d67e3f4bbcb`. A later staging integrity fix is covered by an explicit [byte-equivalence check](../results/navigation-structured/final-stager-equivalence.json): all six generated HDL/memory files for each checkpoint remain identical to the inputs actually simulated. Historical source hashes have not been rewritten.

These checks establish numerical equivalence for the exercised inputs. They do not establish successful navigation for every editable maze. The same raw policies still have the validation failures reported in Step 7.

## Open in Vivado

The local project is `X:/Desktop/EC_311/Project/runs/vivado/navigation_structured_gui/mazebits_navigation_structured.xpr`. Open it with File > Open Project. It references the improved staged memory files and both specialized CNN modules.

To recreate it after staging, use the Vivado Tcl console:

```tcl
cd {X:/Desktop/EC_311/Project}
set mazebits_model_set structured
source scripts/create_navigation_project.tcl
```

The script creates or reopens the selected project and checks all source files. It does not program the board.

## Implemented resources and timing

The final combined build passes the 100 MHz timing constraint with setup slack +0.603 ns, hold slack +0.042 ns, and zero DRC violations. It uses 2,176 LUTs, 1,695 registers, 14.5 BRAM36 equivalents, and no DSP blocks. The CNN instances use 632/524 LUTs and 9/5 BRAM36 equivalents for 8/4 bits respectively. Both retain 3,217,153 cycles per decision (32.17153 ms at 100 MHz). [The benchmark record](../results/navigation-structured/benchmark.json) binds these measurements to the new checkpoint.

The final bitstream SHA-256 is `3aa08afa503ab2ac9856d48f5c5aa702bbe53fa5b03d158081285be3a606536b`. Both the original bitstream and original GUI project were verified unchanged. The improved GUI project was opened and reopened with the selected sources.

## First physical checks

In Hardware Manager, select the XC7A100T, choose Program Device, and browse to:

```text
X:\Desktop\EC_311\Project\runs\vivado\navigation-structured\mazebits_navigation_structured.bit
```

1. Set SW1:SW0 to **00**, then press and release **CPU RESET**. Set **11** and press **Center**. Both panels should show **GOAL / TRIES 014 / CYC HEX 02AF420E**.
2. Reset again in **00**. Press **Right, Center** to place the first wall. Repeat **Down, Center** six times, leaving a wall in column 1 from rows 0 through 6.
3. Set **10** (GOAL). Press **Up six times**, then **Right six times**, then **Center** to place G at the top-right.
4. Set **11** and press **Center**. The expected result is **8-bit GOAL / TRIES 021 / 0406E315** and **4-bit LOOP / TRIES 008 / 0188B808**. The original bitstream produced 4-bit BLOCKED / 004 here, so this helps distinguish the checkpoints.

Mode bits are SW1 then SW0. Modes are 00=WALL, 01=START, 10=GOAL, and 11=VIEW. Arrows move the cursor in editing modes; Center applies the edit. VIEW+Center starts or restarts both policies. CPU RESET clears the map.

## Check the previously failed 26-move maze

Recreate this map, where `#` is a wall and each line is one row:

```text
S#G.....
.######.
.#......
.#......
.#......
.#......
.#......
........
```

From reset in WALL mode: press Right/Center; repeat Down/Center six times. Move Up five times to row 1, then repeat Right/Center five times to extend that row's wall through column 6. Set GOAL mode (10), move Up once and Left four times, then Center. Set VIEW (11) and press Center.

Expected: **8-bit GOAL / TRIES 026 / CYC HEX 04FC561A**; **4-bit LOOP / TRIES 004 / CYC HEX 00C45C04**. The original floating-point and 8-bit policies looped on this layout; the improved 8-bit policy finds the shortest 26-move route. The 4-bit result remains a quantization failure rather than evidence that the maze is unsolvable.

## Other reference outcomes

| Scenario | Improved 8-bit | Improved 4-bit |
|---|---|---|
| Open | GOAL / 014 | GOAL / 014 |
| Single-column detour | GOAL / 021 | LOOP / 008 |
| Disconnected goal | LOOP / 003 | LOOP / 005 |
| Start equals goal | GOAL / 000 | GOAL / 000 |
| IMG_7179 precision-reversal map | GOAL / 016 | GOAL / 016 |
| IMG_7180 long detour | GOAL / 026 | LOOP / 004 |
| Latest photographed elbow barrier | GOAL / 014 | GOAL / 016 |

The last 4-bit route reaches the goal with two extra moves; its shortest route is 14. All known photographs remain development diagnostics, separate from the aggregate validation measurements in [Step 7](step-07.md).

During a run, also confirm Center restarts both trails, switching back to WALL stops navigation without clearing the map, and START overlapping GOAL produces immediate GOAL / 000 / 00000000. Physical observations must be recorded separately from simulation results.

See [runtime, memory, and quality measurement definitions](benchmarking.md). `CYC HEX` counts inference cycles, excluding the intentional animation pauses. A failed route's smaller total is not faster successful navigation.

The improved-build physical open-maze and detour checks passed, as reported by the user. Additional observations and pending checks are tracked in [board-test.json](../results/navigation-structured/board-test.json).
