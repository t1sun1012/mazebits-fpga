# Step 6: Paired neural maze navigation

Step 6 connects the frozen W8A8 and W4A4 CNNs to independent navigation controllers and two VGA panels. Both agents receive the same edited scenario when a comparison starts. They choose their own actions afterward. This is the existing BFS-imitation-trained policy; no reinforcement learning, online training, route search, or illegal-action repair runs on the FPGA.

Step 5's physical editor/display test was reported successful by the user on November 19, 2026. The user also reported that Step 6's open-maze and detour physical tests pass: both panels show GOAL / TRIES 014 on the open maze; the detour produces 8-bit GOAL / TRIES 021 and 4-bit BLOCKED / TRIES 004. [The board-test log](../results/navigation/board-test.json) separates these physical reports from simulator results and lists the remaining checks. Supplied monitor photographs corroborate the [open-maze](../results/navigation/hardware/open-maze.jpg) and [detour](../results/navigation/hardware/detour.jpg) outcomes and cycle-counter values.

## Controls

| SW1:SW0 | Mode | Center button |
|---|---|---|
| 00 | WALL | Toggle a wall at the cursor |
| 01 | START | Place the shared start |
| 10 | GOAL | Place the shared goal |
| 11 | VIEW | Start or restart both neural agents |

Arrows move the editor cursor in the first three modes. VIEW freezes the editor. Set the switches, allow them to settle, then press and release Center. Starting again while an inference is in progress aborts that old inference and resets both agents to the saved start. Leaving VIEW cancels both episodes and returns to editing without discarding the map. CPU RESET clears the map and resets the entire design.

Each model has separate position, visited-cell trail, action count, and terminal status. A deterministic policy revisiting a cell on the unchanged map must repeat its future behavior, so the controller stops that episode as LOOP. An illegal move stops it as BLOCKED instead of selecting a different action. Goal arrival takes priority over the step limit; the board limit is 64 attempted actions. Start equal to goal succeeds immediately with zero inference calls.

## Display and timing

The left panel is W8A8 and the right panel is W4A4. Each displays its own agent, trail, status, attempted-action count, and accumulated CNN cycles. Status is READY, RUNNING, GOAL, BLOCKED, LOOP, or LIMIT. TRIES counts attempted actions, including a final blocked attempt. CYC HEX is the sum of the CNN engines' reported cycles shown in hexadecimal, not elapsed animation time.

A 20,000,000-cycle pause follows each nonterminal action to make movement visible at a 100 MHz board clock. This deliberate 200 ms pause is excluded from CYC HEX. Both precisions use the same serial computation schedule; a narrower model does not imply a faster decision in this implementation. Resource comparisons must use the combined implementation report or measured hierarchy, not sums of the earlier separately optimized board wrappers.

The renderer snapshots the whole displayed state during vertical blanking. The two panels can finish at different times without changing the other agent's state. The clocks, VGA timing, physical controls, and pin mapping match Step 5.

LED0/1 show the accepted mode; LED2..7 show the editor cursor; LED8 means editing; LED9 indicates a rejected edit; LED10 records map changes; LED11 means shared start equals goal; LED12/13 mean the 8/4-bit agent is running; LED14 means both reached the goal; LED15 is the display heartbeat.

## Program and test

The project is at `X:\Desktop\EC_311\Project` on the current computer. Use Hardware Manager to program `runs/vivado/navigation/mazebits_navigation.bit`. The old `mazebits_editor.bit` still runs Step 5 and will not respond to Center in VIEW.

1. Connect programming USB and VGA, select the monitor VGA input, program the Step 6 bitstream, and set SW1:SW0 to 00.
2. Press CPU RESET. The map should be empty, with start at the top-left and goal at the bottom-right.
3. Set SW1:SW0 to 11 and press Center. Both agents should navigate and report GOAL with TRIES 014. LED14 should turn on.
4. Press Center again. Both paths should restart from the original start.
5. Start once more, then return to 00 during movement. Both runs should stop, and wall editing should work again.
6. Test the detour below. The 8-bit agent should report GOAL after 21 attempted actions; the 4-bit agent should report BLOCKED after 4 attempts.
7. Set the start and goal to the same free cell, enter VIEW, and press Center. Both should report GOAL with zero moves and cycles.
8. Record the physical test outcome separately from simulation results.

The reproducible detour is:

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

To draw it: reset in WALL mode; press Right then Center to place the first wall. Repeat Down then Center six times to extend column 1 through row 6. Leave the bottom row open. Change to GOAL mode, move Up six times and Right six times to the top-right cell, and press Center. Select VIEW and press Center.

## Reproduce simulation and implementation

Run from the checkout root. On this computer the existing Python environment remains at `C:\Users\t1sun\Documents\ChatGPT\EC311_Project\.venv`; use its `Scripts\python.exe`, or activate an environment and use `python` below.

~~~powershell
python scripts/run_navigation.py --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
python scripts/run_navigation_vivado.py --stage bitstream --vivado C:/Xilinx/Vivado/2024.1/bin/vivado.bat
~~~

Icarus Verilog is also supported with `--sim iverilog`. The HDL runners use Python's standard library and the committed navigation vectors; they do not retrain or require PyTorch. Regenerating the independent fixed-point references requires the project Python dependencies.

The staging helper checks the frozen model manifests, then creates two instances of the unchanged serial engine under distinct module names. It inlines each precision's own constants. The four-bit engine has actual four-bit weights and hidden activations; it is not an eight-bit model with a display label. Both specializations share the original authored core logic, and their generated-source/model hashes are recorded.

To create the persistent GUI project after staging/building, open Vivado's Tcl Console in an empty session and run:

~~~tcl
source X:/Desktop/EC_311/Project/scripts/create_navigation_project.tcl
~~~

Then use Open Project on `runs/vivado/navigation_gui/mazebits_navigation.xpr`. Source and scripts are tracked in Git; generated Vivado projects, bitstreams, and build directories are local artifacts.

## Verification evidence

The [verified detour preview](../results/navigation/detour_demo.png) renders the reference final states through the VGA module; it is not a board photograph. Reproduce it with `python scripts/render_navigation_demo.py`.

See [Step 6 evidence](../results/navigation/README.md) for final simulator coverage, path outcomes, source hashes, FPGA resources, timing, and the identity of the generated bitstream. The held-out dataset partition and frozen trained models are unchanged.
