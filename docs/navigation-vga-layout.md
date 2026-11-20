# Step 6 navigation display

The navigation renderer preserves the Step 5 single-clock VGA timing: a 100 MHz clock, one pixel update every four cycles, 640x480 visible pixels in an 800x525 raster, negative sync pulses, and black RGB during blanking. All run and editor inputs are copied together at pixel (0,480), before the next visible frame. An update arriving on that clock edge can appear one frame later.

## Panels and controls

The left grid is W8A8 and the right grid is W4A4. Both grids have 32-pixel cells and a 256x256 extent; their upper-left corners are (32,112) and (352,112). Grid index is row*8+column.

| Display element | Location or behavior |
|---|---|
| Mode | Header: WALL, START, GOAL, or VIEW |
| Editor cursor | Amber cell border in WALL/START/GOAL; hidden in VIEW |
| Current agent | Cyan inset tile in the 8-bit panel; purple inset tile in the 4-bit panel |
| Trail | Dark cyan or dark purple visited-cell fill |
| Start and goal | Green S and coral G; overlap uses a purple plus sign |
| Occupied endpoint | Keeps its S/G/+ glyph and outer endpoint color, with the agent color inset |
| Status | At y=380: READY, RUNNING, GOAL, BLOCKED, LOOP, or LIMIT |
| TRIES | At y=396: three decimal digits; counts attempted actions, including a final blocked proposal |
| CYC HEX | At y=412: eight hexadecimal digits for cumulative CNN inference cycles |
| Controls | At y=440 and y=456 |

The cycle counter excludes intentional animation delays and controller overhead. It is not elapsed wall-clock time. Both serial CNNs currently have the same cycles per inference; the display does not imply that four-bit arithmetic is faster.

Navigation overlays and run counters are hidden until a run has started. In edit modes, the editable map remains visible with its cursor. VIEW plus Center starts or restarts the two controllers; returning to an edit mode cancels the run at the top level.

## Pixel colors

Colors are RGB444 values. Background is `112`, free cells `223`, walls `678`, grid lines `345`, start `3C8`, goal `F86`, and cursor `FC5`. The 8-bit agent/trail use `4DF`/`245`; the 4-bit agent/trail use `C8F`/`435`. A trail never colors a wall. Start/goal colors take priority over trail fill, and their glyphs remain visible inside an agent tile.

## Verification

`tb/tb_navigation_vga.sv` checks three complete frames: 1,260,000 pixel updates and 921,600 visible pixels. It checks raster order, four-clock pixel cadence, every active/blanking/sync decision, between-pixel output stability, representative independent panel colors, 11,088 status/count/cycle text pixels, complete navigation-state snapshots across mid-frame input changes, and reset behavior.

The test captures `navigation_running.ppm` and `navigation_result.ppm` directly from simulated RGB pins. Their states and counters are synthetic renderer fixtures, not measured CNN trajectories or board photographs. In particular, the 127-action fixture exercises all three decimal count digits. End-to-end navigation verification separately checks real CNN decisions against the integer Python reference.
