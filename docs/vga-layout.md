# VGA layout and timing

The display uses two identical editor panels so the same maze can later drive the 8-bit and 4-bit navigation comparison. This milestone displays the map and editing state; no agent movement or neural outcome is simulated on screen.

## Pixel timing

All registers use the 100 MHz board clock. A four-state enable counter updates the physical RGB and sync pins every fourth clock, giving a 25 MHz pixel rate. No derived fabric clock or new clock domain is introduced.

| Axis | Visible | Front porch | Sync pulse | Back porch | Total |
|---|---:|---:|---:|---:|---:|
| Horizontal | 640 | 16 | 96 | 48 | 800 pixels |
| Vertical | 480 | 10 | 2 | 33 | 525 lines |

Both sync pulses are negative. RGB is black throughout blanking. The frame rate is 25,000,000 / (800 * 525) = 59.5238 Hz, within the 60 +/- 1 Hz range described for a 25 MHz pixel clock in the [Digilent Nexys 4 DDR reference manual](https://digilent.com/reference/_media/reference/programmable-logic/nexys-4-ddr/nexys4ddr_rm.pdf).

Character selection and bitmap/font lookup occupy separate 100 MHz pipeline stages. Final RGB, sync, active, and pixel coordinates update together on the fourth phase. The pipeline still meets a conservative 10 ns register-to-register timing constraint; no multicycle exception relaxes renderer timing.

## Editor layout

Each grid has eight rows and eight columns of 32 x 32 pixels. The left grid begins at (32,112), and the right grid at (352,112). The 8-bit and 4-bit headings identify the eventual comparison panels. The current edit mode appears at the top right.

| Visual | RGB444 | Meaning |
|---|---|---|
| Dark cell | 223 | Free space |
| Gray cell | 678 | Wall |
| Green cell with S | 3C8 | Start |
| Coral cell with G | F86 | Goal |
| Purple cell with + | B7F | Start and goal overlap |
| Amber outline | FC5 | Cursor |

A protected-endpoint message appears when the editor rejects a wall toggle on the start or goal cell. The top level controls the message's hold interval. Footer text shows the directional-button, center-button, reset, and switch-mode controls.

Walls, endpoints, cursor, mode, and rejection state are copied atomically into display registers at pixel (0,480), the start of vertical blank. Changes made during a visible frame appear in the following frame without tearing. An edit coinciding with that snapshot can wait one additional frame. Reset starts the raster with an empty maze, start 0, goal 63, cursor 0, and WALL mode; the first blanking interval captures the live editor state.

## Verification and images

The VGA testbench checks three complete frames: 1,260,000 total pixel updates and 921,600 visible pixels. It asserts the full raster order, every sync/blanking interval, four-clock update cadence, stable outputs between updates, representative colors in both panels, atomic mid-frame editing, and reset behavior.

The testbench writes baseline_frame.ppm and edited_frame.ppm directly from the physical RGB output values. The baseline frame contains a detour maze; the edited frame contains an additional wall, overlapping endpoints, a cursor on a wall, and the protected-endpoint message. PNG copies are lossless conversions of those actual HDL outputs, rather than an independently redrawn mockup.
