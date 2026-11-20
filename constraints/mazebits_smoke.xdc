# Active constraints for rtl/mazebits_smoke_top.v only.
# Pin assignments copied from the user-provided Nexys-4-DDR-Master.xdc.
# Configuration voltage: Digilent Nexys 4 DDR schematic, configuration/power sheets.
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports CLK100MHZ]
create_clock -name sys_clk_pin -period 10.000 [get_ports CLK100MHZ]

set_property -dict { PACKAGE_PIN C12 IOSTANDARD LVCMOS33 } [get_ports CPU_RESETN]
set_property -dict { PACKAGE_PIN N17 IOSTANDARD LVCMOS33 } [get_ports BTNC]
set_property -dict { PACKAGE_PIN J15 IOSTANDARD LVCMOS33 } [get_ports {SW[0]}]

set_property -dict { PACKAGE_PIN H17 IOSTANDARD LVCMOS33 } [get_ports {LED[0]}]
set_property -dict { PACKAGE_PIN K15 IOSTANDARD LVCMOS33 } [get_ports {LED[1]}]
set_property -dict { PACKAGE_PIN J13 IOSTANDARD LVCMOS33 } [get_ports {LED[2]}]
set_property -dict { PACKAGE_PIN N14 IOSTANDARD LVCMOS33 } [get_ports {LED[3]}]
set_property -dict { PACKAGE_PIN R18 IOSTANDARD LVCMOS33 } [get_ports {LED[4]}]
set_property -dict { PACKAGE_PIN V17 IOSTANDARD LVCMOS33 } [get_ports {LED[5]}]
set_property -dict { PACKAGE_PIN U17 IOSTANDARD LVCMOS33 } [get_ports {LED[6]}]
set_property -dict { PACKAGE_PIN U16 IOSTANDARD LVCMOS33 } [get_ports {LED[7]}]

# Cut only asynchronous source-to-first-stage paths. The second stages remain timed.
set button_first [get_cells -hier -regexp {(^|.*/)button_sync_reg\[0\]$}]
set switch_first [get_cells -hier -regexp {(^|.*/)switch_sync_reg\[0\]$}]
set reset_first [get_cells -hier -regexp {(^|.*/)reset_pipe_reg\[0\]$}]
set_false_path -from [get_ports BTNC] -to [get_pins -of_objects $button_first -filter {REF_PIN_NAME == D}]
set_false_path -from [get_ports {SW[0]}] -to [get_pins -of_objects $switch_first -filter {REF_PIN_NAME == D}]
set_false_path -from [get_ports CPU_RESETN] -to [get_pins -of_objects $reset_first -filter {REF_PIN_NAME == D}]
# LEDs have no synchronous receiver. This cut does not relax core timing.
set_false_path -to [get_ports {LED[*]}]
