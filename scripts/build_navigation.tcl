# Selected paired fixed-point CNN navigation and VGA implementation.
# First stage the frozen dual models with scripts/stage_navigation.py or use run_navigation_vivado.py.
# vivado -mode batch -source scripts/build_navigation.tcl -tclargs bitstream
set repo [file normalize [file join [file dirname [info script]] ..]]
set stage route
if {[llength $argv] > 0} { set stage [lindex $argv 0] }
if {$stage ni {synth route bitstream}} { error "Stage must be synth, route, or bitstream" }
set model_set original
if {[llength $argv] > 1} { set model_set [lindex $argv 1] }
if {[llength $argv] > 2 || $model_set ni {original structured}} {
    error "Usage: build_navigation.tcl stage ?original|structured?"
}
set folder navigation
set bitstream_name mazebits_navigation.bit
if {$model_set eq "structured"} {
    set folder navigation-structured
    set bitstream_name mazebits_navigation_structured.bit
}
set out [file join $repo runs vivado $folder]
file mkdir $out
cd $out
create_project -in_memory -part xc7a100tcsg324-1
set_param general.maxThreads 4
foreach name {button_conditioner.v maze_editor.v maze_navigation.v maze_navigation_vga.v mazebits_navigation_top.v} {
    read_verilog [file join $repo rtl $name]
}
foreach bits {8 4} {
    set generated [file join $out mazebits_cnn_w${bits}a${bits}.v]
    if {![file exists $generated]} { error "Run scripts/run_navigation_vivado.py to stage dual CNN models first" }
    read_verilog $generated
}
read_xdc [file join $repo constraints mazebits_navigation.xdc]
synth_design -top mazebits_navigation_top -part xc7a100tcsg324-1

# Validate CDC collections outside XDC, whose supported Tcl command set is limited.
set buttons_first [get_cells -hier -regexp {(^|.*/)sync_meta_reg\[[0-4]\]$}]
set switches_first [get_cells -hier -regexp {(^|.*/)mode_pipe0_reg\[[01]\]$}]
set reset_first [get_cells -hier -regexp {(^|.*/)reset_pipe_reg\[0\]$}]
if {[llength $buttons_first] != 5 || [llength $switches_first] != 2 || [llength $reset_first] != 1} {
    error "Expected first-stage input synchronizers not found; inspect the constraint collections"
}
report_utilization -file utilization_synth.rpt
report_timing_summary -file timing_synth.rpt
write_checkpoint -force synthesized.dcp
if {$stage ne "synth"} {
    opt_design
    place_design
    phys_opt_design
    route_design
    report_utilization -file utilization_route.rpt
    report_utilization -hierarchical -hierarchical_depth 3 -file utilization_hierarchy.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file timing_route.rpt
    report_drc -file drc_route.rpt
    report_clock_interaction -file clock_interaction.rpt
    write_checkpoint -force routed.dcp
    set setup_paths [get_timing_paths -delay_type max -max_paths 1]
    set hold_paths [get_timing_paths -delay_type min -max_paths 1]
    if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} {
        error "No timed register paths found"
    }
    set setup_slack [get_property SLACK [lindex $setup_paths 0]]
    set hold_slack [get_property SLACK [lindex $hold_paths 0]]
    set all_drc [get_drc_violations -quiet]
    set warning_drc [get_drc_violations -quiet -filter {SEVERITY == "Warning"}]
    set bad_drc [get_drc_violations -quiet -filter {SEVERITY == "Error" || SEVERITY == "Critical Warning"}]
    set memory_async_drc {}
    foreach violation $all_drc {
        if {[string match "REQP-1839*" [get_property NAME $violation]]} {
            lappend memory_async_drc $violation
        }
    }
    set handle [open build_summary.txt w]
    puts $handle "top=mazebits_navigation_top"
    puts $handle "model_set=$model_set"
    puts $handle "bitstream_name=$bitstream_name"
    puts $handle "part=xc7a100tcsg324-1"
    puts $handle "clock_period_ns=10.000"
    puts $handle "pixel_clock_enable_hz=25000000"
    puts $handle "frame_hz=59.5238095238"
    puts $handle "setup_slack_ns=$setup_slack"
    puts $handle "hold_slack_ns=$hold_slack"
    puts $handle "critical_or_error_drc_count=[llength $bad_drc]"
    puts $handle "total_drc_count=[llength $all_drc]"
    puts $handle "warning_drc_count=[llength $warning_drc]"
    puts $handle "memory_async_drc_count=[llength $memory_async_drc]"
    puts $handle "vivado_version=[version -short]"
    close $handle
    if {$setup_slack < 0 || $hold_slack < 0} { error "Timing failed: setup=$setup_slack, hold=$hold_slack" }
    if {[llength $bad_drc] > 0 || [llength $memory_async_drc] > 0} {
        error "DRC errors/critical warnings or unsafe asynchronous RAM controls remain"
    }
    if {$stage eq "bitstream"} { write_bitstream -force $bitstream_name }
}
puts "MazeBits dual-CNN navigation/VGA $stage complete; reports: $out"
exit
