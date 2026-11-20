# Vivado batch entry point; run from any directory.
# vivado -mode batch -source scripts/build_vivado.tcl -tclargs 8 route
# Precision: 8 or 4. Stage: synth, route (default), bitstream.
set repo [file normalize [file join [file dirname [info script]] ..]]
set bits 8
set stage route
if {[llength $argv] > 0} { set bits [lindex $argv 0] }
if {[llength $argv] > 1} { set stage [lindex $argv 1] }
if {$bits ni {8 4}} { error "Precision must be 8 or 4" }
if {$stage ni {synth route bitstream}} { error "Stage must be synth, route, or bitstream" }

set out [file join $repo runs vivado w${bits}a${bits}]
file mkdir $out
cd $out
set model [file join $repo model cnn-w${bits}a${bits}]
file copy -force [file join $model weights.mem] [file join $out weights.mem]
file copy -force [file join $model biases.mem] [file join $out biases.mem]
create_project -in_memory -part xc7a100tcsg324-1
set_param general.maxThreads 4
set_property include_dirs [list $model] [current_fileset]
read_verilog [file join $repo rtl mazebits_cnn.v]
read_verilog [file join $repo rtl mazebits_smoke_top.v]
read_xdc [file join $repo constraints mazebits_smoke.xdc]

synth_design -top mazebits_smoke_top -part xc7a100tcsg324-1
# XDC permits constraint commands; validate design-dependent collections here.
set button_first [get_cells -hier -regexp {(^|.*/)button_sync_reg\[0\]$}]
set switch_first [get_cells -hier -regexp {(^|.*/)switch_sync_reg\[0\]$}]
set reset_stages [get_cells -hier -regexp {(^|.*/)reset_pipe_reg\[[01]\]$}]
if {[llength $button_first] != 1 || [llength $switch_first] != 1 || [llength $reset_stages] != 2} {
    error "Expected CDC synchronizer registers not found; review constraint names"
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
    report_timing_summary -delay_type min_max -report_unconstrained -file timing_route.rpt
    report_drc -file drc_route.rpt
    report_clock_interaction -file clock_interaction.rpt
    write_checkpoint -force routed.dcp

    set setup_paths [get_timing_paths -delay_type max -max_paths 1]
    set hold_paths [get_timing_paths -delay_type min -max_paths 1]
    if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} {
        error "No timed register paths found; review constraints"
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
    puts $handle "precision=W${bits}A${bits}"
    puts $handle "part=xc7a100tcsg324-1"
    puts $handle "clock_period_ns=10.000"
    puts $handle "setup_slack_ns=$setup_slack"
    puts $handle "hold_slack_ns=$hold_slack"
    puts $handle "critical_or_error_drc_count=[llength $bad_drc]"
    puts $handle "total_drc_count=[llength $all_drc]"
    puts $handle "warning_drc_count=[llength $warning_drc]"
    puts $handle "memory_async_drc_count=[llength $memory_async_drc]"
    puts $handle "vivado_version=[version -short]"
    close $handle
    if {$setup_slack < 0 || $hold_slack < 0} {
        error "Timing failed: setup $setup_slack ns, hold $hold_slack ns"
    }
    if {[llength $bad_drc] > 0} {
        error "Critical/error DRC violations; see drc_route.rpt"
    }
    if {[llength $memory_async_drc] > 0} {
        error "REQP-1839 asynchronous RAM-control warnings must be resolved; see drc_route.rpt"
    }
    if {$stage eq "bitstream"} {
        write_bitstream -force mazebits_smoke_w${bits}a${bits}.bit
    }
}
puts "MazeBits W${bits}A${bits} $stage completed; reports: $out"
exit
