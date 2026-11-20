# Run scripts/run_navigation_vivado.py first to stage the dual-precision sources.
# This script creates/opens a persistent GUI project, without programming hardware.
set repo [file normalize [file join [file dirname [info script]] ..]]
set generated [file join $repo runs vivado navigation]
set out [file join $repo runs vivado navigation_gui]
set xpr [file join $out mazebits_navigation.xpr]
if {[file exists $xpr]} {
    open_project $xpr
} else {
    foreach bits {8 4} {
        if {![file exists [file join $generated mazebits_cnn_w${bits}a${bits}.v]]} {
            error "Run scripts/run_navigation_vivado.py first to stage both CNN models"
        }
    }
    create_project mazebits_navigation $out -part xc7a100tcsg324-1
    foreach name {button_conditioner.v maze_editor.v maze_navigation.v maze_navigation_vga.v mazebits_navigation_top.v} {
        add_files [file join $repo rtl $name]
    }
    foreach bits {8 4} {
        add_files [file join $generated mazebits_cnn_w${bits}a${bits}.v]
        foreach kind {weights biases} {
            add_files [file join $generated ${kind}_w${bits}a${bits}.mem]
        }
    }
    add_files -fileset constrs_1 [file join $repo constraints mazebits_navigation.xdc]
    set_property top mazebits_navigation_top [get_filesets sources_1]
    set_property target_language Verilog [current_project]
    update_compile_order -fileset sources_1
}
foreach source [get_files -of_objects [get_filesets sources_1]] {
    if {![file exists $source]} { error "Missing project source: $source" }
}
puts "MazeBits navigation project ready: $xpr"
