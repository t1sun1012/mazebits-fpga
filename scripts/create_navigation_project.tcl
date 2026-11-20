# Create/open a persistent selected-model GUI project without programming hardware.
# Interactive improved model:
#   set mazebits_model_set structured
#   source X:/Desktop/EC_311/Project/scripts/create_navigation_project.tcl
# For the original project, explicitly set mazebits_model_set original first.
# Ambient argv is deliberately ignored when this file is sourced in the GUI.
set mazebits_project_repo [file normalize [file join [file dirname [info script]] ..]]

proc mazebits_read_binary {path} {
    if {![file isfile $path]} { error "Missing project input: $path" }
    set handle [open $path rb]
    set bytes [read $handle]
    close $handle
    return $bytes
}

proc mazebits_open_navigation_project {model_set} {
    global mazebits_project_repo
    if {$model_set ni {original structured}} { error "Model set must be original or structured" }
    set repo $mazebits_project_repo
    set generated_folder navigation
    set gui_folder navigation_gui
    set project_name mazebits_navigation
    set model_prefix cnn-
    if {$model_set eq "structured"} {
        set generated_folder navigation-structured
        set gui_folder navigation_structured_gui
        set project_name mazebits_navigation_structured
        set model_prefix cnn-structured-
    }
    set generated [file join $repo runs vivado $generated_folder]
    set out [file join $repo runs vivado $gui_folder]
    set xpr [file join $out ${project_name}.xpr]
    set expected {}
    foreach name {button_conditioner.v maze_editor.v maze_navigation.v maze_navigation_vga.v mazebits_navigation_top.v} {
        lappend expected [file normalize [file join $repo rtl $name]]
    }
    # Check staged source and memories against the explicitly selected bundle,
    # even when reopening an existing project. Never silently open old weights.
    set engine [string map [list "\r\n" "\n"] [mazebits_read_binary [file join $repo rtl mazebits_cnn.v]]]
    foreach bits {8 4} {
        set model [file join $repo model ${model_prefix}w${bits}a${bits}]
        set header [string trimright [string map [list "\r\n" "\n"] [mazebits_read_binary [file join $model quantization.vh]]]]
        set specialized [string map [list "module mazebits_cnn #(" "module mazebits_cnn_w${bits}a${bits} #(" {`include "quantization.vh"} $header] $engine]
        set generated_source [file join $generated mazebits_cnn_w${bits}a${bits}.v]
        if {[string map [list "\r\n" "\n"] [mazebits_read_binary $generated_source]] ne $specialized} {
            error "Staged CNN does not match $model_set model; rerun scripts/run_navigation_vivado.py with --model-set $model_set"
        }
        lappend expected [file normalize $generated_source]
        foreach kind {weights biases} {
            set staged [file join $generated ${kind}_w${bits}a${bits}.mem]
            if {[mazebits_read_binary $staged] ne [mazebits_read_binary [file join $model ${kind}.mem]]} {
                error "Staged $kind do not match $model_set model"
            }
            lappend expected [file normalize $staged]
        }
    }
    foreach path $expected {
        if {![file isfile $path]} { error "Missing project source: $path" }
    }
    if {[file exists $xpr]} {
        open_project $xpr
    } else {
        create_project $project_name $out -part xc7a100tcsg324-1
        add_files $expected
        add_files -fileset constrs_1 [file join $repo constraints mazebits_navigation.xdc]
        set_property top mazebits_navigation_top [get_filesets sources_1]
        set_property target_language Verilog [current_project]
        update_compile_order -fileset sources_1
    }
    set actual {}
    foreach source [get_files -of_objects [get_filesets sources_1]] {
        lappend actual [file normalize $source]
    }
    if {[lsort $actual] ne [lsort $expected]} {
        error "Existing project source set does not match selected $model_set model"
    }
    set constraint_files {}
    foreach path [get_files -of_objects [get_filesets constrs_1]] {
        lappend constraint_files [file normalize $path]
    }
    if {$constraint_files ne [list [file normalize [file join $repo constraints mazebits_navigation.xdc]]]} {
        error "Existing project constraints do not match the verified board pins"
    }
    if {[get_property PART [current_project]] ne "xc7a100tcsg324-1" ||
        [get_property TOP [get_filesets sources_1]] ne "mazebits_navigation_top"} {
        error "Existing project part or top does not match the verified design"
    }
    puts "MazeBits $model_set navigation project ready: $xpr"
    return $xpr
}

if {![info exists mazebits_model_set]} { set mazebits_model_set original }
mazebits_open_navigation_project $mazebits_model_set
