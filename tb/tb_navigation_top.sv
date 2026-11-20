`timescale 1ns/1ps
// Control integration only: deterministic, resettable policy stubs isolate board
// inputs, snapshot/abort/restart behavior, and independent navigation states.
module navigation_stub_core #(
    parameter integer PRECISION=8,
    parameter integer LATENCY=40
) (
    input wire clk,rst,start,
    input wire [63:0] walls,
    input wire [5:0] agent,goal,
    output reg busy,done,
    output reg [1:0] action,
    output reg [31:0] cycles,
    output wire signed [31:0] score_up,score_right,score_down,score_left,
    output wire trace_valid,
    output wire [3:0] trace_layer,
    output wire [9:0] trace_index,
    output wire signed [31:0] trace_acc,
    output wire [7:0] trace_act
);
    integer remaining;
    integer requests=0;
    reg [5:0] held_agent,held_goal;
    reg [63:0] held_walls;
    assign score_up=0;assign score_right=0;assign score_down=0;assign score_left=0;
    assign trace_valid=0;assign trace_layer=0;assign trace_index=0;assign trace_acc=0;assign trace_act=0;
    always @(posedge clk) begin
        if(rst) begin busy<=0;done<=0;cycles<=0;action<=0;remaining<=0;end
        else begin
            done<=0;
            if(start && !busy) begin
                requests<=requests+1;
                held_agent<=agent;held_goal<=goal;held_walls<=walls;
                busy<=1;remaining<=LATENCY;cycles<=0;
            end else if(busy) begin
                if(agent!==held_agent || goal!==held_goal || walls!==held_walls)
                    $fatal(1,"top changed core inputs during inference");
                if(remaining==1) begin
                    busy<=0;done<=1;cycles<=LATENCY;
                    if(PRECISION==8) begin
                        if(held_agent[2:0] < held_goal[2:0]) action<=1;
                        else if(held_agent[2:0] > held_goal[2:0]) action<=3;
                        else if(held_agent[5:3] < held_goal[5:3]) action<=2;
                        else action<=0;
                    end else begin
                        if(held_agent[5:3] < held_goal[5:3]) action<=2;
                        else if(held_agent[5:3] > held_goal[5:3]) action<=0;
                        else if(held_agent[2:0] < held_goal[2:0]) action<=1;
                        else action<=3;
                    end
                end else remaining<=remaining-1;
            end
        end
    end
endmodule
`define STUB_PORTS \
    input wire clk,rst,start, input wire [63:0] walls, input wire [5:0] agent,goal, \
    output wire busy,done, output wire [1:0] action, output wire [31:0] cycles, \
    output wire signed [31:0] score_up,score_right,score_down,score_left, \
    output wire trace_valid, output wire [3:0] trace_layer, output wire [9:0] trace_index, \
    output wire signed [31:0] trace_acc, output wire [7:0] trace_act
module mazebits_cnn_w8a8 #(parameter WEIGHTS_FILE="",parameter BIASES_FILE="") (`STUB_PORTS);
    navigation_stub_core #(.PRECISION(8)) core(.*);
endmodule
module mazebits_cnn_w4a4 #(parameter WEIGHTS_FILE="",parameter BIASES_FILE="") (`STUB_PORTS);
    navigation_stub_core #(.PRECISION(4)) core(.*);
endmodule
`undef STUB_PORTS

module tb_navigation_top;
    reg clk=0;
    always #5 clk=~clk;
    reg reset_n=0;
    reg [4:0] buttons=0;
    reg [1:0] switches=0;
    wire [15:0] led;
    wire [3:0] red,green,blue;
    wire hs,vs;
    integer checks=0, saved8,saved4,guard;
    reg [63:0] snapshot_walls;
    mazebits_navigation_top #(.DEBOUNCE_CYCLES(4),.ERROR_HOLD_CYCLES(20),.STEP_DELAY_CYCLES(2)) dut (
        .CLK100MHZ(clk),.CPU_RESETN(reset_n),
        .BTNU(buttons[0]),.BTNR(buttons[1]),.BTND(buttons[2]),
        .BTNL(buttons[3]),.BTNC(buttons[4]),.SW(switches),.LED(led),
        .VGA_R(red),.VGA_G(green),.VGA_B(blue),.VGA_HS(hs),.VGA_VS(vs)
    );
    task ticks(input integer count);
        integer i;
        begin for(i=0;i<count;i=i+1) begin @(posedge clk);#1;end end
    endtask
    task check(input bit ok,input string why);
        begin checks=checks+1;if(!ok) $fatal(1,"%s",why);end
    endtask
    task tap(input [4:0] mask);
        begin
            @(negedge clk);buttons=mask;ticks(12);
            @(negedge clk);buttons=0;ticks(12);
        end
    endtask
    task mode(input [1:0] requested);
        begin
            @(negedge clk);switches=requested;ticks(14);
            check(led[1:0]===requested,"mode acceptance");
        end
    endtask
    task ready;
        begin
            guard=0;
            while((dut.status8==1 || dut.status4==1) && guard<1600) begin ticks(1);guard=guard+1;end
            check(guard<1600,"dual run timeout");
        end
    endtask
    reg [1:0] previous_mode=0;
    integer masked_mode_exits=0;
    always @(posedge clk) begin
        #1;
        if(previous_mode==3 && dut.mode!=3) begin
            // mode changes one clock before registered run_valid clears. The
            // display input must already be hidden on this exact transition.
            check(dut.display.run_valid===0,"mode exit exposed stale display run");
            masked_mode_exits=masked_mode_exits+1;
        end
        previous_mode=dut.mode;
    end
    initial begin
        ticks(8);@(negedge clk);reset_n=1;ticks(14);
        check(dut.walls===0 && dut.start_cell===0 && dut.goal_cell===63 &&
              dut.cursor===0 && dut.run_valid===0 && led[14:12]===0,"power up");
        // An unsettled button cannot create a wall or start an episode.
        @(negedge clk);buttons=2;ticks(1);@(negedge clk);buttons=0;ticks(12);
        check(dut.cursor===0,"button bounce");
        tap(2);tap(16);
        check(dut.cursor===1 && dut.walls===64'h2 && led[10]===1,"wall editing");
        mode(3);
        check(dut.run_valid===0 && dut.status8===0 && dut.status4===0,"VIEW should wait for center");
        tap(2);check(dut.cursor===1 && dut.walls===64'h2,"VIEW arrows mutated editor");
        snapshot_walls=dut.walls;
        tap(16);
        check(dut.run_valid===1 && dut.walls8===snapshot_walls && dut.walls4===snapshot_walls &&
              dut.goal8===63 && dut.goal4===63 && led[13:12]===3,"shared run snapshot/start LEDs");
        ready();
        // Horizontal-first 8-bit stub blocks on the wall; 4-bit stub keeps going.
        check(dut.status8===3 && dut.steps8===1 && dut.agent8===0 && dut.cycles8===40,"8-bit blocked episode");
        check(dut.status4===2 && dut.steps4===14 && dut.agent4===63 && dut.cycles4===560,"4-bit independent success");
        check(led[14:12]===0 && dut.walls===snapshot_walls,"terminal LEDs/map");
        // Holding center is one restart only, even across multiple decisions.
        saved8=dut.cnn8.core.requests;saved4=dut.cnn4.core.requests;
        @(negedge clk);buttons=16;ticks(200);
        check(dut.cnn8.core.requests===saved8+1 && dut.status8===3,"held center repeatedly restarted");
        check(dut.cnn4.core.requests>saved4,"4-bit did not restart");
        @(negedge clk);buttons=0;ticks(12);
        // Returning to edit cancels both engines and hides old trails/results.
        mode(0);
        check(dut.run_valid===0 && dut.status8===0 && dut.status4===0 &&
              dut.cnn_busy8===0 && dut.cnn_busy4===0 && dut.trail8===0 && dut.trail4===0,"leave VIEW abort");
        check(dut.walls===64'h2 && dut.start_cell===0 && dut.goal_cell===63,"cancel lost editor map");
        tap(16);check(dut.walls===0,"remove wall after canceled run");
        mode(3);tap(16);
        check(dut.cnn_busy8===1 && dut.cnn_busy4===1,"expected in-flight pair");
        saved8=dut.cnn8.core.requests;saved4=dut.cnn4.core.requests;
        tap(16);
        check(dut.steps8===0 && dut.steps4===0 && dut.agent8===0 && dut.agent4===0 &&
              dut.cnn8.core.requests===saved8+1 && dut.cnn4.core.requests===saved4+1,"restart in-flight pair");
        ready();
        check(dut.status8===2 && dut.status4===2 && dut.steps8===14 && dut.steps4===14 &&
              led[14]===1 && led[13:12]===0,"both finish success");
        check(dut.trail8!==dut.trail4,"independent routes collapsed");
        // Edited start/goal overlap is an immediate zero-inference success.
        mode(1);tap(16);mode(2);tap(16);
        check(dut.start_cell===1 && dut.goal_cell===1,"endpoint editing");
        saved8=dut.cnn8.core.requests;saved4=dut.cnn4.core.requests;
        mode(3);tap(16);
        check(dut.status8===2 && dut.status4===2 && dut.steps8===0 && dut.steps4===0 &&
              dut.cnn8.core.requests===saved8 && dut.cnn4.core.requests===saved4 && led[14]===1,"overlap requested inference");
        // Reset clears the entire demo and restores physical switch mode later.
        @(negedge clk);reset_n=0;ticks(8);
        check(dut.walls===0 && dut.start_cell===0 && dut.goal_cell===63 &&
              dut.run_valid===0 && dut.status8===0 && dut.status4===0,"physical reset");
        @(negedge clk);reset_n=1;switches=0;buttons=0;ticks(14);
        check(^{red,green,blue,hs,vs}!==1'bx,"VGA outputs unknown");
        check(masked_mode_exits>=2,"mode exit display masking was not exercised");
        $display("PASS_NAVIGATION_TOP checks=%0d controls=1 snapshot=1 independent=1 restart=1 held_press=1 cancel=1 overlap=1 reset=1 mode_exit_mask=1",checks);
        $finish;
    end
    initial begin #1000000;$fatal(1,"navigation top watchdog");end
endmodule
