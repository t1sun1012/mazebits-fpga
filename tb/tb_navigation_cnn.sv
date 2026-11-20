`timescale 1ns/1ps
// Both physical precisions elaborate together; no CNN stub or policy repair.
module tb_navigation_cnn;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1, start=0;
    reg [63:0] walls=0;
    reg [5:0] start_cell=0, goal_cell=63;
    wire start8,reset8,busy8,done8,event8;
    wire start4,reset4,busy4,done4,event4;
    wire [63:0] walls8,walls4,trail8,trail4;
    wire [5:0] input_agent8,input_agent4,input_goal8,input_goal4,agent8,agent4;
    wire [1:0] action8,action4;
    wire [2:0] status8,status4;
    wire [6:0] steps8,steps4;
    wire [31:0] cycles8,cycles4,total8,total4;
    wire signed [31:0] up8,right8,down8,left8,up4,right4,down4,left4;
    maze_navigation #(.STEP_DELAY_CYCLES(0)) nav8 (
        .clk(clk),.rst(rst),.start(start),.walls(walls),.start_cell(start_cell),.goal_cell(goal_cell),
        .cnn_busy(busy8),.cnn_done(done8),.cnn_action(action8),.cnn_cycles(cycles8),
        .cnn_start(start8),.cnn_reset(reset8),.cnn_walls(walls8),.cnn_agent(input_agent8),.cnn_goal(input_goal8),
        .agent_cell(agent8),.trail(trail8),.status(status8),.steps(steps8),.total_cycles(total8),.step_event(event8));
    maze_navigation #(.STEP_DELAY_CYCLES(0)) nav4 (
        .clk(clk),.rst(rst),.start(start),.walls(walls),.start_cell(start_cell),.goal_cell(goal_cell),
        .cnn_busy(busy4),.cnn_done(done4),.cnn_action(action4),.cnn_cycles(cycles4),
        .cnn_start(start4),.cnn_reset(reset4),.cnn_walls(walls4),.cnn_agent(input_agent4),.cnn_goal(input_goal4),
        .agent_cell(agent4),.trail(trail4),.status(status4),.steps(steps4),.total_cycles(total4),.step_event(event4));
    mazebits_cnn_w8a8 #(.WEIGHTS_FILE("weights_w8a8.mem"),.BIASES_FILE("biases_w8a8.mem")) cnn8 (
        .clk(clk),.rst(reset8),.start(start8),.walls(walls8),.agent(input_agent8),.goal(input_goal8),
        .busy(busy8),.done(done8),.action(action8),.cycles(cycles8),
        .score_up(up8),.score_right(right8),.score_down(down8),.score_left(left8),
        .trace_valid(),.trace_layer(),.trace_index(),.trace_acc(),.trace_act());
    mazebits_cnn_w4a4 #(.WEIGHTS_FILE("weights_w4a4.mem"),.BIASES_FILE("biases_w4a4.mem")) cnn4 (
        .clk(clk),.rst(reset4),.start(start4),.walls(walls4),.agent(input_agent4),.goal(input_goal4),
        .busy(busy4),.done(done4),.action(action4),.cycles(cycles4),
        .score_up(up4),.score_right(right4),.score_down(down4),.score_left(left4),
        .trace_valid(),.trace_layer(),.trace_index(),.trace_acc(),.trace_act());

    reg [63:0] expected_walls[0:3];
    reg [5:0] expected_starts[0:3],expected_goals[0:3];
    reg [31:0] expected8[0:2815],expected4[0:2815];
    reg [31:0] final8[0:23],final4[0:23];
    integer current_case=0,seen8=0,seen4=0,total_events=0,score_words=0;
    integer watchdog=0,base8,base4,c;
    reg checking=0;
    reg [111:0] held8,held4;

    always @(posedge clk) begin
        #1;
        if (checking) begin
            if (event8) begin
                if (seen8>=64) $fatal(1,"too many W8 events");
                base8=current_case*64*11+seen8*11;
                if (action8 !== expected8[base8] || agent8 !== expected8[base8+1] ||
                    trail8 !== {expected8[base8+3],expected8[base8+2]} ||
                    status8 !== expected8[base8+4] || steps8 !== expected8[base8+5] ||
                    total8 !== expected8[base8+6] || cycles8 !== 32'd3217153)
                    $fatal(1,"W8 trajectory mismatch case=%0d event=%0d action=%0d agent=%0d status=%0d steps=%0d cycles=%0d",current_case,seen8,action8,agent8,status8,steps8,total8);
                if ({up8,right8,down8,left8} !== {expected8[base8+7],expected8[base8+8],expected8[base8+9],expected8[base8+10]})
                    $fatal(1,"W8 signed score mismatch case=%0d event=%0d",current_case,seen8);
                $display("NAV_EVENT bits=8 case=%0d attempt=%0d action=%0d cell=%0d status=%0d cycles=%0d scores=%0d,%0d,%0d,%0d",current_case,steps8,action8,agent8,status8,total8,up8,right8,down8,left8);
                seen8=seen8+1; total_events=total_events+1; score_words=score_words+4;
            end
            if (event4) begin
                if (seen4>=64) $fatal(1,"too many W4 events");
                base4=current_case*64*11+seen4*11;
                if (action4 !== expected4[base4] || agent4 !== expected4[base4+1] ||
                    trail4 !== {expected4[base4+3],expected4[base4+2]} ||
                    status4 !== expected4[base4+4] || steps4 !== expected4[base4+5] ||
                    total4 !== expected4[base4+6] || cycles4 !== 32'd3217153)
                    $fatal(1,"W4 trajectory mismatch case=%0d event=%0d action=%0d agent=%0d status=%0d steps=%0d cycles=%0d",current_case,seen4,action4,agent4,status4,steps4,total4);
                if ({up4,right4,down4,left4} !== {expected4[base4+7],expected4[base4+8],expected4[base4+9],expected4[base4+10]})
                    $fatal(1,"W4 signed score mismatch case=%0d event=%0d",current_case,seen4);
                $display("NAV_EVENT bits=4 case=%0d attempt=%0d action=%0d cell=%0d status=%0d cycles=%0d scores=%0d,%0d,%0d,%0d",current_case,steps4,action4,agent4,status4,total4,up4,right4,down4,left4);
                seen4=seen4+1; total_events=total_events+1; score_words=score_words+4;
            end
        end
    end

    task launch_dummy;
        begin
            @(negedge clk); walls=0; start_cell=1; goal_cell=62; start=1;
            @(negedge clk); start=0;
            repeat (100) @(negedge clk);
            if (!busy8 || !busy4) $fatal(1,"warm-up did not launch both cores");
        end
    endtask

    task run_case(input integer n);
        begin
            @(negedge clk);
            current_case=n; seen8=0; seen4=0;
            walls=expected_walls[n]; start_cell=expected_starts[n]; goal_cell=expected_goals[n];
            checking=1; start=1;
            @(negedge clk); start=0;
            // Changing the live edit inputs cannot affect the captured scenario.
            walls=64'hffffffffffffffff; start_cell=63; goal_cell=1;
            watchdog=0;
            while (status8<2 || status4<2) begin
                @(negedge clk); watchdog=watchdog+1;
                if (watchdog>210000000) $fatal(1,"episode watchdog expired case=%0d",n);
            end
            if (seen8 !== final8[n*6+1] || seen4 !== final4[n*6+1])
                $fatal(1,"event count mismatch case=%0d",n);
            if (status8 !== final8[n*6] || steps8 !== final8[n*6+1] || agent8 !== final8[n*6+2] ||
                trail8 !== {final8[n*6+4],final8[n*6+3]} || total8 !== final8[n*6+5])
                $fatal(1,"final W8 mismatch case=%0d",n);
            if (status4 !== final4[n*6] || steps4 !== final4[n*6+1] || agent4 !== final4[n*6+2] ||
                trail4 !== {final4[n*6+4],final4[n*6+3]} || total4 !== final4[n*6+5])
                $fatal(1,"final W4 mismatch case=%0d",n);
            held8={status8,steps8,agent8,trail8,total8};
            held4={status4,steps4,agent4,trail4,total4};
            repeat(12) begin
                @(negedge clk);
                if ({status8,steps8,agent8,trail8,total8} !== held8 ||
                    {status4,steps4,agent4,trail4,total4} !== held4 || start8 || start4 || busy8 || busy4)
                    $fatal(1,"terminal results changed or engine restarted case=%0d",n);
            end
            $display("NAV_CASE case=%0d steps8=%0d status8=%0d steps4=%0d status4=%0d",n,seen8,status8,seen4,status4);
            checking=0;
        end
    endtask

    initial begin
        $readmemh("walls.mem",expected_walls);
        $readmemh("starts.mem",expected_starts);
        $readmemh("goals.mem",expected_goals);
        $readmemh("events_w8a8.mem",expected8);
        $readmemh("events_w4a4.mem",expected4);
        $readmemh("final_w8a8.mem",final8);
        $readmemh("final_w4a4.mem",final4);
        repeat(4) @(negedge clk);
        rst=0;
        launch_dummy;
        rst=1;
        repeat(4) @(negedge clk);
        if (busy8 || busy4 || status8 || status4 || steps8 || steps4) $fatal(1,"reset did not abort both cores");
        rst=0;
        launch_dummy;
        // First golden run restarts BOTH real cores while they are busy.
        for(c=0;c<4;c=c+1) run_case(c);
        if (total_events!=60 || score_words!=240) $fatal(1,"incomplete scalar-vector coverage");
        $display("PASS_NAVIGATION_CNN cases=4 episodes=8 decisions=60 score_words=240 cycles_per_decision=3217153 reset_abort=1 busy_restart=1 snapshot=1 hold_terminal=1");
        $finish;
    end
endmodule
