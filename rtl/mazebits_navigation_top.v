`timescale 1ns/1ps
`default_nettype none
// Step 6: shared editable scenario, independent W8A8/W4A4 agents, and VGA.
// VIEW + center starts/restarts both; leaving VIEW cancels both episodes.
module mazebits_navigation_top #(
    parameter integer DEBOUNCE_CYCLES = 1000000,
    parameter integer ERROR_HOLD_CYCLES = 50000000,
    parameter integer STEP_DELAY_CYCLES = 20000000,
    parameter integer MAX_STEPS = 64
) (
    input wire CLK100MHZ,
    input wire CPU_RESETN,
    input wire BTNU,
    input wire BTNR,
    input wire BTND,
    input wire BTNL,
    input wire BTNC,
    input wire [1:0] SW,
    output wire [15:0] LED,
    output wire [3:0] VGA_R,
    output wire [3:0] VGA_G,
    output wire [3:0] VGA_B,
    output wire VGA_HS,
    output wire VGA_VS
);
    function integer count_width;
        input integer n;
        integer v;
        begin
            count_width = 1;
            v = n-1;
            while (v > 1) begin
                count_width = count_width+1;
                v = v >> 1;
            end
        end
    endfunction
    localparam integer MODE_COUNT_BITS = count_width(DEBOUNCE_CYCLES);
    localparam integer ERROR_COUNT_BITS = count_width(ERROR_HOLD_CYCLES);

    // All logic remains on CLK100MHZ; no fabric-generated clock or CDC for pixels.
    (* ASYNC_REG = "TRUE" *) reg [1:0] reset_pipe = 2'b11;
    always @(posedge CLK100MHZ)
        reset_pipe <= {reset_pipe[0], !CPU_RESETN};
    reg reset = 1'b1;
    always @(posedge CLK100MHZ)
        reset <= reset_pipe[1];

    wire [4:0] press;
    button_conditioner #(.WIDTH(5), .DEBOUNCE_CYCLES(DEBOUNCE_CYCLES)) buttons (
        .clk(CLK100MHZ), .rst(reset),
        .raw_buttons({BTNC, BTNL, BTND, BTNR, BTNU}),
        .stable_buttons(), .press(press)
    );

    // Treat the mode switches as one bus: accept only a full stable candidate.
    (* ASYNC_REG = "TRUE" *) reg [1:0] mode_pipe0, mode_pipe1;
    reg [1:0] mode_candidate, mode;
    reg [MODE_COUNT_BITS-1:0] mode_count;
    always @(posedge CLK100MHZ) begin
        if (reset) begin
            mode_pipe0 <= 0;
            mode_pipe1 <= 0;
            mode_candidate <= 0;
            mode <= 0;
            mode_count <= 0;
        end else begin
            mode_pipe0 <= SW;
            mode_pipe1 <= mode_pipe0;
            if (mode_pipe1 != mode_candidate) begin
                mode_candidate <= mode_pipe1;
                mode_count <= 0;
            end else if (mode_candidate == mode) begin
                mode_count <= 0;
            end else if (mode_count == DEBOUNCE_CYCLES-1) begin
                mode <= mode_candidate;
                mode_count <= 0;
            end else begin
                mode_count <= mode_count + 1'b1;
            end
        end
    end

    wire [63:0] walls;
    wire [5:0] start_cell, goal_cell, cursor;
    wire changed, rejected;
    maze_editor editor (
        .clk(CLK100MHZ), .rst(reset),
        .move_up(press[0]), .move_right(press[1]), .move_down(press[2]),
        .move_left(press[3]), .apply(press[4]), .mode(mode),
        .walls(walls), .start_cell(start_cell), .goal_cell(goal_cell),
        .cursor(cursor), .changed(changed), .rejected(rejected)
    );

    reg [ERROR_COUNT_BITS-1:0] error_count;
    reg modified;
    always @(posedge CLK100MHZ) begin
        if (reset) begin
            error_count <= 0;
            modified <= 0;
        end else begin
            if (rejected)
                error_count <= ERROR_HOLD_CYCLES-1;
            else if (error_count != 0)
                error_count <= error_count-1'b1;
            if (changed)
                modified <= 1'b1;
        end
    end
    wire reject_visible = rejected || (error_count != 0);
    wire navigation_reset = reset || (mode != 2'b11);
    wire start_run = (mode == 2'b11) && press[4];
    reg run_valid;
    always @(posedge CLK100MHZ) begin
        if(navigation_reset) run_valid<=0;
        else if(start_run) run_valid<=1;
    end
    wire [63:0] walls8, walls4, trail8, trail4;
    wire [5:0] agent8, agent4, goal8, goal4;
    wire [2:0] status8, status4;
    wire [6:0] steps8, steps4;
    wire [31:0] cycles8, cycles4, cnn_cycles8, cnn_cycles4;
    wire cnn_start8, cnn_start4, cnn_reset8, cnn_reset4;
    wire cnn_busy8, cnn_busy4, cnn_done8, cnn_done4;
    wire [1:0] action8, action4;
    wire step_event8, step_event4;
    maze_navigation #(.MAX_STEPS(MAX_STEPS), .STEP_DELAY_CYCLES(STEP_DELAY_CYCLES)) nav8 (
        .clk(CLK100MHZ),.rst(navigation_reset),.start(start_run),
        .walls(walls),.start_cell(start_cell),.goal_cell(goal_cell),
        .cnn_busy(cnn_busy8),.cnn_done(cnn_done8),.cnn_action(action8),.cnn_cycles(cnn_cycles8),
        .cnn_start(cnn_start8),.cnn_reset(cnn_reset8),.cnn_walls(walls8),
        .cnn_agent(),.cnn_goal(goal8),.agent_cell(agent8),.trail(trail8),
        .status(status8),.steps(steps8),.total_cycles(cycles8),.step_event(step_event8)
    );
    maze_navigation #(.MAX_STEPS(MAX_STEPS), .STEP_DELAY_CYCLES(STEP_DELAY_CYCLES)) nav4 (
        .clk(CLK100MHZ),.rst(navigation_reset),.start(start_run),
        .walls(walls),.start_cell(start_cell),.goal_cell(goal_cell),
        .cnn_busy(cnn_busy4),.cnn_done(cnn_done4),.cnn_action(action4),.cnn_cycles(cnn_cycles4),
        .cnn_start(cnn_start4),.cnn_reset(cnn_reset4),.cnn_walls(walls4),
        .cnn_agent(),.cnn_goal(goal4),.agent_cell(agent4),.trail(trail4),
        .status(status4),.steps(steps4),.total_cycles(cycles4),.step_event(step_event4)
    );
    mazebits_cnn_w8a8 #(.WEIGHTS_FILE("weights_w8a8.mem"),.BIASES_FILE("biases_w8a8.mem")) cnn8 (
        .clk(CLK100MHZ),.rst(cnn_reset8),.start(cnn_start8),.walls(walls8),.agent(agent8),.goal(goal8),
        .busy(cnn_busy8),.done(cnn_done8),.action(action8),.cycles(cnn_cycles8),
        .score_up(),.score_right(),.score_down(),.score_left(),
        .trace_valid(),.trace_layer(),.trace_index(),.trace_acc(),.trace_act()
    );
    mazebits_cnn_w4a4 #(.WEIGHTS_FILE("weights_w4a4.mem"),.BIASES_FILE("biases_w4a4.mem")) cnn4 (
        .clk(CLK100MHZ),.rst(cnn_reset4),.start(cnn_start4),.walls(walls4),.agent(agent4),.goal(goal4),
        .busy(cnn_busy4),.done(cnn_done4),.action(action4),.cycles(cnn_cycles4),
        .score_up(),.score_right(),.score_down(),.score_left(),
        .trace_valid(),.trace_layer(),.trace_index(),.trace_acc(),.trace_act()
    );
    wire frame_start;
    maze_navigation_vga display (
        .clk(CLK100MHZ), .rst(reset),
        .walls(walls), .start_cell(start_cell), .goal_cell(goal_cell),
        .cursor(cursor), .mode(mode), .rejected(reject_visible),
        .run_valid(run_valid && !navigation_reset),.agent8(agent8),.agent4(agent4),
        .trail8(trail8),.trail4(trail4),.status8(status8),.status4(status4),
        .steps8(steps8),.steps4(steps4),.cycles8(cycles8),.cycles4(cycles4),
        .red(VGA_R), .green(VGA_G), .blue(VGA_B),
        .hsync(VGA_HS), .vsync(VGA_VS), .pixel_tick(),
        .pixel_x(), .pixel_y(), .active(), .frame_start(frame_start)
    );
    reg [5:0] frame_counter;
    always @(posedge CLK100MHZ) begin
        if (reset)
            frame_counter <= 0;
        else if (frame_start)
            frame_counter <= frame_counter+1'b1;
    end

    assign LED[1:0] = mode;
    assign LED[4:2] = cursor[2:0];
    assign LED[7:5] = cursor[5:3];
    assign LED[8] = (mode != 2'b11);
    assign LED[9] = reject_visible;
    assign LED[10] = modified;
    assign LED[11] = (start_cell == goal_cell);
    assign LED[12] = run_valid && (status8 == 3'd1);
    assign LED[13] = run_valid && (status4 == 3'd1);
    assign LED[14] = run_valid && (status8 == 3'd2) && (status4 == 3'd2);
    assign LED[15] = frame_counter[5];
endmodule
`default_nettype wire
