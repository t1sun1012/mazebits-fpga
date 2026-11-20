// Minimal on-board inference check. The editor and VGA are later milestones.
module mazebits_smoke_top #(
    parameter integer DEBOUNCE_CYCLES = 1000000,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIASES_FILE = "biases.mem"
) (
    input wire CLK100MHZ,
    input wire CPU_RESETN,
    input wire BTNC,
    input wire [0:0] SW,
    output wire [7:0] LED
);
    `include "quantization.vh"

    // Fully synchronous reset sampling prevents asynchronous controls from
    // reaching inferred block RAMs after Vivado's power optimizations. The
    // declaration initial values provide deterministic FPGA power-up reset.
    (* ASYNC_REG = "TRUE" *) reg [1:0] reset_pipe = 2'b11;
    always @(posedge CLK100MHZ)
        reset_pipe <= {reset_pipe[0], !CPU_RESETN};
    reg reset = 1'b1;
    always @(posedge CLK100MHZ)
        reset <= reset_pipe[1];

    (* ASYNC_REG = "TRUE" *) reg [1:0] button_sync;
    (* ASYNC_REG = "TRUE" *) reg [1:0] switch_sync;
    reg button_stable;
    reg [19:0] debounce_count;
    reg button_press;

    always @(posedge CLK100MHZ) begin
        if (reset) begin
            button_sync <= 2'b00;
            switch_sync <= 2'b00;
            button_stable <= 1'b0;
            debounce_count <= 20'd0;
            button_press <= 1'b0;
        end else begin
            button_sync <= {button_sync[0], BTNC};
            switch_sync <= {switch_sync[0], SW[0]};
            button_press <= 1'b0;
            if (button_sync[1] == button_stable) begin
                debounce_count <= 20'd0;
            end else if (debounce_count == DEBOUNCE_CYCLES - 1) begin
                debounce_count <= 20'd0;
                button_stable <= button_sync[1];
                button_press <= button_sync[1];
            end else begin
                debounce_count <= debounce_count + 1'b1;
            end
        end
    end

    wire busy;
    wire done;
    wire [1:0] action;
    wire start = button_press && !busy;
    // Bit r*8+c represents row r, column c. The detour wall stops at row 6.
    wire [63:0] walls = switch_sync[1] ? 64'h0002020202020202 : 64'd0;
    wire [5:0] goal = switch_sync[1] ? 6'd7 : 6'd63;
    reg completed;
    reg selected_fixture;

    always @(posedge CLK100MHZ) begin
        if (reset) begin
            completed <= 1'b0;
            selected_fixture <= 1'b0;
        end else begin
            if (start) begin
                completed <= 1'b0;
                selected_fixture <= switch_sync[1];
            end
            if (done)
                completed <= 1'b1;
        end
    end

    mazebits_cnn #(
        .WEIGHTS_FILE(WEIGHTS_FILE),
        .BIASES_FILE(BIASES_FILE)
    ) cnn (
        .clk(CLK100MHZ), .rst(reset), .start(start),
        .walls(walls), .agent(6'd0), .goal(goal),
        .busy(busy), .done(done), .action(action),
        .score_up(), .score_right(), .score_down(), .score_left(),
        .cycles(), .trace_valid(), .trace_layer(), .trace_index(),
        .trace_acc(), .trace_act()
    );

    assign LED[1:0] = action;
    assign LED[2] = busy;
    assign LED[3] = completed;
    assign LED[4] = selected_fixture;
    assign LED[5] = (W_BITS == 4);
    assign LED[7:6] = 2'b00;
endmodule
