`timescale 1ns / 1ps
`default_nettype none

// Serial, bit-exact engine for the frozen 8x8 MazeBits CNN.
// Compile with ONE model directory on the include search path. The matching
// quantization.vh defines both physical precision and all layer scale factors.
// Input contract: agent and goal are free cells; walls uses bit row*8+column.
// All inputs are captured only when start && !busy. A reset aborts an inference.
module mazebits_cnn #(
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIASES_FILE = "biases.mem"
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [63:0] walls,
    input  wire [5:0] agent,
    input  wire [5:0] goal,
    output reg busy,
    output reg done,
    output reg [1:0] action,
    output reg signed [31:0] score_up,
    output reg signed [31:0] score_right,
    output reg signed [31:0] score_down,
    output reg signed [31:0] score_left,
    output reg [31:0] cycles,
    output reg trace_valid,
    output reg [3:0] trace_layer,
    output reg [9:0] trace_index,
    output reg signed [31:0] trace_acc,
    output reg [7:0] trace_act
);
    `include "quantization.vh"

    localparam [2:0] S_IDLE = 3'd0, S_BIAS_REQ = 3'd1,
                     S_BIAS_LOAD = 3'd2, S_ISSUE = 3'd3,
                     S_MAC = 3'd4, S_STORE = 3'd5, S_FINISH = 3'd6,
                     S_PRODUCT = 3'd7;
    reg [2:0] state;
    reg [63:0] walls_latched;
    reg [5:0] agent_latched, goal_latched;
    reg [3:0] layer, output_channel, input_channel;
    reg [2:0] x, y;
    reg [1:0] kernel_x, kernel_y;
    reg [14:0] weight_address;
    reg signed [31:0] accumulator;
    reg signed [31:0] pending_up, pending_right, pending_down, pending_left;

    // Every layer completely fills its destination bank before the next layer
    // reads it. Memories intentionally have no reset; this permits BRAM inference.
    (* rom_style = "block" *) reg signed [W_BITS-1:0] weights [0:WEIGHT_WORDS-1];
    (* rom_style = "distributed" *) reg signed [31:0] biases [0:BIAS_WORDS-1];
    (* ram_style = "block" *) reg [A_BITS-1:0] bank_a [0:1023];
    (* ram_style = "block" *) reg [A_BITS-1:0] bank_b [0:1023];
    reg signed [W_BITS-1:0] weight_q;
    reg signed [2*W_BITS:0] product_q;
    reg signed [31:0] bias_q;
    reg [A_BITS-1:0] bank_a_q, bank_b_q, binary_q;
    reg padding_q, first_layer_q, read_a_q;
    reg [A_BITS-1:0] padding_value_q;

    initial begin
        $readmemh(WEIGHTS_FILE, weights, 0, WEIGHT_WORDS-1);
        $readmemh(BIASES_FILE, biases, 0, BIAS_WORDS-1);
    end

    function [14:0] weight_offset;
        input [3:0] index;
        begin
            case (index)
                0: weight_offset = L0_WEIGHT_OFFSET;
                1: weight_offset = L1_WEIGHT_OFFSET;
                2: weight_offset = L2_WEIGHT_OFFSET;
                3: weight_offset = L3_WEIGHT_OFFSET;
                4: weight_offset = L4_WEIGHT_OFFSET;
                5: weight_offset = L5_WEIGHT_OFFSET;
                6: weight_offset = L6_WEIGHT_OFFSET;
                7: weight_offset = L7_WEIGHT_OFFSET;
                default: weight_offset = L8_WEIGHT_OFFSET;
            endcase
        end
    endfunction

    function [7:0] bias_offset;
        input [3:0] index;
        begin
            case (index)
                0: bias_offset = L0_BIAS_OFFSET;
                1: bias_offset = L1_BIAS_OFFSET;
                2: bias_offset = L2_BIAS_OFFSET;
                3: bias_offset = L3_BIAS_OFFSET;
                4: bias_offset = L4_BIAS_OFFSET;
                5: bias_offset = L5_BIAS_OFFSET;
                6: bias_offset = L6_BIAS_OFFSET;
                7: bias_offset = L7_BIAS_OFFSET;
                default: bias_offset = L8_BIAS_OFFSET;
            endcase
        end
    endfunction

    // ReLU makes negative rounded results zero. For nonnegative values,
    // adding half an LSB implements round-to-nearest with ties away from zero.
    // Widen BEFORE left shifts and rounding so saturation never wraps.
    function [A_BITS-1:0] round_relu;
        input signed [31:0] value;
        input integer shift;
        reg [63:0] widened;
        begin
            if (value < 0) begin
                round_relu = {A_BITS{1'b0}};
            end else begin
                widened = {32'b0, value};
                if (shift > 0)
                    widened = (widened + (64'd1 << (shift - 1))) >> shift;
                else if (shift < 0)
                    widened = widened << (-shift);
                if (widened > ((64'd1 << A_BITS) - 1))
                    round_relu = {A_BITS{1'b1}};
                else
                    round_relu = widened[A_BITS-1:0];
            end
        end
    endfunction

    function [A_BITS-1:0] activation;
        input signed [31:0] value;
        input [3:0] index;
        begin
            // Constant shifts become wiring/adders, rather than a wide barrel
            // shifter. In particular L0_SHIFT=-1 for the W8A8 model.
            case (index)
                0: activation = round_relu(value, L0_SHIFT);
                1: activation = round_relu(value, L1_SHIFT);
                2: activation = round_relu(value, L2_SHIFT);
                3: activation = round_relu(value, L3_SHIFT);
                4: activation = round_relu(value, L4_SHIFT);
                5: activation = round_relu(value, L5_SHIFT);
                6: activation = round_relu(value, L6_SHIFT);
                7: activation = round_relu(value, L7_SHIFT);
                default: activation = {A_BITS{1'b0}};
            endcase
        end
    endfunction

    wire is_head = (layer == 4'd8);
    wire [4:0] input_channels = (layer == 0) ? L0_C_IN : L1_C_IN;
    wire [4:0] output_channels = is_head ? L8_C_OUT : L0_C_OUT;
    wire signed [4:0] source_x = is_head ? $signed({2'b00, x}) :
        $signed({2'b00, x}) + $signed({3'b000, kernel_x}) - 5'sd1;
    wire signed [4:0] source_y = is_head ? $signed({2'b00, y}) :
        $signed({2'b00, y}) + $signed({3'b000, kernel_y}) - 5'sd1;
    wire inside_grid = source_x >= 0 && source_x < 8 &&
                       source_y >= 0 && source_y < 8;
    wire [5:0] source_cell = {source_y[2:0], source_x[2:0]};
    wire [9:0] read_address = {input_channel, source_cell};
    wire [9:0] output_address = {output_channel, y, x};
    wire [7:0] bias_address = bias_offset(layer) + output_channel;
    wire [14:0] channel_weight_offset =
        (layer == 0) ? output_channel * (L0_C_IN * L0_KERNEL * L0_KERNEL) :
        is_head ? output_channel * (L8_C_IN * L8_KERNEL * L8_KERNEL) :
        output_channel * (L1_C_IN * L1_KERNEL * L1_KERNEL);
    wire last_product = ({1'b0, input_channel} == input_channels - 1'b1) &&
                        (is_head || (kernel_x == 2 && kernel_y == 2));
    wire [A_BITS-1:0] quantized = activation(accumulator, layer);
    wire [A_BITS-1:0] operand = padding_q ? padding_value_q :
        first_layer_q ? binary_q : read_a_q ? bank_a_q : bank_b_q;

    // Activations are UNSIGNED; the leading zero is essential for codes 128..255.
    wire signed [2*W_BITS:0] product = $signed({1'b0, operand}) * weight_q;
    wire signed [31:0] product_extended =
        {{(31-2*W_BITS){product_q[2*W_BITS]}}, product_q};

    // Separate synchronous memory processes expose one read and one write port
    // per feature bank. S_ISSUE reads, S_PRODUCT registers the multiplication,
    // and S_MAC adds it. Splitting multiply/add keeps the 100 MHz path short.
    always @(posedge clk) begin
        if (state == S_PRODUCT)
            product_q <= product;
        if (state == S_BIAS_REQ)
            bias_q <= biases[bias_address];
        if (state == S_ISSUE) begin
            weight_q <= weights[weight_address];
            padding_q <= !inside_grid;
            padding_value_q <= (layer == 0 && input_channel == 0) ?
                               {{(A_BITS-1){1'b0}}, 1'b1} : {A_BITS{1'b0}};
            first_layer_q <= (layer == 0);
            read_a_q <= layer[0];
            case (input_channel)
                0: binary_q <= {{(A_BITS-1){1'b0}}, walls_latched[source_cell]};
                1: binary_q <= {{(A_BITS-1){1'b0}}, (source_cell == agent_latched)};
                default: binary_q <= {{(A_BITS-1){1'b0}}, (source_cell == goal_latched)};
            endcase
        end
    end

    always @(posedge clk) begin
        if (state == S_ISSUE && layer != 0 && layer[0] && inside_grid)
            bank_a_q <= bank_a[read_address];
        if (state == S_STORE && !is_head && !layer[0])
            bank_a[output_address] <= quantized;
    end

    always @(posedge clk) begin
        if (state == S_ISSUE && layer != 0 && !layer[0] && inside_grid)
            bank_b_q <= bank_b[read_address];
        if (state == S_STORE && !is_head && layer[0])
            bank_b[output_address] <= quantized;
    end

    reg signed [31:0] pair01_score, pair23_score, best_score;
    reg [1:0] pair01_action, pair23_action, best_action;
    always @* begin
        // Balanced tournament: at most two signed comparisons on any path.
        // Strict greater-than preserves the lower action index on every tie.
        pair01_score = pending_up;
        pair01_action = 2'd0;
        if (pending_right > pending_up) begin
            pair01_score = pending_right;
            pair01_action = 2'd1;
        end
        pair23_score = pending_down;
        pair23_action = 2'd2;
        if (pending_left > pending_down) begin
            pair23_score = pending_left;
            pair23_action = 2'd3;
        end
        best_score = pair01_score;
        best_action = pair01_action;
        if (pair23_score > pair01_score) begin
            best_score = pair23_score;
            best_action = pair23_action;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            action <= 2'd0;
            score_up <= 0;
            score_right <= 0;
            score_down <= 0;
            score_left <= 0;
            cycles <= 0;
            trace_valid <= 1'b0;
            trace_layer <= 0;
            trace_index <= 0;
            trace_acc <= 0;
            trace_act <= 0;
            walls_latched <= 0;
            agent_latched <= 0;
            goal_latched <= 0;
            layer <= 0;
            output_channel <= 0;
            input_channel <= 0;
            x <= 0;
            y <= 0;
            kernel_x <= 0;
            kernel_y <= 0;
            weight_address <= 0;
            accumulator <= 0;
            pending_up <= 0;
            pending_right <= 0;
            pending_down <= 0;
            pending_left <= 0;
        end else begin
            done <= 1'b0;
            trace_valid <= 1'b0;
            if (busy)
                cycles <= cycles + 1'b1;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        walls_latched <= walls;
                        agent_latched <= agent;
                        goal_latched <= goal;
                        layer <= 0;
                        output_channel <= 0;
                        x <= 0;
                        y <= 0;
                        cycles <= 0;
                        busy <= 1'b1;
                        state <= S_BIAS_REQ;
                    end
                end
                S_BIAS_REQ: state <= S_BIAS_LOAD;
                S_BIAS_LOAD: begin
                    accumulator <= bias_q;
                    weight_address <= weight_offset(layer) + channel_weight_offset;
                    input_channel <= 0;
                    kernel_x <= 0;
                    kernel_y <= 0;
                    state <= S_ISSUE;
                end
                S_ISSUE: state <= S_PRODUCT;
                S_PRODUCT: state <= S_MAC;
                S_MAC: begin
                    accumulator <= accumulator + product_extended;
                    if (last_product) begin
                        state <= S_STORE;
                    end else begin
                        weight_address <= weight_address + 1'b1;
                        if (is_head) begin
                            input_channel <= input_channel + 1'b1;
                        end else if (kernel_x == 2) begin
                            kernel_x <= 0;
                            if (kernel_y == 2) begin
                                kernel_y <= 0;
                                input_channel <= input_channel + 1'b1;
                            end else begin
                                kernel_y <= kernel_y + 1'b1;
                            end
                        end else begin
                            kernel_x <= kernel_x + 1'b1;
                        end
                        state <= S_ISSUE;
                    end
                end
                S_STORE: begin
                    trace_valid <= 1'b1;
                    trace_layer <= layer;
                    trace_index <= output_address;
                    trace_acc <= accumulator;
                    trace_act <= is_head ? 8'b0 : quantized;
                    if (is_head && {y, x} == agent_latched) begin
                        case (output_channel)
                            0: pending_up <= accumulator;
                            1: pending_right <= accumulator;
                            2: pending_down <= accumulator;
                            3: pending_left <= accumulator;
                            default: ; // Head channels are exactly 0..3.
                        endcase
                    end
                    state <= S_BIAS_REQ;
                    if (x != 7) begin
                        x <= x + 1'b1;
                    end else begin
                        x <= 0;
                        if (y != 7) begin
                            y <= y + 1'b1;
                        end else begin
                            y <= 0;
                            if ({1'b0, output_channel} != output_channels - 1'b1) begin
                                output_channel <= output_channel + 1'b1;
                            end else begin
                                output_channel <= 0;
                                if (is_head)
                                    state <= S_FINISH;
                                else
                                    layer <= layer + 1'b1;
                            end
                        end
                    end
                end
                S_FINISH: begin
                    score_up <= pending_up;
                    score_right <= pending_right;
                    score_down <= pending_down;
                    score_left <= pending_left;
                    action <= best_action;
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= S_IDLE;
                end
                default: begin
                    state <= S_IDLE;
                    busy <= 1'b0;
                end
            endcase
        end
    end
endmodule
`default_nettype wire
