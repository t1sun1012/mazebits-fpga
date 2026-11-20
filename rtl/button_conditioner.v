`timescale 1ns / 1ps
`default_nettype none

// Independent two-flop synchronizers and stable-time debouncers.
// DEBOUNCE_CYCLES=1,000,000 corresponds to 10 ms at 100 MHz.
// A press pulse is emitted once on a debounced rising edge, with no autorepeat.
module button_conditioner #(
    parameter integer WIDTH = 5,
    parameter integer DEBOUNCE_CYCLES = 1000000
) (
    input wire clk,
    input wire rst,
    input wire [WIDTH-1:0] raw_buttons,
    output reg [WIDTH-1:0] stable_buttons,
    output reg [WIDTH-1:0] press
);
    function integer count_width;
        input integer length;
        integer remaining;
        begin
            count_width = 0;
            remaining = length - 1;
            while (remaining > 0) begin
                count_width = count_width + 1;
                remaining = remaining >> 1;
            end
            if (count_width == 0)
                count_width = 1;
        end
    endfunction

    localparam integer COUNTER_BITS = count_width(DEBOUNCE_CYCLES);
    (* ASYNC_REG = "TRUE" *) reg [WIDTH-1:0] sync_meta;
    (* ASYNC_REG = "TRUE" *) reg [WIDTH-1:0] sync_buttons;
    reg [COUNTER_BITS-1:0] count [0:WIDTH-1];
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            sync_meta <= {WIDTH{1'b0}};
            sync_buttons <= {WIDTH{1'b0}};
            stable_buttons <= {WIDTH{1'b0}};
            press <= {WIDTH{1'b0}};
            for (i = 0; i < WIDTH; i = i + 1)
                count[i] <= {COUNTER_BITS{1'b0}};
        end else begin
            sync_meta <= raw_buttons;
            sync_buttons <= sync_meta;
            press <= {WIDTH{1'b0}};
            for (i = 0; i < WIDTH; i = i + 1) begin
                if (sync_buttons[i] == stable_buttons[i]) begin
                    count[i] <= {COUNTER_BITS{1'b0}};
                end else if (count[i] == DEBOUNCE_CYCLES - 1) begin
                    count[i] <= {COUNTER_BITS{1'b0}};
                    stable_buttons[i] <= sync_buttons[i];
                    press[i] <= sync_buttons[i];
                end else begin
                    count[i] <= count[i] + 1'b1;
                end
            end
        end
    end
endmodule
`default_nettype wire
