`timescale 1ns / 1ps
`default_nettype none

// All inputs are synchronous one-clock events. Bit r*8+c is maze cell (r,c).
// 00=wall, 01=start, 10=goal, 11=view (freeze all editing and cursor movement).
module maze_editor (
    input wire clk,
    input wire rst,
    input wire move_up,
    input wire move_right,
    input wire move_down,
    input wire move_left,
    input wire apply,
    input wire [1:0] mode,
    output reg [63:0] walls,
    output reg [5:0] start_cell,
    output reg [5:0] goal_cell,
    output reg [5:0] cursor,
    output reg changed,
    output reg rejected
);
    always @(posedge clk) begin
        if (rst) begin
            walls <= 64'b0;
            start_cell <= 6'd0;
            goal_cell <= 6'd63;
            cursor <= 6'd0;
            changed <= 1'b0;
            rejected <= 1'b0;
        end else begin
            changed <= 1'b0;
            rejected <= 1'b0;
            if (mode != 2'b11) begin
                // Apply wins over movement and always acts on the current cell.
                if (apply) begin
                    case (mode)
                        2'b00: begin
                            if (cursor == start_cell || cursor == goal_cell)
                                rejected <= 1'b1;
                            else begin
                                walls[cursor] <= !walls[cursor];
                                changed <= 1'b1;
                            end
                        end
                        2'b01: begin
                            walls[cursor] <= 1'b0;
                            start_cell <= cursor;
                            changed <= (start_cell != cursor) || walls[cursor];
                        end
                        2'b10: begin
                            walls[cursor] <= 1'b0;
                            goal_cell <= cursor;
                            changed <= (goal_cell != cursor) || walls[cursor];
                        end
                        default: begin end
                    endcase
                end else if (move_up) begin
                    if (cursor[5:3] != 0)
                        cursor <= cursor - 6'd8;
                end else if (move_right) begin
                    if (cursor[2:0] != 7)
                        cursor <= cursor + 6'd1;
                end else if (move_down) begin
                    if (cursor[5:3] != 7)
                        cursor <= cursor + 6'd8;
                end else if (move_left) begin
                    if (cursor[2:0] != 0)
                        cursor <= cursor - 6'd1;
                end
            end
        end
    end
endmodule
`default_nettype wire
