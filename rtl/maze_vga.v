`timescale 1ns/1ps
`default_nettype none

// All logic stays in the board's 100 MHz domain. Pixel updates are enabled
// every fourth clock: 25 MHz / (800*525) = 59.5238 Hz.
module maze_vga (
    input wire clk,
    input wire rst,
    input wire [63:0] walls,
    input wire [5:0] start_cell,
    input wire [5:0] goal_cell,
    input wire [5:0] cursor,
    input wire [1:0] mode,
    input wire rejected,
    output reg [3:0] red,
    output reg [3:0] green,
    output reg [3:0] blue,
    output reg hsync,
    output reg vsync,
    output reg pixel_tick,
    output reg [9:0] pixel_x,
    output reg [9:0] pixel_y,
    output reg active,
    output reg frame_start
);
    reg [1:0] phase;
    reg [9:0] scan_x, scan_y;
    reg [63:0] view_walls;
    reg [5:0] view_start, view_goal, view_cursor;
    reg [1:0] view_mode;
    reg view_rejected;

    function [34:0] font;
        input [7:0] character;
        begin
            case (character)
                8'd48: font = 35'b01110100011001110101110011000101110;
                8'd49: font = 35'b00100011000010000100001000010001110;
                8'd50: font = 35'b01110100010000100010001000100011111;
                8'd51: font = 35'b11110000010000101110000010000111110;
                8'd52: font = 35'b00010001100101010010111110001000010;
                8'd53: font = 35'b11111100001000011110000010000111110;
                8'd54: font = 35'b01110100001000011110100011000101110;
                8'd55: font = 35'b11111000010001000100010000100001000;
                8'd56: font = 35'b01110100011000101110100011000101110;
                8'd57: font = 35'b01110100011000101111000010000101110;
                8'd65: font = 35'b01110100011000111111100011000110001;
                8'd66: font = 35'b11110100011000111110100011000111110;
                8'd67: font = 35'b01111100001000010000100001000001111;
                8'd68: font = 35'b11110100011000110001100011000111110;
                8'd69: font = 35'b11111100001000011110100001000011111;
                8'd70: font = 35'b11111100001000011110100001000010000;
                8'd71: font = 35'b01111100001000010111100011000101111;
                8'd72: font = 35'b10001100011000111111100011000110001;
                8'd73: font = 35'b11111001000010000100001000010011111;
                8'd74: font = 35'b00111000100001000010100101001001100;
                8'd75: font = 35'b10001100101010011000101001001010001;
                8'd76: font = 35'b10000100001000010000100001000011111;
                8'd77: font = 35'b10001110111010110101100011000110001;
                8'd78: font = 35'b10001110011010110011100011000110001;
                8'd79: font = 35'b01110100011000110001100011000101110;
                8'd80: font = 35'b11110100011000111110100001000010000;
                8'd81: font = 35'b01110100011000110001101011001001101;
                8'd82: font = 35'b11110100011000111110101001001010001;
                8'd83: font = 35'b01111100001000001110000010000111110;
                8'd84: font = 35'b11111001000010000100001000010000100;
                8'd85: font = 35'b10001100011000110001100011000101110;
                8'd86: font = 35'b10001100011000110001100010101000100;
                8'd87: font = 35'b10001100011000110101101011010101010;
                8'd88: font = 35'b10001100010101000100010101000110001;
                8'd89: font = 35'b10001100010101000100001000010000100;
                8'd90: font = 35'b11111000010001000100010001000011111;
                8'd58: font = 35'b00000001000010000000001000010000000;
                8'd43: font = 35'b00000001000010011111001000010000000;
                default: font = 35'b0;
            endcase
        end
    endfunction

    // Eight-pixel text cells make position decode shifts rather than dividers.
    function [7:0] text_char;
        input [383:0] text;
        input integer length;
        input [5:0] index;
        begin
            if (index < length)
                text_char = text >> ((length-1-index)*8);
            else
                text_char = 8'd32;
        end
    endfunction

    reg [11:0] base_next, text_color_next;
    reg text_next;
    reg [7:0] character_next;
    reg [2:0] glyph_col_next, glyph_row_next;
    reg [8:0] grid_x, grid_y;
    reg [5:0] cell_index;
    reg [4:0] cell_x, cell_y;
    reg in_grid;
    always @* begin
        base_next = 12'h112;
        text_next = 1'b0;
        text_color_next = 12'hBCD;
        character_next = 8'd32;
        glyph_col_next = 0;
        glyph_row_next = 0;
        grid_x = 0;
        grid_y = 0;
        cell_index = 0;
        cell_x = 0;
        cell_y = 0;
        in_grid = 1'b0;

        if (scan_x >= 352 && scan_x < 608 && scan_y >= 16 && scan_y < 48)
            base_next = 12'h223;
        if (scan_x >= 32 && scan_x < 608 && scan_y == 56)
            base_next = 12'h345;
        if (scan_y >= 392 && scan_y < 404) begin
            if (scan_x >= 32 && scan_x < 44) base_next = 12'h3C8;
            if (scan_x >= 192 && scan_x < 204) base_next = 12'hF86;
            if (scan_x >= 352 && scan_x < 364) base_next = 12'hFC5;
        end

        if (scan_x >= 32 && scan_x < 160 && scan_y >= 24 && scan_y < 38) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-32) >> 1;
            glyph_row_next = (scan_y-24) >> 1;
            text_color_next = 12'hEFF;
            character_next = text_char("MAZEBITS", 8, (scan_x-32)>>4);
        end
        else if (scan_x >= 368 && scan_x < 448 && scan_y >= 24 && scan_y < 38) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-368) >> 1;
            glyph_row_next = (scan_y-24) >> 1;
            text_color_next = 12'hFC5;
            case (view_mode)
                    0: character_next = text_char("WALL ", 5, (scan_x-368)>>4);
                    1: character_next = text_char("START", 5, (scan_x-368)>>4);
                    2: character_next = text_char("GOAL ", 5, (scan_x-368)>>4);
                    default: character_next = text_char("VIEW ", 5, (scan_x-368)>>4);
                endcase
        end
        else if (scan_x >= 32 && scan_x < 320 && scan_y >= 62 && scan_y < 69) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-32) ;
            glyph_row_next = (scan_y-62) ;
            text_color_next = view_rejected ? 12'hF86 : 12'h9AC;
            if (view_rejected)
                    character_next = text_char("START AND GOAL CELLS STAY OPEN", 30, (scan_x-32)>>3);
                else
                    character_next = text_char("EDIT ONE MAP FOR BOTH PRECISIONS", 32, (scan_x-32)>>3);
        end
        else if (scan_x >= 32 && scan_x < 112 && scan_y >= 84 && scan_y < 98) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-32) >> 1;
            glyph_row_next = (scan_y-84) >> 1;
            text_color_next = 12'h4DF;
            character_next = text_char("8 BIT", 5, (scan_x-32)>>4);
        end
        else if (scan_x >= 352 && scan_x < 432 && scan_y >= 84 && scan_y < 98) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-352) >> 1;
            glyph_row_next = (scan_y-84) >> 1;
            text_color_next = 12'hC8F;
            character_next = text_char("4 BIT", 5, (scan_x-352)>>4);
        end
        else if (scan_x >= 48 && scan_x < 104 && scan_y >= 394 && scan_y < 401) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-48) ;
            glyph_row_next = (scan_y-394) ;
            text_color_next = 12'hBCD;
            character_next = text_char("S START", 7, (scan_x-48)>>3);
        end
        else if (scan_x >= 208 && scan_x < 256 && scan_y >= 394 && scan_y < 401) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-208) ;
            glyph_row_next = (scan_y-394) ;
            text_color_next = 12'hBCD;
            character_next = text_char("G GOAL", 6, (scan_x-208)>>3);
        end
        else if (scan_x >= 368 && scan_x < 416 && scan_y >= 394 && scan_y < 401) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-368) ;
            glyph_row_next = (scan_y-394) ;
            text_color_next = 12'hBCD;
            character_next = text_char("CURSOR", 6, (scan_x-368)>>3);
        end
        else if (scan_x >= 32 && scan_x < 384 && scan_y >= 420 && scan_y < 427) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-32) ;
            glyph_row_next = (scan_y-420) ;
            text_color_next = 12'hBCD;
            character_next = text_char("ARROWS MOVE   CENTER APPLIES   RESET CLEARS", 43, (scan_x-32)>>3);
        end
        else if (scan_x >= 32 && scan_x < 384 && scan_y >= 448 && scan_y < 455) begin
            text_next = 1'b1;
            glyph_col_next = (scan_x-32) ;
            glyph_row_next = (scan_y-448) ;
            text_color_next = 12'h9AC;
            character_next = text_char("SW1:0   00 WALL  01 START  10 GOAL  11 VIEW", 43, (scan_x-32)>>3);
        end

        if (scan_y >= 112 && scan_y < 368) begin
            if (scan_x >= 32 && scan_x < 288) begin
                in_grid = 1'b1;
                grid_x = scan_x - 32;
            end else if (scan_x >= 352 && scan_x < 608) begin
                in_grid = 1'b1;
                grid_x = scan_x - 352;
            end
            grid_y = scan_y - 112;
        end
        if (in_grid) begin
            cell_index = {grid_y[7:5], grid_x[7:5]};
            cell_x = grid_x[4:0];
            cell_y = grid_y[4:0];
            base_next = view_walls[cell_index] ? 12'h678 : 12'h223;
            if (cell_index == view_start && cell_index == view_goal)
                base_next = 12'hB7F;
            else if (cell_index == view_start)
                base_next = 12'h3C8;
            else if (cell_index == view_goal)
                base_next = 12'hF86;
            if (cell_x == 0 || cell_y == 0)
                base_next = 12'h345;
            if (cell_index == view_cursor &&
                (cell_x <= 2 || cell_x >= 29 || cell_y <= 2 || cell_y >= 29))
                base_next = 12'hFC5;

            if ((cell_index == view_start || cell_index == view_goal) &&
                cell_x >= 11 && cell_x < 21 && cell_y >= 9 && cell_y < 23) begin
                text_next = 1'b1;
                text_color_next = 12'h112;
                character_next = (cell_index == view_start && cell_index == view_goal) ? 8'd43 :
                                 (cell_index == view_start ? 8'd83 : 8'd71);
                glyph_col_next = (cell_x-11) >> 1;
                glyph_row_next = (cell_y-9) >> 1;
            end
        end
    end

    // Character selection and font lookup occupy separate 100 MHz pipeline
    // stages. The final color and sync update together on phase 3.
    reg [11:0] base_stage, text_color_stage;
    reg text_stage;
    reg [7:0] character_stage;
    reg [2:0] glyph_col_stage, glyph_row_stage;
    reg [11:0] color_stage;
    wire [34:0] glyph = font(character_stage);
    wire glyph_pixel = glyph_col_stage < 5 && glyph_row_stage < 7 &&
                       glyph[34 - (glyph_row_stage*5 + glyph_col_stage)];

    always @(posedge clk) begin
        if (rst) begin
            phase <= 0;
            scan_x <= 0;
            scan_y <= 0;
            view_walls <= 0;
            view_start <= 0;
            view_goal <= 63;
            view_cursor <= 0;
            view_mode <= 0;
            view_rejected <= 0;
            base_stage <= 0;
            text_color_stage <= 0;
            text_stage <= 0;
            character_stage <= 0;
            glyph_col_stage <= 0;
            glyph_row_stage <= 0;
            color_stage <= 0;
            red <= 0; green <= 0; blue <= 0;
            hsync <= 1; vsync <= 1;
            pixel_tick <= 0;
            pixel_x <= 0; pixel_y <= 0;
            active <= 0; frame_start <= 0;
        end else begin
            phase <= phase + 1'b1;
            pixel_tick <= 0;
            frame_start <= 0;
            if (phase == 0) begin
                base_stage <= base_next;
                text_color_stage <= text_color_next;
                text_stage <= text_next;
                character_stage <= character_next;
                glyph_col_stage <= glyph_col_next;
                glyph_row_stage <= glyph_row_next;
            end
            if (phase == 1)
                color_stage <= (text_stage && glyph_pixel) ? text_color_stage : base_stage;
            if (phase == 3) begin
                pixel_tick <= 1;
                pixel_x <= scan_x;
                pixel_y <= scan_y;
                active <= (scan_x < 640 && scan_y < 480);
                frame_start <= (scan_x == 0 && scan_y == 0);
                hsync <= !(scan_x >= 656 && scan_x < 752);
                vsync <= !(scan_y >= 490 && scan_y < 492);
                if (scan_x < 640 && scan_y < 480)
                    {red,green,blue} <= color_stage;
                else
                    {red,green,blue} <= 12'h000;

                // Atomic display snapshot during blanking; updates cannot tear
                // a visible frame. Initial visible frame is the empty reset map.
                if (scan_x == 0 && scan_y == 480) begin
                    view_walls <= walls;
                    view_start <= start_cell;
                    view_goal <= goal_cell;
                    view_cursor <= cursor;
                    view_mode <= mode;
                    view_rejected <= rejected;
                end
                if (scan_x == 799) begin
                    scan_x <= 0;
                    if (scan_y == 524) scan_y <= 0;
                    else scan_y <= scan_y + 1'b1;
                end else scan_x <= scan_x + 1'b1;
            end
        end
    end
endmodule
`default_nettype wire
