`timescale 1ns/1ps
`default_nettype none
// Step 6: the verified 640x480 raster with independently rendered neural agents.
// All navigation state is copied together in vertical blank; no frame tearing.
module maze_navigation_vga (
    input wire clk, rst,
    input wire [63:0] walls,
    input wire [5:0] start_cell, goal_cell, cursor,
    input wire [1:0] mode,
    input wire rejected,
    input wire run_valid,
    input wire [5:0] agent8, agent4,
    input wire [63:0] trail8, trail4,
    input wire [2:0] status8, status4,
    input wire [6:0] steps8, steps4,
    input wire [31:0] cycles8, cycles4,
    output reg [3:0] red, green, blue,
    output reg hsync, vsync, pixel_tick,
    output reg [9:0] pixel_x, pixel_y,
    output reg active, frame_start
);
    reg [1:0] phase;
    reg [9:0] scan_x, scan_y;
    reg [63:0] view_walls;
    reg [5:0] view_start, view_goal, view_cursor;
    reg [1:0] view_mode;
    reg view_rejected, view_run_valid;
    reg [5:0] view_agent8, view_agent4;
    reg [63:0] view_trail8, view_trail4;
    reg [2:0] view_status8, view_status4;
    reg [6:0] view_steps8, view_steps4;
    reg [31:0] view_cycles8, view_cycles4;
    reg [11:0] view_steps_bcd8, view_steps_bcd4;

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

    // Seven-bit attempt counts (0..127) use a small fixed combinational
    // double-dabble network, registered once per frame. No division in pixels.
    function [11:0] decimal_digits;
        input [6:0] value;
        reg [11:0] bcd;
        integer i;
        begin
            bcd = 0;
            for (i=6; i>=0; i=i-1) begin
                if (bcd[3:0] >= 5) bcd[3:0] = bcd[3:0]+4'd3;
                if (bcd[7:4] >= 5) bcd[7:4] = bcd[7:4]+4'd3;
                if (bcd[11:8] >= 5) bcd[11:8] = bcd[11:8]+4'd3;
                bcd = {bcd[10:0],value[i]};
            end
            decimal_digits = bcd;
        end
    endfunction

    function [7:0] hex_character;
        input [3:0] nibble;
        begin
            hex_character = (nibble < 10) ? 8'd48+{4'b0,nibble} : 8'd55+{4'b0,nibble};
        end
    endfunction

    function [63:0] status_text;
        input [2:0] value;
        begin
            case (value)
                0: status_text = "READY   ";
                1: status_text = "RUNNING ";
                2: status_text = "GOAL    ";
                3: status_text = "BLOCKED ";
                4: status_text = "LOOP    ";
                5: status_text = "LIMIT   ";
                default: status_text = "ERROR   ";
            endcase
        end
    endfunction

    reg [11:0] base_next, text_color_next;
    reg text_next;
    reg [7:0] character_next;
    reg [2:0] glyph_col_next, glyph_row_next;
    reg [8:0] grid_x, grid_y, panel_x;
    reg [5:0] cell_index;
    reg [4:0] cell_x, cell_y;
    reg in_grid, in_panel, right_panel;
    reg [5:0] panel_agent;
    reg [63:0] panel_trail;
    reg [2:0] panel_status;
    reg [11:0] panel_steps, agent_color, trail_color;
    reg [31:0] panel_cycles;
    reg [5:0] character_index;
    always @* begin
        base_next = 12'h112;
        text_next = 1'b0;
        text_color_next = 12'hBCD;
        character_next = 8'd32;
        glyph_col_next = 0;
        glyph_row_next = 0;
        grid_x = 0; grid_y = 0; panel_x = 0;
        cell_index = 0; cell_x = 0; cell_y = 0;
        in_grid = 1'b0; in_panel = 1'b0; right_panel = 1'b0;
        character_index = 0;
        if (scan_x >= 32 && scan_x < 288) begin
            in_panel = 1'b1;
            panel_x = scan_x-32;
        end else if (scan_x >= 352 && scan_x < 608) begin
            in_panel = 1'b1;
            right_panel = 1'b1;
            panel_x = scan_x-352;
        end
        panel_agent = right_panel ? view_agent4 : view_agent8;
        panel_trail = right_panel ? view_trail4 : view_trail8;
        panel_status = !view_run_valid ? 3'd0 : right_panel ? view_status4 : view_status8;
        panel_steps = !view_run_valid ? 12'b0 : right_panel ? view_steps_bcd4 : view_steps_bcd8;
        panel_cycles = !view_run_valid ? 32'b0 : right_panel ? view_cycles4 : view_cycles8;
        agent_color = right_panel ? 12'hC8F : 12'h4DF;
        trail_color = right_panel ? 12'h435 : 12'h245;

        if (scan_x >= 352 && scan_x < 608 && scan_y >= 16 && scan_y < 48)
            base_next = 12'h223;
        if (scan_x >= 32 && scan_x < 608 && scan_y == 56)
            base_next = 12'h345;

        if (scan_x >= 32 && scan_x < 160 && scan_y >= 24 && scan_y < 38) begin
            text_next=1; glyph_col_next=(scan_x-32)>>1; glyph_row_next=(scan_y-24)>>1;
            text_color_next=12'hEFF; character_next=text_char("MAZEBITS",8,(scan_x-32)>>4);
        end else if (scan_x >= 368 && scan_x < 448 && scan_y >= 24 && scan_y < 38) begin
            text_next=1; glyph_col_next=(scan_x-368)>>1; glyph_row_next=(scan_y-24)>>1;
            text_color_next=12'hFC5;
            case (view_mode)
                0: character_next=text_char("WALL ",5,(scan_x-368)>>4);
                1: character_next=text_char("START",5,(scan_x-368)>>4);
                2: character_next=text_char("GOAL ",5,(scan_x-368)>>4);
                default: character_next=text_char("VIEW ",5,(scan_x-368)>>4);
            endcase
        end else if (scan_x >= 32 && scan_x < 416 && scan_y >= 62 && scan_y < 69) begin
            text_next=1; glyph_col_next=scan_x-32; glyph_row_next=scan_y-62;
            text_color_next=view_rejected ? 12'hF86 : 12'h9AC;
            if (view_rejected) character_next=text_char("START AND GOAL CELLS STAY OPEN",30,(scan_x-32)>>3);
            else if (view_mode==3) character_next=text_char("CENTER STARTS OR RESTARTS BOTH POLICIES",39,(scan_x-32)>>3);
            else character_next=text_char("EDIT ONE MAP FOR BOTH PRECISIONS",32,(scan_x-32)>>3);
        end else if (in_panel && panel_x < 80 && scan_y >= 84 && scan_y < 98) begin
            text_next=1; glyph_col_next=panel_x>>1; glyph_row_next=(scan_y-84)>>1;
            text_color_next=agent_color;
            character_next=right_panel ? text_char("4 BIT",5,panel_x>>4) : text_char("8 BIT",5,panel_x>>4);
        end else if (in_panel && panel_x < 64 && scan_y >= 380 && scan_y < 387) begin
            text_next=1; glyph_col_next=panel_x; glyph_row_next=scan_y-380;
            text_color_next=(panel_status==2) ? 12'h3C8 :
                            (panel_status>=3) ? 12'hF86 : agent_color;
            character_next=text_char(status_text(panel_status),8,panel_x>>3);
        end else if (in_panel && panel_x < 72 && scan_y >= 396 && scan_y < 403) begin
            text_next=1; glyph_col_next=panel_x; glyph_row_next=scan_y-396;
            character_index=panel_x>>3;
            if (character_index<6) character_next=text_char("TRIES ",6,character_index);
            else if (character_index==6) character_next=hex_character(panel_steps[11:8]);
            else if (character_index==7) character_next=hex_character(panel_steps[7:4]);
            else character_next=hex_character(panel_steps[3:0]);
        end else if (in_panel && panel_x < 128 && scan_y >= 412 && scan_y < 419) begin
            text_next=1; glyph_col_next=panel_x; glyph_row_next=scan_y-412;
            character_index=panel_x>>3;
            if (character_index<8) character_next=text_char("CYC HEX ",8,character_index);
            else character_next=hex_character(panel_cycles >> ((15-character_index)*4));
        end else if (scan_x >= 32 && scan_x < 416 && scan_y >= 440 && scan_y < 447) begin
            text_next=1; glyph_col_next=scan_x-32; glyph_row_next=scan_y-440;
            text_color_next=12'hBCD;
            character_next=text_char("00 WALL  01 START  10 GOAL  11 VIEW + CENTER RUN",48,(scan_x-32)>>3);
        end else if (scan_x >= 32 && scan_x < 416 && scan_y >= 456 && scan_y < 463) begin
            text_next=1; glyph_col_next=scan_x-32; glyph_row_next=scan_y-456;
            text_color_next=12'h9AC;
            character_next=text_char("ARROWS MOVE  CENTER APPLIES  CPU RESET CLEARS",45,(scan_x-32)>>3);
        end

        if (in_panel && scan_y >= 112 && scan_y < 368) begin
            in_grid=1; grid_x=panel_x; grid_y=scan_y-112;
        end
        if (in_grid) begin
            cell_index={grid_y[7:5],grid_x[7:5]};
            cell_x=grid_x[4:0]; cell_y=grid_y[4:0];
            base_next=view_walls[cell_index] ? 12'h678 : 12'h223;
            if (view_run_valid && panel_trail[cell_index] && !view_walls[cell_index])
                base_next=trail_color;
            if (cell_index==view_start && cell_index==view_goal) base_next=12'hB7F;
            else if (cell_index==view_start) base_next=12'h3C8;
            else if (cell_index==view_goal) base_next=12'hF86;
            if (view_run_valid && cell_index==panel_agent &&
                cell_x>=5 && cell_x<=26 && cell_y>=5 && cell_y<=26)
                base_next=agent_color;
            if (cell_x==0 || cell_y==0) base_next=12'h345;
            if (view_mode!=3 && cell_index==view_cursor &&
                (cell_x<=2 || cell_x>=29 || cell_y<=2 || cell_y>=29)) base_next=12'hFC5;

            if ((cell_index==view_start || cell_index==view_goal ||
                 (view_run_valid && cell_index==panel_agent)) &&
                cell_x>=11 && cell_x<21 && cell_y>=9 && cell_y<23) begin
                text_next=1; text_color_next=12'h112;
                if (cell_index==view_start && cell_index==view_goal) character_next=8'd43;
                else if (cell_index==view_start) character_next=8'd83;
                else if (cell_index==view_goal) character_next=8'd71;
                else character_next=8'd65;
                glyph_col_next=(cell_x-11)>>1; glyph_row_next=(cell_y-9)>>1;
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
            view_run_valid <= 0;
            view_agent8 <= 0; view_agent4 <= 0;
            view_trail8 <= 0; view_trail4 <= 0;
            view_status8 <= 0; view_status4 <= 0;
            view_steps8 <= 0; view_steps4 <= 0;
            view_steps_bcd8 <= 0; view_steps_bcd4 <= 0;
            view_cycles8 <= 0; view_cycles4 <= 0;
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
                    view_run_valid <= run_valid;
                    view_agent8 <= agent8; view_agent4 <= agent4;
                    view_trail8 <= trail8; view_trail4 <= trail4;
                    view_status8 <= status8; view_status4 <= status4;
                    view_steps8 <= steps8; view_steps4 <= steps4;
                    view_steps_bcd8 <= decimal_digits(steps8);
                    view_steps_bcd4 <= decimal_digits(steps4);
                    view_cycles8 <= cycles8; view_cycles4 <= cycles4;
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
