`timescale 1ns/1ps
module tb_navigation_vga;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1;
    reg [63:0] walls=0;
    reg [5:0] start_cell=0,goal_cell=63,cursor=0;
    reg [1:0] mode=0;
    reg rejected=0;
    reg run_valid=0;
    reg [5:0] agent8=0,agent4=0;
    reg [63:0] trail8=0,trail4=0;
    reg [2:0] status8=0,status4=0;
    reg [6:0] steps8=0,steps4=0;
    reg [31:0] cycles8=0,cycles4=0;
    wire [3:0] red,green,blue;
    wire hsync,vsync,pixel_tick,active,frame_start;
    wire [9:0] pixel_x,pixel_y;
    maze_navigation_vga dut(
        .clk(clk),.rst(rst),.walls(walls),.start_cell(start_cell),
        .goal_cell(goal_cell),.cursor(cursor),.mode(mode),.rejected(rejected),
        .run_valid(run_valid),.agent8(agent8),.agent4(agent4),
        .trail8(trail8),.trail4(trail4),.status8(status8),.status4(status4),
        .steps8(steps8),.steps4(steps4),.cycles8(cycles8),.cycles4(cycles4),
        .red(red),.green(green),.blue(blue),.hsync(hsync),.vsync(vsync),
        .pixel_tick(pixel_tick),.pixel_x(pixel_x),.pixel_y(pixel_y),
        .active(active),.frame_start(frame_start)
    );
    integer px=0,py=0,frame=0,pixels=0,visible_pixels=0;
    integer horizontal_low=0,vertical_low=0,clocks_since_tick=0;
    integer file_baseline,file_edited;
    reg [34:0] held_outputs;
    reg [11:0] color;
    reg [11:0] expected_color;
    integer mirror;

    task expect_rgb(input integer x,input integer y,input [11:0] wanted);
        begin
            if (px==x && py==y && color!==wanted)
                $fatal(1,"pixel color frame=%0d x=%0d y=%0d got=%h wanted=%h",
                       frame,px,py,color,wanted);
        end
    endtask

    // Fixed fixture strings below are independent of the DUT formatting.
    // Check every visible font pixel and character-spacing background pixel.
    reg [127:0] expected_text;
    integer expected_length,text_origin_y,local_x,char_index;
    reg [7:0] expected_character;
    reg [34:0] expected_glyph;
    reg [11:0] expected_foreground;
    integer glyph_checks=0;
    function [34:0] golden_font;
        input [7:0] character;
        begin
            case (character)
                8'd48: golden_font = 35'b01110100011001110101110011000101110;
                8'd49: golden_font = 35'b00100011000010000100001000010001110;
                8'd50: golden_font = 35'b01110100010000100010001000100011111;
                8'd51: golden_font = 35'b11110000010000101110000010000111110;
                8'd52: golden_font = 35'b00010001100101010010111110001000010;
                8'd53: golden_font = 35'b11111100001000011110000010000111110;
                8'd54: golden_font = 35'b01110100001000011110100011000101110;
                8'd55: golden_font = 35'b11111000010001000100010000100001000;
                8'd56: golden_font = 35'b01110100011000101110100011000101110;
                8'd57: golden_font = 35'b01110100011000101111000010000101110;
                8'd65: golden_font = 35'b01110100011000111111100011000110001;
                8'd66: golden_font = 35'b11110100011000111110100011000111110;
                8'd67: golden_font = 35'b01111100001000010000100001000001111;
                8'd68: golden_font = 35'b11110100011000110001100011000111110;
                8'd69: golden_font = 35'b11111100001000011110100001000011111;
                8'd70: golden_font = 35'b11111100001000011110100001000010000;
                8'd71: golden_font = 35'b01111100001000010111100011000101111;
                8'd72: golden_font = 35'b10001100011000111111100011000110001;
                8'd73: golden_font = 35'b11111001000010000100001000010011111;
                8'd74: golden_font = 35'b00111000100001000010100101001001100;
                8'd75: golden_font = 35'b10001100101010011000101001001010001;
                8'd76: golden_font = 35'b10000100001000010000100001000011111;
                8'd77: golden_font = 35'b10001110111010110101100011000110001;
                8'd78: golden_font = 35'b10001110011010110011100011000110001;
                8'd79: golden_font = 35'b01110100011000110001100011000101110;
                8'd80: golden_font = 35'b11110100011000111110100001000010000;
                8'd81: golden_font = 35'b01110100011000110001101011001001101;
                8'd82: golden_font = 35'b11110100011000111110101001001010001;
                8'd83: golden_font = 35'b01111100001000001110000010000111110;
                8'd84: golden_font = 35'b11111001000010000100001000010000100;
                8'd85: golden_font = 35'b10001100011000110001100011000101110;
                8'd86: golden_font = 35'b10001100011000110001100010101000100;
                8'd87: golden_font = 35'b10001100011000110101101011010101010;
                8'd88: golden_font = 35'b10001100010101000100010101000110001;
                8'd89: golden_font = 35'b10001100010101000100001000010000100;
                8'd90: golden_font = 35'b11111000010001000100010001000011111;
                8'd58: golden_font = 35'b00000001000010000000001000010000000;
                8'd43: golden_font = 35'b00000001000010011111001000010000000;
                default: golden_font = 35'b0;
            endcase
        end
    endfunction


    task check_panel_text(input integer origin_x,input integer panel);
        begin
            expected_length=0; expected_text=0; text_origin_y=0;
            expected_foreground=12'hBCD;
            if(py>=380 && py<387) begin
                text_origin_y=380; expected_length=8;
                expected_foreground=panel ? 12'hC8F : 12'h4DF;
                if(frame==0) expected_text="READY   ";
                else if(frame==1 && panel==0) expected_text="RUNNING ";
                else if(frame==1 && panel==1) begin expected_text="BLOCKED "; expected_foreground=12'hF86; end
                else if(panel==0) begin expected_text="GOAL    "; expected_foreground=12'h3C8; end
                else begin expected_text="LOOP    "; expected_foreground=12'hF86; end
            end
            if(py>=396 && py<403) begin
                text_origin_y=396; expected_length=9;
                if(frame==0) expected_text="TRIES 000";
                else if(frame==1 && panel==0) expected_text="TRIES 003";
                else if(frame==1 && panel==1) expected_text="TRIES 004";
                else if(panel==0) expected_text="TRIES 127";
                else expected_text="TRIES 064";
            end
            if(py>=412 && py<419) begin
                text_origin_y=412; expected_length=16;
                if(frame==0) expected_text="CYC HEX 00000000";
                else if(frame==1 && panel==0) expected_text="CYC HEX 1234ABCD";
                else if(frame==1 && panel==1) expected_text="CYC HEX 89ABCDEF";
                else if(panel==0) expected_text="CYC HEX 89ABCDEF";
                else expected_text="CYC HEX 01234567";
            end
            local_x=px-origin_x;
            if(expected_length>0 && local_x>=0 && local_x<expected_length*8) begin
                char_index=local_x/8;
                expected_character=expected_text >> ((expected_length-1-char_index)*8);
                expected_glyph=golden_font(expected_character);
                expected_color=12'h112;
                if((local_x%8)<5 && expected_glyph[34-((py-text_origin_y)*5+local_x%8)])
                    expected_color=expected_foreground;
                if(color!==expected_color) $fatal(1,"text pixel frame%0d panel%0d x%0d y%0d char%0h got%h expected%h",frame,panel,px,py,expected_character,color,expected_color);
                glyph_checks=glyph_checks+1;
            end
        end
    endtask

    initial begin
        file_baseline=$fopen("navigation_running.ppm","w");
        file_edited=$fopen("navigation_result.ppm","w");
        if (!file_baseline || !file_edited) $fatal(1,"cannot create frame captures");
        $fwrite(file_baseline,"P3\n640 480\n255\n");
        $fwrite(file_edited,"P3\n640 480\n255\n");
        repeat(4) @(posedge clk);
        @(negedge clk); rst=0;
        held_outputs={red,green,blue,hsync,vsync,active,pixel_x,pixel_y};

        while(pixels<1260000) begin
            @(posedge clk); #1;
            clocks_since_tick=clocks_since_tick+1;
            if (pixel_tick!==0 && pixel_tick!==1) $fatal(1,"unknown pixel_tick");
            if (!pixel_tick) begin
                if ({red,green,blue,hsync,vsync,active,pixel_x,pixel_y}!==held_outputs)
                    $fatal(1,"outputs changed between pixel enables");
                if (frame_start!==0) $fatal(1,"frame_start must be a one-clock pulse");
            end else begin
                if(clocks_since_tick!=4) $fatal(1,"pixel interval is %0d clocks",clocks_since_tick);
                clocks_since_tick=0;
                if(pixel_x!==px[9:0] || pixel_y!==py[9:0])
                    $fatal(1,"raster order wanted%0d,%0d got%0d,%0d",px,py,pixel_x,pixel_y);
                if(active!==(px<640 && py<480)) $fatal(1,"active window mismatch");
                if(hsync!==!(px>=656 && px<752)) $fatal(1,"horizontal porch/sync mismatch");
                if(vsync!==!(py>=490 && py<492)) $fatal(1,"vertical porch/sync mismatch");
                if(frame_start!==(px==0 && py==0)) $fatal(1,"frame start mismatch");
                color={red,green,blue};
                if (^color===1'bx) $fatal(1,"unknown RGB");
                if(!active && color!==0) $fatal(1,"blanking RGB must be black");
                if(!hsync) horizontal_low=horizontal_low+1;
                if(!vsync) vertical_low=vertical_low+1;
                if(active) begin
                    visible_pixels=visible_pixels+1;
                    if(frame==1) $fwrite(file_baseline,"%0d %0d %0d\n",red*17,green*17,blue*17);
                    if(frame==2) $fwrite(file_edited,"%0d %0d %0d\n",red*17,green*17,blue*17);
                end

                // Distinct precision panels, trails, agents, preserved endpoints,
                // absent VIEW cursor, and wall priority, all from actual RGB pins.
                for(mirror=0;mirror<2;mirror=mirror+1) begin
                    expect_rgb(37+320*mirror,117,12'h3C8);
                    expect_rgb(261+320*mirror,341,(frame==0)?12'hF86:((frame==2 && mirror==0)?12'h245:12'h223));
                    expect_rgb(261+320*mirror,117,(frame==0)?12'h223:
                        ((frame==2 && mirror==0)?12'h4DF:12'hF86));
                    expect_rgb(259+320*mirror,115,(frame==0)?12'h223:12'hF86);
                    expect_rgb(69+320*mirror,181,(frame==0)?12'h223:12'h678);
                    expect_rgb(101+320*mirror,181,(frame==2)?12'h678:12'h223);
                    expect_rgb(64+320*mirror,144,12'h345);
                    expect_rgb(97+320*mirror,145,12'h223); // hidden cursor10 in VIEW
                    expect_rgb(33+320*mirror,113,(frame==0)?12'hFC5:12'h3C8);
                end
                expect_rgb(37,149,(frame==0)?12'h223:12'h245); // left trail8
                expect_rgb(357,149,(frame==0)?12'h223:(frame==1)?12'h435:12'hC8F);
                expect_rgb(37,213,(frame==0)?12'h223:(frame==1)?12'h4DF:12'h245);
                expect_rgb(357,181,(frame==0)?12'h223:(frame==1)?12'hC8F:12'h435);
                expect_rgb(357,213,12'h223); // right has never visited24
                expect_rgb(32,24,12'hEFF);
                expect_rgb(34,24,12'h112);
                expect_rgb(0,0,12'h112);
                check_panel_text(32,0);
                check_panel_text(352,1);
                if(active) begin
                    if(frame==0 && {dut.view_run_valid,dut.view_agent8,dut.view_agent4,dut.view_trail8,dut.view_trail4,dut.view_status8,dut.view_status4,dut.view_steps8,dut.view_steps4,dut.view_cycles8,dut.view_cycles4}!==0)
                        $fatal(1,"navigation state leaked into initial frame");
                    if(frame==1 && (dut.view_run_valid!==1 || dut.view_agent8!==24 || dut.view_agent4!==16 ||
                        dut.view_trail8!==64'h1010101 || dut.view_trail4!==64'h10101 || dut.view_status8!==1 || dut.view_status4!==3 ||
                        dut.view_steps8!==3 || dut.view_steps4!==4 || dut.view_cycles8!==32'h1234ABCD || dut.view_cycles4!==32'h89ABCDEF ||
                        dut.view_mode!==3 || dut.view_cursor!==10 || dut.view_rejected!==0))
                        $fatal(1,"navigation snapshot changed within frame1");
                    if(frame==2 && (dut.view_run_valid!==1 || dut.view_agent8!==7 || dut.view_agent4!==8 ||
                        dut.view_trail8!==64'hFF81818181818181 || dut.view_trail4!==64'h10101 || dut.view_status8!==2 || dut.view_status4!==4 ||
                        dut.view_steps8!==127 || dut.view_steps4!==64 || dut.view_cycles8!==32'h89ABCDEF || dut.view_cycles4!==32'h01234567 ||
                        dut.view_mode!==3 || dut.view_cursor!==18 || dut.view_rejected!==1))
                        $fatal(1,"navigation snapshot changed within frame2");
                end

                // Change all run inputs mid-frame; old full-frame data must hold.
                if(px==100 && py==150 && frame==0) begin
                    walls=64'h0002020202020202;
                    start_cell=0; goal_cell=7; cursor=10; mode=3; rejected=0;
                    run_valid=1; agent8=24; agent4=16;
                    trail8=64'h1010101; trail4=64'h10101;
                    status8=1; status4=3; steps8=3; steps4=4;
                    cycles8=32'h1234ABCD; cycles4=32'h89ABCDEF;
                end
                if(px==100 && py==150 && frame==1) begin
                    walls=64'h0002020202060202;
                    cursor=18; mode=3; rejected=1;
                    run_valid=1; agent8=7; agent4=8;
                    trail8=64'hFF81818181818181; trail4=64'h10101;
                    status8=2; status4=4; steps8=127; steps4=64;
                    cycles8=32'h89ABCDEF; cycles4=32'h01234567;
                end

                pixels=pixels+1;
                held_outputs={red,green,blue,hsync,vsync,active,pixel_x,pixel_y};
                if(px==799) begin
                    px=0;
                    if(py==524) begin
                        if(horizontal_low!=50400 || vertical_low!=1600)
                            $fatal(1,"sync width totals HS%0d VS%0d",horizontal_low,vertical_low);
                        horizontal_low=0; vertical_low=0;
                        py=0; frame=frame+1;
                    end else py=py+1;
                end else px=px+1;
            end
        end
        if(visible_pixels!=921600 || frame!=3)
            $fatal(1,"incomplete frame coverage");
        $fclose(file_baseline); $fclose(file_edited);

        // Reset during raster activity restores black pins and starts a new
        // complete pixel period; the renderer's initial map is deterministic.
        @(negedge clk); rst=1;
        @(posedge clk); #1;
        if({red,green,blue}!==0 || hsync!==1 || vsync!==1 || active!==0 ||
           pixel_tick!==0 || frame_start!==0 || pixel_x!==0 || pixel_y!==0)
            $fatal(1,"reset output mismatch");
        @(negedge clk); rst=0;
        repeat(3) begin @(posedge clk); #1; if(pixel_tick!==0) $fatal(1,"early tick after reset"); end
        @(posedge clk); #1;
        if(pixel_tick!==1 || frame_start!==1 || pixel_x!==0 || pixel_y!==0 ||
           {red,green,blue}!==12'h112 || dut.view_walls!==0 ||
           dut.view_start!==0 || dut.view_goal!==63 || dut.view_cursor!==0 ||
           dut.view_run_valid!==0 || dut.view_trail8!==0 || dut.view_trail4!==0 ||
           dut.view_steps8!==0 || dut.view_steps4!==0 || dut.view_cycles8!==0 || dut.view_cycles4!==0)
            $fatal(1,"reset raster restart mismatch");
        $display("PASS_NAVIGATION_VGA frames=3 pixels=1260000 active_pixels=921600 glyph_pixels=%0d snapshots=2",glyph_checks);
        $finish;
    end
    initial begin
        #60000000;
        $fatal(1,"VGA simulation watchdog");
    end
endmodule
