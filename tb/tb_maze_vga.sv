`timescale 1ns/1ps
module tb_maze_vga;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1;
    reg [63:0] walls=0;
    reg [5:0] start_cell=0,goal_cell=63,cursor=0;
    reg [1:0] mode=0;
    reg rejected=0;
    wire [3:0] red,green,blue;
    wire hsync,vsync,pixel_tick,active,frame_start;
    wire [9:0] pixel_x,pixel_y;
    maze_vga dut(
        .clk(clk),.rst(rst),.walls(walls),.start_cell(start_cell),
        .goal_cell(goal_cell),.cursor(cursor),.mode(mode),.rejected(rejected),
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

    initial begin
        file_baseline=$fopen("baseline_frame.ppm","w");
        file_edited=$fopen("edited_frame.ppm","w");
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

                // Each panel must show the same maze, including all endpoint,
                // wall, cursor, and grid colors. Samples avoid glyph interiors.
                for(mirror=0;mirror<2;mirror=mirror+1) begin
                    expect_rgb(37+320*mirror,117,(frame<2)?12'h3C8:12'h223); // start0
                    expect_rgb(261+320*mirror,341,(frame==0)?12'hF86:12'h223); // oldgoal63
                    expect_rgb(261+320*mirror,117,(frame==1)?12'hF86:12'h223); // goal7
                    expect_rgb(69+320*mirror,181,(frame==0)?12'h223:12'h678); // detourwall17
                    expect_rgb(101+320*mirror,181,(frame==2)?12'h678:12'h223); // editedwall18
                    expect_rgb(133+320*mirror,213,(frame==2)?12'hB7F:12'h223); // overlap27
                    expect_rgb(64+320*mirror,144,12'h345); // cellgridline9
                    if(frame==0) expect_rgb(33+320*mirror,113,12'hFC5);
                    if(frame==1) expect_rgb(97+320*mirror,145,12'hFC5);
                    if(frame==2) expect_rgb(97+320*mirror,177,12'hFC5);
                end
                expect_rgb(32,24,12'hEFF); // M glyph first foreground pixel
                expect_rgb(34,24,12'h112); // M glyph background
                expect_rgb(0,0,12'h112);
                expect_rgb(32,394,12'h3C8);
                expect_rgb(192,394,12'hF86);
                expect_rgb(352,394,12'hFC5);

                // Live edits made halfway down a frame must not affect that
                // frame; they become visible only after the blanking snapshot.
                if(px==100 && py==150 && frame==0) begin
                    walls=64'h0002020202020202;
                    start_cell=0; goal_cell=7; cursor=10; mode=0; rejected=0;
                end
                if(px==100 && py==150 && frame==1) begin
                    walls=64'h0002020202060202;
                    start_cell=27; goal_cell=27; cursor=18; mode=2; rejected=1;
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
           dut.view_start!==0 || dut.view_goal!==63 || dut.view_cursor!==0)
            $fatal(1,"reset raster restart mismatch");
        $display("PASS_VGA frames=3 pixels=1260000 active_pixels=921600");
        $finish;
    end
    initial begin
        #60000000;
        $fatal(1,"VGA simulation watchdog");
    end
endmodule
