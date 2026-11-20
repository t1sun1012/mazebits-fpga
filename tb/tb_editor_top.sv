`timescale 1ns/1ps
module tb_editor_top;
    reg clk=0;
    always #5 clk=~clk;
    reg reset_n=0;
    reg [4:0] buttons=0;
    reg [1:0] switches=0;
    wire [15:0] led;
    wire [3:0] red,green,blue;
    wire hs,vs;
    mazebits_editor_top #(.DEBOUNCE_CYCLES(4),.ERROR_HOLD_CYCLES(20)) dut (
        .CLK100MHZ(clk),.CPU_RESETN(reset_n),
        .BTNU(buttons[0]),.BTNR(buttons[1]),.BTND(buttons[2]),
        .BTNL(buttons[3]),.BTNC(buttons[4]),.SW(switches),.LED(led),
        .VGA_R(red),.VGA_G(green),.VGA_B(blue),.VGA_HS(hs),.VGA_VS(vs)
    );
    task ticks(input integer count);
        integer i;
        begin
            for(i=0;i<count;i=i+1) begin @(posedge clk); #1; end
        end
    endtask
    task tap(input [4:0] mask);
        begin
            @(negedge clk); buttons=mask;
            ticks(12);
            @(negedge clk); buttons=0;
            ticks(12);
        end
    endtask
    task set_mode(input [1:0] requested);
        begin
            @(negedge clk); switches=requested;
            ticks(14);
            if(led[1:0] !== requested) $fatal(1,"mode did not settle");
        end
    endtask
    initial begin
        ticks(8);
        @(negedge clk); reset_n=1;
        ticks(14);
        if(dut.walls !== 64'd0 || dut.start_cell !== 0 || dut.goal_cell !== 63 ||
           dut.cursor !== 0 || led[8] !== 1 || led[10] !== 0)
            $fatal(1,"top power-up/reset state");
        // Raw contact noise must not move the cursor.
        @(negedge clk); buttons=5'b00010; ticks(1);
        @(negedge clk); buttons=0; ticks(1);
        @(negedge clk); buttons=5'b00010; ticks(1);
        @(negedge clk); buttons=0; ticks(12);
        if(dut.cursor !== 0) $fatal(1,"top button bounce moved cursor");
        tap(5'b00010);
        if(dut.cursor !== 1 || led[7:2] !== 6'd1) $fatal(1,"right button/cursor LEDs");
        tap(5'b10000);
        if(dut.walls !== 64'd2 || led[10] !== 1) $fatal(1,"wall edit/modified LED");
        // Hold right: exactly one movement, even after many debounce intervals.
        @(negedge clk); buttons=5'b00010; ticks(40);
        if(dut.cursor !== 2) $fatal(1,"held button repeated");
        @(negedge clk); buttons=0; ticks(12);
        tap(5'b01000);
        // Mode bus accepts only a stable complete candidate.
        @(negedge clk); switches=1; ticks(2);
        @(negedge clk); switches=2; ticks(2);
        @(negedge clk); switches=0; ticks(2);
        if(led[1:0] !== 0) $fatal(1,"mode chatter accepted");
        set_mode(1); tap(5'b10000);
        if(dut.start_cell !== 1 || dut.walls !== 0) $fatal(1,"start placement did not clear wall");
        set_mode(2); tap(5'b10000);
        if(dut.goal_cell !== 1 || led[11] !== 1) $fatal(1,"overlapping endpoints indicator");
        set_mode(0);
        @(negedge clk); buttons=5'b10000; ticks(10);
        if(led[9] !== 1 || dut.walls !== 0) $fatal(1,"protected endpoint rejection feedback");
        @(negedge clk); buttons=0; ticks(40);
        if(led[9] !== 0) $fatal(1,"rejection feedback did not clear");
        set_mode(3);
        tap(5'b10100);
        if(dut.cursor !== 1 || dut.walls !== 0 || led[8] !== 0)
            $fatal(1,"view mode mutated map/cursor");
        // Reset clears the edited state even while a button is held.
        @(negedge clk); reset_n=0; buttons=5'b00010;
        ticks(8);
        if(dut.cursor !== 0 || dut.start_cell !== 0 || dut.goal_cell !== 63 ||
           dut.walls !== 0 || led[10] !== 0)
            $fatal(1,"reset did not restore editor");
        @(negedge clk); buttons=0; switches=0; reset_n=1;
        ticks(14);
        if(^{red,green,blue,hs,vs} === 1'bx) $fatal(1,"unknown physical VGA outputs");
        $display("PASS_EDITOR_TOP reset=1 bounce=1 held_press=1 mode_bus=1 edit=1 endpoint_safety=1 view=1 leds=1");
        $finish;
    end
    initial begin
        #1000000;
        $fatal(1,"top test watchdog");
    end
endmodule
