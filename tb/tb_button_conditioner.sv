`timescale 1ns / 1ps
module tb_button_conditioner;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1;
    reg [4:0] raw_buttons = 0;
    wire [4:0] stable_buttons, press;
    button_conditioner #(.WIDTH(5),.DEBOUNCE_CYCLES(4)) dut(.*);
    wire [0:0] fast_stable, fast_press;
    button_conditioner #(.WIDTH(1),.DEBOUNCE_CYCLES(1)) fast (
        .clk(clk),.rst(rst),.raw_buttons(raw_buttons[0]),
        .stable_buttons(fast_stable),.press(fast_press)
    );
    integer pulses [0:4];
    integer fast_pulses = 0, checks=0, i;
    reg [4:0] previous_press=0;

    always @(posedge clk) begin
        #1;
        if (!rst) begin
            if ((press & previous_press) != 0)
                $fatal(1,"press must be one cycle");
            previous_press = press;
            for(i=0;i<5;i=i+1)
                if (press[i] === 1'b1) pulses[i] = pulses[i]+1;
            if (fast_press[0]) fast_pulses=fast_pulses+1;
        end else previous_press=0;
    end

    task sample(input reg [4:0] raw, input integer clocks);
        begin
            @(negedge clk); raw_buttons = raw;
            repeat(clocks) begin @(posedge clk); #2; end
        end
    endtask

    task check(input reg [4:0] value, input integer p0,p1,p2,p3,p4);
        begin
            checks=checks+1;
            if (stable_buttons !== value || pulses[0]!=p0 || pulses[1]!=p1 ||
                pulses[2]!=p2 || pulses[3]!=p3 || pulses[4]!=p4)
                $fatal(1,"debounce check=%0d stable=%b expected=%b pulses=%0d,%0d,%0d,%0d,%0d",
                       checks, stable_buttons,value,pulses[0],pulses[1],pulses[2],pulses[3],pulses[4]);
        end
    endtask

    initial begin
        for(i=0;i<5;i=i+1) pulses[i]=0;
        repeat(2) @(posedge clk);
        #2; check(0,0,0,0,0,0);
        @(negedge clk); rst=0;
        // Repeated one-clock chatter is shorter than the four stable samples.
        sample(1,1); sample(0,1); sample(1,1); sample(0,8);
        check(0,0,0,0,0,0);
        // Exact press threshold: 2 synchronizer cycles plus 4 stable samples.
        sample(1,5); check(0,0,0,0,0,0);
        @(posedge clk); #2; check(1,1,0,0,0,0);
        if (fast_stable !== 1 || fast_pulses<1)
            $fatal(1,"DEBOUNCE_CYCLES=1 did not work");
        sample(1,20); check(1,1,0,0,0,0);
        // Release chatter cannot rearm the held button.
        sample(0,2); sample(1,8); check(1,1,0,0,0,0);
        sample(0,6); check(0,1,0,0,0,0);
        sample(1,6); check(1,2,0,0,0,0);
        // Other channels debounce independently and simultaneous presses survive.
        sample(5'b11111,6); check(5'b11111,2,1,1,1,1);
        sample(5'b11111,12); check(5'b11111,2,1,1,1,1);
        sample(0,6); check(0,2,1,1,1,1);
        sample(5'b10100,6); check(5'b10100,2,1,2,1,2);
        // Synchronous reset clears a partial or held press without stale pulses.
        @(negedge clk); rst=1; raw_buttons=0;
        @(posedge clk); #2;
        if (stable_buttons !== 0 || press !== 0 || fast_stable !== 0 || fast_press !== 0)
            $fatal(1,"reset did not clear conditioner");
        @(negedge clk); rst=0;
        sample(0,10); check(0,2,1,2,1,2);
        sample(5'b01010,6); check(5'b01010,2,2,2,2,2);
        $display("PASS_BUTTONS checks=%0d width=5 thresholds=1,4",checks);
        $finish;
    end
endmodule
