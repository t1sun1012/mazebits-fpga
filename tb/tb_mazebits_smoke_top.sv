`timescale 1ns/1ps
// This test isolates the board controls with a fixed-latency stub. CNN numerical
// correctness is tested separately against the frozen Step 3 vectors.
module tb_mazebits_smoke_top;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, button = 0;
    reg [0:0] switches = 0;
    wire [7:0] leds;
    mazebits_smoke_top #(.DEBOUNCE_CYCLES(4)) dut (
        .CLK100MHZ(clk), .CPU_RESETN(resetn), .BTNC(button),
        .SW(switches), .LED(leds)
    );
    `include "quantization.vh"
    integer total_starts = 0;
    always @(posedge clk) if (!dut.reset && dut.start) total_starts <= total_starts + 1;
    task tick;
        input integer count;
        integer i;
        begin for (i=0;i<count;i=i+1) begin @(posedge clk); #1; end end
    endtask
    task wait_finished;
        integer n;
        begin
            n=0;
            while (!leds[3] && n<300) begin tick(1); n=n+1; end
            if (!leds[3] || leds[2]) $fatal(1,"board did not finish");
        end
    endtask
    initial begin
        tick(4);
        resetn=1;
        tick(10);
        if (leds !== ((W_BITS == 4) ? 8'h20 : 8'h00)) $fatal(1,"bad reset/precision LEDs");
        // Bounces shorter than the stable threshold must not create a request.
        button=1; tick(2); button=0; tick(2);
        button=1; tick(1); button=0; tick(12);
        if (total_starts != 0) $fatal(1,"bounce triggered inference");

        switches=1; tick(4);
        button=1; tick(12);
        if (!leds[2] || !leds[4] || total_starts != 1) $fatal(1,"first request not captured");
        if (dut.cnn.captured_walls !== 64'h0002020202020202 ||
            dut.cnn.captured_goal !== 6'd7) $fatal(1,"wrong detour input");
        switches=0; tick(5);
        if (dut.cnn.captured_walls !== 64'h0002020202020202 ||
            !leds[4]) $fatal(1,"switch changed active request");
        // Release and press again while busy. The second press is ignored,
        // including after inference completes while the button remains held.
        button=0; tick(12); button=1; tick(12);
        if (total_starts != 1) $fatal(1,"busy request accepted");
        wait_finished();
        tick(20);
        if (total_starts != 1 || leds[1:0] !== 2'd2) $fatal(1,"held button retriggered or result wrong");

        button=0; tick(12); button=1; tick(12);
        if (total_starts != 2 || leds[4] || !leds[2] || leds[3]) $fatal(1,"second request/control state wrong");
        if (dut.cnn.captured_walls !== 64'd0 ||
            dut.cnn.captured_goal !== 6'd63) $fatal(1,"wrong open input");
        // Reset aborts the active request and clears completion/action.
        resetn=0; button=0; tick(5);
        if (leds[4:0] !== 5'd0) $fatal(1,"reset failed to abort/clear");
        resetn=1; tick(12);
        button=1; tick(12);
        if (total_starts != 3) $fatal(1,"restart after reset failed");
        wait_finished();
        tick(20);
        if (leds[1:0] !== 2'd1 || total_starts != 3) $fatal(1,"open result or held-button state wrong");
        $display("PASS_BOARD bits=%0d requests=%0d",W_BITS,total_starts);
        $finish;
    end
    initial begin #100000; $fatal(1,"board controls watchdog"); end
endmodule

module mazebits_cnn #(
    parameter WEIGHTS_FILE="weights.mem",
    parameter BIASES_FILE="biases.mem"
) (
    input wire clk,rst,start,
    input wire [63:0] walls,
    input wire [5:0] agent,goal,
    output reg busy=0,done=0,
    output reg [1:0] action=0,
    output wire signed [31:0] score_up,score_right,score_down,score_left,
    output wire [31:0] cycles,
    output wire trace_valid,
    output wire [3:0] trace_layer,
    output wire [9:0] trace_index,
    output wire signed [31:0] trace_acc,
    output wire [7:0] trace_act
);
    reg [63:0] captured_walls;
    reg [5:0] captured_goal;
    integer remaining;
    assign score_up=0, score_right=0, score_down=0, score_left=0;
    assign cycles=0, trace_valid=0, trace_layer=0, trace_index=0, trace_acc=0, trace_act=0;
    always @(posedge clk) begin
        done<=0;
        if (rst) begin
            busy<=0; done<=0; action<=0; remaining<=0;
            captured_walls<=0; captured_goal<=0;
        end else if (start && !busy) begin
            if (agent !== 6'd0) $fatal(1,"wrong smoke fixture agent");
            busy<=1; remaining<=150; captured_walls<=walls; captured_goal<=goal;
        end else if (busy) begin
            if (remaining == 0) begin
                busy<=0; done<=1; action<=(captured_goal == 7) ? 2'd2 : 2'd1;
            end else remaining<=remaining-1;
        end
    end
endmodule
