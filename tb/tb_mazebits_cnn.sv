// Golden traces come from the frozen Step 3 integer/scalar reference.
`timescale 1ns/1ps

module tb_mazebits_cnn;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, start = 0;
    reg [63:0] walls = 0;
    reg [5:0] agent = 0, goal = 63;
    wire busy, done, trace_valid;
    wire [1:0] action;
    wire signed [31:0] score_up, score_right, score_down, score_left;
    wire [31:0] cycles;
    wire [3:0] trace_layer;
    wire [9:0] trace_index;
    wire signed [31:0] trace_acc;
    wire [7:0] trace_act;

    mazebits_cnn dut (
        .clk(clk), .rst(rst), .start(start), .walls(walls),
        .agent(agent), .goal(goal), .busy(busy), .done(done),
        .action(action), .score_up(score_up), .score_right(score_right),
        .score_down(score_down), .score_left(score_left), .cycles(cycles),
        .trace_valid(trace_valid), .trace_layer(trace_layer),
        .trace_index(trace_index), .trace_acc(trace_acc), .trace_act(trace_act)
    );

    reg inputs [0:1535];
    reg [31:0] expected_acc [0:67583]; // 8 * (8*1024 + 256)
    reg [7:0] expected_act [0:65535]; // 8 * 8*1024
    reg [31:0] expected_scores [0:31];
    reg [1:0] expected_actions [0:7];
    integer case_count = 8;
    integer c, i, elapsed, traced, layer_wanted, index_wanted;
    integer count_agent, count_goal, compared = 0, repeats = 0;
    integer cycle_reference = -1;
    integer watchdog = 4000000;
    reg [127:0] held_scores;
    reg [1:0] held_action;
    reg [31:0] held_cycles;

    task load_case(input integer n);
        integer p;
        begin
            count_agent = 0;
            count_goal = 0;
            for (p=0; p<64; p=p+1) begin
                walls[p] = inputs[n*192+p];
                if (inputs[n*192+64+p] === 1'b1) begin
                    agent = p;
                    count_agent = count_agent+1;
                end
                if (inputs[n*192+128+p] === 1'b1) begin
                    goal = p;
                    count_goal = count_goal+1;
                end
            end
            if (count_agent != 1 || count_goal != 1)
                $fatal(1, "invalid one-hot case %0d", n);
        end
    endtask

    task run_case(input integer n, input integer repeated);
        begin
            @(negedge clk);
            load_case(n);
            held_scores = {score_up,score_right,score_down,score_left};
            held_action = action;
            start = 1;
            @(posedge clk); #1;
            if (busy !== 1'b1 || done !== 1'b0 || cycles !== 0)
                $fatal(1, "start handshake case %0d", n);
            @(negedge clk);
            start = 0;
            elapsed = 0;
            traced = 0;
            while (done !== 1'b1) begin
                @(posedge clk); #1;
                elapsed = elapsed+1;
                if (elapsed > watchdog)
                    $fatal(1, "watchdog case %0d", n);
                if (done !== 1'b0 && done !== 1'b1)
                    $fatal(1, "unknown done case %0d", n);
                if (trace_valid !== 1'b0 && trace_valid !== 1'b1)
                    $fatal(1, "unknown trace_valid case %0d", n);
                if (done === 1'b0 && busy !== 1'b1)
                    $fatal(1, "busy dropped before done case %0d", n);
                if (done === 1'b0 &&
                    ({score_up,score_right,score_down,score_left} !== held_scores ||
                     action !== held_action))
                    $fatal(1, "result changed before completion");
                if (trace_valid === 1'b1) begin
                    if (traced >= 8448)
                        $fatal(1, "extra trace case %0d", n);
                    layer_wanted = (traced < 8192) ? traced/1024 : 8;
                    index_wanted = (traced < 8192) ? traced%1024 : traced-8192;
                    if (trace_layer !== layer_wanted[3:0] ||
                        trace_index !== index_wanted[9:0])
                        $fatal(1, "trace ordering case=%0d trace=%0d layer=%0d index=%0d",
                               n, traced, trace_layer, trace_index);
                    if (trace_acc !== expected_acc[n*8448+traced])
                        $fatal(1, "acc mismatch case=%0d layer=%0d index=%0d got=%0d expected=%0d",
                               n, layer_wanted, index_wanted, trace_acc,
                               $signed(expected_acc[n*8448+traced]));
                    if (traced < 8192) begin
                        if (trace_act !== expected_act[n*8192+traced])
                            $fatal(1, "act mismatch case=%0d layer=%0d index=%0d got=%0d expected=%0d",
                                   n, layer_wanted, index_wanted, trace_act,
                                   expected_act[n*8192+traced]);
                    end else if (trace_act !== 0)
                        $fatal(1, "head trace_act must be zero");
                    traced = traced+1;
                end
                // Perturb live inputs and assert start while busy: snapshots must win.
                if (elapsed == 4) begin
                    walls = ~walls;
                    agent = ~agent;
                    goal = ~goal;
                    start = 1;
                end
                if (elapsed == 9)
                    start = 0;
                // A request on the final busy edge is ignored too.
                if (cycle_reference > 0 && elapsed == cycle_reference-1)
                    start = 1;
            end
            start = 0;
            if (busy !== 0 || traced != 8448)
                $fatal(1, "completion/traces case=%0d busy=%b traces=%0d", n, busy, traced);
            if (cycles !== elapsed)
                $fatal(1, "cycle accounting got=%0d observed=%0d", cycles, elapsed);
            if (cycle_reference < 0)
                cycle_reference = elapsed;
            else if (elapsed != cycle_reference)
                $fatal(1, "latency changed between inputs");
            if (score_up !== expected_scores[n*4] ||
                score_right !== expected_scores[n*4+1] ||
                score_down !== expected_scores[n*4+2] ||
                score_left !== expected_scores[n*4+3] ||
                action !== expected_actions[n])
                $fatal(1, "final score/action mismatch case %0d", n);
            if (repeated)
                repeats = repeats+1;
            else
                compared = compared+16645;
            $display("RTL_CASE case=%0d repeated=%0d cycles=%0d traces=%0d words=16645",
                     n, repeated, elapsed, traced);
            held_scores = {score_up,score_right,score_down,score_left};
            held_action = action;
            held_cycles = cycles;
            repeat (3) begin
                @(posedge clk); #1;
                if (done !== 0 || busy !== 0 || trace_valid !== 0 ||
                    {score_up,score_right,score_down,score_left} !== held_scores ||
                    action !== held_action || cycles !== held_cycles)
                    $fatal(1, "idle retention or one-cycle pulse failure");
            end
        end
    endtask

    initial begin
        // White-box arithmetic checks supplement the real-model fixtures.
        if (dut.round_relu(-32'sd1, 1) !== 0 ||
            dut.round_relu(32'sh80000000, -1) !== 0 ||
            dut.round_relu(32'sd0, 0) !== 0 ||
            dut.round_relu(32'sd1, 1) !== 1 ||
            dut.round_relu(32'sd3, 1) !== 2 ||
            dut.round_relu(32'sd5, 2) !== 1 ||
            dut.round_relu(32'sd6, 2) !== 2 ||
            dut.round_relu(32'sd7, 2) !== 2 ||
            dut.round_relu(32'sd1, -1) !== 2 ||
            dut.round_relu(32'sd127, -1) !== ((dut.A_BITS == 8) ? 254 : 15) ||
            dut.round_relu(32'sd128, -1) !== ((1 << dut.A_BITS)-1) ||
            dut.round_relu(32'sh7fffffff, 0) !== ((1 << dut.A_BITS)-1) ||
            dut.round_relu(32'sh7fffffff, -31) !== ((1 << dut.A_BITS)-1))
            $fatal(1, "rounding/ReLU/saturation boundary failure");
        force dut.pending_up = -32'sd3;
        force dut.pending_right = -32'sd3;
        force dut.pending_down = -32'sd3;
        force dut.pending_left = -32'sd3;
        #1;
        if (dut.best_action !== 0) $fatal(1, "all-equal argmax tie");
        force dut.pending_right = -32'sd1;
        force dut.pending_down = -32'sd1;
        #1;
        if (dut.best_action !== 1) $fatal(1, "negative signed argmax/tie");
        force dut.pending_left = 32'sd0;
        #1;
        if (dut.best_action !== 3) $fatal(1, "signed argmax zero versus negative");
        release dut.pending_up;
        release dut.pending_right;
        release dut.pending_down;
        release dut.pending_left;
        $display("RTL_ARITHMETIC_PASS rounding_cases=13 argmax_cases=3");
        if ($value$plusargs("CASES=%d", case_count)) begin end
        if ($test$plusargs("SMOKE")) case_count = 1;
        if ($value$plusargs("WATCHDOG=%d", watchdog)) begin end
        if (case_count < 1 || case_count > 8)
            $fatal(1, "CASES must be 1..8");
        $readmemh("inputs.mem", inputs);
        $readmemh("expected_acc.mem", expected_acc);
        $readmemh("expected_act.mem", expected_act);
        $readmemh("scores.mem", expected_scores);
        $readmemh("actions.mem", expected_actions);
        repeat (3) @(posedge clk);
        @(negedge clk); rst = 0;

        // Abort an active transaction and ensure the next accepted start is clean.
        load_case(0); start = 1;
        @(posedge clk); #1;
        if (busy !== 1) $fatal(1, "abort fixture did not start");
        @(negedge clk); start = 0;
        repeat (100) @(posedge clk);
        @(negedge clk); rst = 1; start = 1;
        @(posedge clk); #1;
        if (busy !== 0 || done !== 0 || trace_valid !== 0 || cycles !== 0 ||
            {score_up,score_right,score_down,score_left} !== 0 || action !== 0)
            $fatal(1, "reset did not abort/clear transaction");
        @(negedge clk); rst = 0; start = 0;
        for (c=0; c<case_count; c=c+1)
            run_case(c, 0);
        run_case(0, 1);
        $display("RTL_PASS cases=%0d words=%0d repeat_cases=%0d cycles=%0d reset_abort=1 busy_start_ignored=1",
                 case_count, compared, repeats, cycle_reference);
        $finish;
    end
endmodule
