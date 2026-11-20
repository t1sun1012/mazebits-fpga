`timescale 1ns/1ps
module tb_maze_navigation;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1,start=0;
    reg [63:0] walls=0;
    reg [5:0] start_cell=0,goal_cell=63;
    reg cnn_busy=0,cnn_done=0;
    reg [1:0] cnn_action=0;
    reg [31:0] cnn_cycles=0;
    wire cnn_start,cnn_reset,step_event;
    wire [63:0] cnn_walls,trail;
    wire [5:0] cnn_agent,cnn_goal,agent_cell;
    wire [2:0] status;
    wire [6:0] steps;
    wire [31:0] total_cycles;
    integer checks=0, i, begin_tick, tick_count=0;
    maze_navigation #(.MAX_STEPS(3),.STEP_DELAY_CYCLES(4)) dut (.*);
    always @(posedge clk) tick_count=tick_count+1;
    task ticks(input integer n);
        integer k;
        begin for(k=0;k<n;k=k+1) begin @(posedge clk); #1; end end
    endtask
    task check(input bit ok,input string why);
        begin checks=checks+1; if(!ok) $fatal(1,"%s",why); end
    endtask
    task scenario(input [63:0] w,input [5:0] s,input [5:0] g);
        begin
            @(negedge clk); walls=w;start_cell=s;goal_cell=g;
            cnn_busy=0;cnn_done=0;start=1;
            #1; check(cnn_reset===1,"start must reset core immediately");
            ticks(1);
            check(cnn_walls===w && cnn_goal===g && agent_cell===s &&
                  trail===(64'b1<<s) && steps===0 && total_cycles===0,"scenario snapshot");
            @(negedge clk); start=0;
        end
    endtask
    task request;
        integer guard;
        begin
            guard=0;
            while(cnn_start!==1 && guard<30) begin ticks(1);guard=guard+1;end
            check(cnn_start===1 && status===1,"missing inference request");
            check(cnn_agent===agent_cell && cnn_reset===0,"request agent/reset");
            @(negedge clk);cnn_busy=1;
            ticks(2);
            check(cnn_start===0,"cnn_start must be one cycle");
        end
    endtask
    task response(input [1:0] a,input [31:0] cycles);
        begin
            @(negedge clk);cnn_action=a;cnn_cycles=cycles;cnn_busy=0;cnn_done=1;
            ticks(1);check(step_event===1,"result did not create step_event");
            @(negedge clk);cnn_done=0;
            ticks(1);check(step_event===0,"step_event did not pulse");
        end
    endtask
    task action(input [1:0] a,input [31:0] cycles);
        begin request();response(a,cycles);end
    endtask
    task terminal(input [2:0] code,input [5:0] pos,input [6:0] attempts,input [31:0] cycles);
        reg [63:0] saved_trail;
        begin
            check(status===code && agent_cell===pos && steps===attempts && total_cycles===cycles,"terminal result mismatch");
            saved_trail=trail;
            @(negedge clk);cnn_done=1;cnn_cycles=999;cnn_action=2;
            ticks(12);
            check(status===code && agent_cell===pos && steps===attempts && total_cycles===cycles &&
                  trail===saved_trail && cnn_start===0 && step_event===0,"terminal state accepted stale done");
            @(negedge clk);cnn_done=0;
        end
    endtask
    initial begin
        ticks(2);check(status===0 && steps===0 && trail===0 && cnn_reset===1,"reset values");
        @(negedge clk);rst=0;
        // All four geometric boundaries must reject without six-bit wraparound.
        scenario(0,0,63);action(0,11);terminal(3,0,1,11);
        scenario(0,7,63);action(1,12);terminal(3,7,1,12);
        scenario(0,63,0);action(2,13);terminal(3,63,1,13);
        scenario(0,8,63);action(3,14);terminal(3,8,1,14);
        // A wall blocks an otherwise legal neighbor; no policy repair.
        scenario(64'h400,9,63);action(1,21);terminal(3,9,1,21);
        check(trail===64'h200,"blocked action changed trail");
        // All four legal moves and goal success are tested independently.
        scenario(0,9,1);action(0,31);terminal(2,1,1,31);
        scenario(0,9,10);action(1,32);terminal(2,10,1,32);
        scenario(0,9,17);action(2,33);terminal(2,17,1,33);
        scenario(0,9,8);action(3,34);terminal(2,8,1,34);
        // No inference at a zero-length success or invalid endpoint.
        scenario(0,27,27);terminal(2,27,0,0);
        scenario(64'h1,0,63);terminal(3,0,0,0);
        scenario(64'h8000000000000000,0,63);terminal(3,0,0,0);
        // Snapshot external inputs once, then ignore all subsequent edits.
        scenario(0,9,63);request();
        @(negedge clk);walls=~64'd0;start_cell=62;goal_cell=10;
        response(1,42);
        check(status===1 && agent_cell===10 && cnn_walls===0 && cnn_goal===63,"live inputs leaked into episode");
        action(3,43);terminal(4,9,2,85);
        check(trail===(64'h200|64'h400),"loop trail wrong");
        // Pacing delays only requests; compute-cycle totals exclude idle time.
        scenario(0,0,63);request();
        @(negedge clk);cnn_action=1;cnn_cycles=100;cnn_busy=0;cnn_done=1;
        ticks(1);begin_tick=tick_count;
        @(negedge clk);cnn_done=0;
        repeat(4) begin ticks(1);check(cnn_start===0,"next request before pacing delay");end
        ticks(1);check(cnn_start===1 && tick_count-begin_tick===5,"pacing cadence");
        @(negedge clk);cnn_busy=1;ticks(2);response(1,101);
        action(1,102);terminal(5,3,3,303);
        check(trail===64'hf,"limit trail/count");
        // A goal reached on the last allowed attempt wins over step limit.
        scenario(0,0,3);action(1,1);action(1,1);action(1,1);terminal(2,3,3,3);
        // Immediate reverse creates a loop on attempt 2; count retains the attempt.
        scenario(0,0,63);action(1,7);action(3,7);terminal(4,0,2,14);
        // Busy without done must not advance the position/counters.
        scenario(0,0,63);request();ticks(20);
        check(status===1 && agent_cell===0 && steps===0 && total_cycles===0,"busy advanced episode");
        // Restart has priority over a simultaneous old completion.
        @(negedge clk);start=1;walls=0;start_cell=18;goal_cell=19;cnn_done=1;cnn_busy=0;cnn_action=2;cnn_cycles=500;
        ticks(1);
        check(agent_cell===18 && steps===0 && total_cycles===0 && status===1,"restart accepted old completion");
        @(negedge clk);start=0; // stale done holds reset state, never counts a step
        ticks(3);check(cnn_start===0 && steps===0 && cnn_reset===1,"stale done escaped restart reset");
        @(negedge clk);cnn_done=0;
        action(1,55);terminal(2,19,1,55);
        // A spurious done before new busy acknowledgement is ignored.
        scenario(0,1,2);
        while(cnn_start!==1) ticks(1);
        @(negedge clk);cnn_done=1;cnn_busy=0;cnn_action=3;
        ticks(3);check(steps===0 && agent_cell===1,"unacknowledged stale response counted");
        @(negedge clk);cnn_done=0;cnn_busy=1;ticks(2);
        response(1,56);terminal(2,2,1,56);
        // Reset mid-inference immediately aborts and clears every episode output.
        scenario(0,0,63);request();
        @(negedge clk);rst=1;ticks(1);
        check(status===0 && agent_cell===0 && trail===0 && steps===0 && total_cycles===0 && cnn_start===0 && cnn_reset===1,"midflight reset");
        @(negedge clk);rst=0;cnn_busy=0;cnn_done=1;
        ticks(8);check(status===0 && steps===0 && cnn_start===0,"old completion after reset");
        @(negedge clk);cnn_done=0;
        scenario(0,62,63);action(1,57);terminal(2,63,1,57);
        $display("PASS_NAVIGATION checks=%0d bounds=4 legal_moves=4 wall=1 loop=1 goal=1 start_goal=1 limit=1 snapshot=1 restart=1 stale_done=1 reset=1 pacing=1",checks);
        $finish;
    end
    initial begin #200000; $fatal(1,"navigation watchdog");end
endmodule
