`timescale 1ns / 1ps
module tb_maze_editor;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, move_up = 0, move_right = 0, move_down = 0, move_left = 0, apply = 0;
    reg [1:0] mode = 0;
    wire [63:0] walls;
    wire [5:0] start_cell, goal_cell, cursor;
    wire changed, rejected;
    maze_editor dut(.*);

    integer checks = 0, n;
    reg [63:0] expected_walls = 0;
    integer expected_start = 0, expected_goal = 63, expected_cursor = 0;

    task check(input bit expected_changed, input bit expected_rejected);
        begin
            checks = checks + 1;
            if (walls !== expected_walls || start_cell !== expected_start[5:0] ||
                goal_cell !== expected_goal[5:0] || cursor !== expected_cursor[5:0] ||
                changed !== expected_changed || rejected !== expected_rejected)
                $fatal(1, "editor check %0d failed cursor=%0d expected=%0d walls=%h expected=%h start=%0d goal=%0d changed=%b rejected=%b",
                       checks, cursor, expected_cursor, walls, expected_walls,
                       start_cell, goal_cell, changed, rejected);
            if (walls[start_cell] !== 0 || walls[goal_cell] !== 0)
                $fatal(1, "endpoint is a wall");
        end
    endtask

    // Direction bit order is up, right, down, left. Expected position is supplied,
    // so the checker does not reuse the DUT's coordinate/update expressions.
    task drive(input reg [3:0] directions, input bit commit,
               input integer tool, input integer position,
               input bit expected_changed, input bit expected_rejected);
        begin
            @(negedge clk);
            {move_up,move_right,move_down,move_left} = directions;
            apply = commit;
            mode = tool;
            expected_cursor = position;
            @(posedge clk); #1;
            check(expected_changed, expected_rejected);
            @(negedge clk);
            {move_up,move_right,move_down,move_left} = 0;
            apply = 0;
            @(posedge clk); #1;
            check(0,0);
        end
    endtask

    task go(input integer destination);
        begin
            while (expected_cursor / 8 > destination / 8)
                drive(4'b1000,0,0,expected_cursor-8,0,0);
            while (expected_cursor / 8 < destination / 8)
                drive(4'b0010,0,0,expected_cursor+8,0,0);
            while (expected_cursor % 8 > destination % 8)
                drive(4'b0001,0,0,expected_cursor-1,0,0);
            while (expected_cursor % 8 < destination % 8)
                drive(4'b0100,0,0,expected_cursor+1,0,0);
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        #1; check(0,0);
        @(negedge clk); rst = 0;

        // Every boundary, including an opposite-direction simultaneous request.
        drive(4'b1000,0,0,0,0,0);
        drive(4'b0001,0,0,0,0,0);
        drive(4'b1010,0,0,0,0,0); // blocked UP wins over DOWN
        go(7);
        drive(4'b1000,0,0,7,0,0);
        drive(4'b0100,0,0,7,0,0);
        go(63);
        drive(4'b0100,0,0,63,0,0);
        drive(4'b0010,0,0,63,0,0);
        go(56);
        drive(4'b0010,0,0,56,0,0);
        drive(4'b0001,0,0,56,0,0);

        // Exercise each wall bit, independent of its neighbors, and both endpoints.
        for (n=0;n<64;n=n+1) begin
            go(n);
            if (n==0 || n==63) begin
                drive(0,1,0,n,0,1);
            end else begin
                expected_walls = 64'h1 << n;
                drive(0,1,0,n,1,0);
                expected_walls = 0;
                drive(0,1,0,n,1,0);
            end
        end

        // Multiple walls survive edits; placing an endpoint clears its wall.
        go(9); expected_walls = 64'h1 << 9; drive(0,1,0,9,1,0);
        go(10); expected_walls = expected_walls | (64'h1 << 10); drive(0,1,0,10,1,0);
        go(9); expected_start = 9; expected_walls = 64'h1 << 10; drive(0,1,1,9,1,0);
        drive(0,1,1,9,0,0); // idempotent endpoint placement
        drive(0,1,0,9,0,1);
        go(10); expected_goal = 10; expected_walls = 0; drive(0,1,2,10,1,0);
        drive(0,1,2,10,0,0);
        expected_start = 10; drive(0,1,1,10,1,0); // allowed overlap
        drive(0,1,0,10,0,1);

        // View mode freezes cursor and map despite every event being asserted.
        drive(4'b1111,1,3,10,0,0);
        drive(4'b1111,0,3,10,0,0);
        go(27);
        drive(4'b1111,0,0,19,0,0); // up wins
        drive(4'b0111,0,0,20,0,0); // right wins
        drive(4'b0011,0,0,28,0,0); // down wins
        expected_walls = 64'h1 << 28;
        drive(4'b1111,1,0,28,1,0); // apply wins, current cell remains selected

        @(negedge clk);
        rst = 1; apply = 1; move_up = 1;
        expected_walls=0; expected_start=0; expected_goal=63; expected_cursor=0;
        @(posedge clk); #1; check(0,0);
        @(negedge clk); rst=0; apply=0; move_up=0;
        @(posedge clk); #1; check(0,0);
        $display("PASS_EDITOR checks=%0d wall_cells=64 boundaries=8 priorities=4", checks);
        $finish;
    end
endmodule
