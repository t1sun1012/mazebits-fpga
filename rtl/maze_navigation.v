`timescale 1ns/1ps
`default_nettype none

// One deterministic, stateless policy episode. A step counts an attempted action,
// including the final blocked action. No route search or action repair is used.
// cnn_done is accepted only after this controller has observed cnn_busy for its
// current request. rst/start abort outstanding inference through cnn_reset.
module maze_navigation #(
    parameter integer MAX_STEPS = 64,
    parameter integer STEP_DELAY_CYCLES = 20000000
) (
    input wire clk,
    input wire rst,
    input wire start,
    input wire [63:0] walls,
    input wire [5:0] start_cell,
    input wire [5:0] goal_cell,
    input wire cnn_busy,
    input wire cnn_done,
    input wire [1:0] cnn_action,
    input wire [31:0] cnn_cycles,
    output reg cnn_start,
    output wire cnn_reset,
    output reg [63:0] cnn_walls,
    output wire [5:0] cnn_agent,
    output reg [5:0] cnn_goal,
    output reg [5:0] agent_cell,
    output reg [63:0] trail,
    output reg [2:0] status,
    output reg [6:0] steps,
    output reg [31:0] total_cycles,
    output reg step_event
);
    localparam [2:0] READY=0, RUN=1, GOAL=2, BLOCKED=3, LOOP=4, LIMIT=5;
    localparam [2:0] N_IDLE=0, N_RESET=1, N_REQUEST=2, N_ACK=3,
                     N_WAIT=4, N_DELAY=5;
    function integer count_width;
        input integer n;
        integer v;
        begin
            count_width=1;
            v=n-1;
            while(v>1) begin count_width=count_width+1; v=v>>1; end
        end
    endfunction
    // Supported parameter bounds keep the exposed seven-bit attempt count exact.
    // synthesis translate_off
    initial begin
        if(MAX_STEPS < 1 || MAX_STEPS > 127) $fatal(1, "MAX_STEPS must be 1..127");
        if(STEP_DELAY_CYCLES < 0) $fatal(1, "STEP_DELAY_CYCLES must be nonnegative");
    end
    // synthesis translate_on
    localparam integer DELAY_BITS=count_width(STEP_DELAY_CYCLES);
    reg [2:0] state;
    reg [DELAY_BITS-1:0] delay_count;
    assign cnn_reset = rst || start || (state == N_RESET);
    assign cnn_agent = agent_cell;

    reg [5:0] next_cell;
    reg outside;
    always @* begin
        next_cell=agent_cell;
        outside=1'b0;
        case(cnn_action)
            2'd0: begin outside=(agent_cell[5:3]==0); next_cell=agent_cell-6'd8; end
            2'd1: begin outside=(agent_cell[2:0]==7); next_cell=agent_cell+6'd1; end
            2'd2: begin outside=(agent_cell[5:3]==7); next_cell=agent_cell+6'd8; end
            2'd3: begin outside=(agent_cell[2:0]==0); next_cell=agent_cell-6'd1; end
        endcase
    end
    wire blocked = outside || cnn_walls[next_cell];

    always @(posedge clk) begin
        if(rst) begin
            state<=N_IDLE;
            cnn_start<=0;
            cnn_walls<=0;
            cnn_goal<=0;
            agent_cell<=0;
            trail<=0;
            status<=READY;
            steps<=0;
            total_cycles<=0;
            delay_count<=0;
            step_event<=0;
        end else begin
            cnn_start<=0;
            step_event<=0;
            if(start) begin
                cnn_walls<=walls;
                cnn_goal<=goal_cell;
                agent_cell<=start_cell;
                trail<=64'b1 << start_cell;
                steps<=0;
                total_cycles<=0;
                delay_count<=0;
                // The editor guarantees free endpoints. Reject invalid external
                // scenarios here as a defensive zero-inference terminal state.
                if(walls[start_cell] || walls[goal_cell]) begin
                    status<=BLOCKED;
                    state<=N_IDLE;
                end else if(start_cell==goal_cell) begin
                    status<=GOAL;
                    state<=N_IDLE;
                end else begin
                    status<=RUN;
                    state<=N_RESET;
                end
            end else begin
                case(state)
                    N_IDLE: begin end
                    N_RESET: begin
                        // Reset is held until the previous request is quiescent.
                        if(!cnn_busy && !cnn_done) state<=N_REQUEST;
                    end
                    N_REQUEST: begin
                        if(!cnn_busy && !cnn_done) begin
                            cnn_start<=1;
                            state<=N_ACK;
                        end
                    end
                    N_ACK: begin
                        if(cnn_busy) state<=N_WAIT;
                    end
                    N_WAIT: begin
                        if(cnn_done && !cnn_busy) begin
                            steps<=steps+1'b1;
                            total_cycles<=total_cycles+cnn_cycles;
                            step_event<=1;
                            if(blocked) begin
                                status<=BLOCKED;
                                state<=N_IDLE;
                            end else begin
                                agent_cell<=next_cell;
                                trail[next_cell]<=1'b1;
                                if(next_cell==cnn_goal) begin
                                    status<=GOAL;
                                    state<=N_IDLE;
                                end else if(trail[next_cell]) begin
                                    status<=LOOP;
                                    state<=N_IDLE;
                                end else if(steps==MAX_STEPS-1) begin
                                    status<=LIMIT;
                                    state<=N_IDLE;
                                end else if(STEP_DELAY_CYCLES==0) begin
                                    state<=N_REQUEST;
                                end else begin
                                    delay_count<=STEP_DELAY_CYCLES-1;
                                    state<=N_DELAY;
                                end
                            end
                        end
                    end
                    N_DELAY: begin
                        if(delay_count==0) state<=N_REQUEST;
                        else delay_count<=delay_count-1'b1;
                    end
                    default: begin
                        state<=N_IDLE;
                        status<=READY;
                    end
                endcase
            end
        end
    end
endmodule
`default_nettype wire
