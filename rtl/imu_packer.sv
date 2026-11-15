// ============================================================================
// imu_packer.sv - IMU sample FIFO, IMU/servo packet builder, UART arbiter
// Target: Nexys4 DDR, 100 MHz. No block RAM (the FIFO is LUT-RAM).
//
// imu_fifo     : 32 x 208-bit samples in distributed RAM; a sample that arrives while
//                the FIFO is full is dropped and sets a sticky overflow flag (cleared
//                when a packet reports it).
// imu_packer   : one IMU packet with all samples in the FIFO at its start (format:
//                tools/fpga_model.py imu_packet), all little-endian:
//     A5 5A 49 4D | ver=1 | flags (bit0 IMU1 ok, bit1 IMU2 ok, bit2 FIFO overflow) |
//     whoami1 | whoami2 | servo_mode | servo_target_us u16 | servo_now_us u16 | n |
//     n x { sample_no u16, ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2 int16 } |
//     sum16 (all bytes after the 4-byte sync)
// uart_arbiter : uart_packer (result packets) and imu_packer share one uart_tx.
//                A packet is always sent whole: an IMU packet starts only when no
//                result packet is running or waiting; a result packet waits for a
//                running IMU packet to finish.
// ============================================================================
`timescale 1ns/1ps

module imu_fifo #(
    parameter int W     = 208,
    parameter int DEPTH = 32
)(
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     wr,
    input  logic [W-1:0]             wdata,
    input  logic                     pop,
    output logic [W-1:0]             q,           // oldest sample (registered, valid 1 cycle after a pop)
    output logic [$clog2(DEPTH):0]   count,
    output logic                     ovf,         // sticky: a sample was dropped
    input  logic                     ovf_clr
);
    localparam int AW = $clog2(DEPTH);
    (* ram_style = "distributed" *) logic [W-1:0] mem [DEPTH];
    logic [AW-1:0] wp, rp;
    logic          full, do_wr;
    assign full  = (count == (AW+1)'(DEPTH));
    assign do_wr = wr && !full;

    always_ff @(posedge clk) if (do_wr) mem[wp] <= wdata;
    always_ff @(posedge clk) q <= mem[rp];

    always_ff @(posedge clk) begin
        if (rst) begin
            wp <= '0; rp <= '0; count <= '0; ovf <= 1'b0;
        end else begin
            if (do_wr) wp <= wp + 1'b1;
            if (pop)   rp <= rp + 1'b1;
            if (do_wr && !pop)      count <= count + 1'b1;
            else if (!do_wr && pop) count <= count - 1'b1;
            if (wr && full)   ovf <= 1'b1;                 // set wins over clear
            else if (ovf_clr) ovf <= 1'b0;
        end
    end
endmodule


module imu_packer (
    input  logic         clk,
    input  logic         rst,
    input  logic         start,
    output logic         busy,
    output logic         done,
    // FIFO
    input  logic [5:0]   fifo_count,
    input  logic [207:0] fifo_q,
    output logic         fifo_pop,
    input  logic         fifo_ovf,
    output logic         ovf_clr,
    // status (snapshot at start)
    input  logic [1:0]   imu_ok,
    input  logic [7:0]   who1,
    input  logic [7:0]   who2,
    input  logic [7:0]   servo_mode,
    input  logic [15:0]  servo_target_us,
    input  logic [15:0]  servo_now_us,
    // byte stream to uart_tx (through the arbiter)
    output logic         tx_valid,
    output logic [7:0]   tx_data,
    input  logic         tx_ready
);
    typedef enum logic [1:0] {SEC_HDR, SEC_DATA, SEC_SUM} sec_t;
    typedef enum logic [1:0] {P_IDLE, P_PREP, P_SEND, P_PUSH} pst_t;
    sec_t         sec;
    pst_t         ps;
    logic [111:0] hsh;              // sync + header, byte 0 in [7:0]
    logic [199:0] dsh;              // rest of the current sample
    logic [4:0]   bcnt;
    logic [5:0]   scnt;
    logic [15:0]  sum;
    logic [7:0]   cur;
    logic         in_sync;

    assign in_sync = (sec == SEC_HDR) && (bcnt < 5'd4);

    always_ff @(posedge clk) begin
        done     <= 1'b0;
        tx_valid <= 1'b0;
        fifo_pop <= 1'b0;
        ovf_clr  <= 1'b0;
        if (rst) begin
            ps <= P_IDLE; busy <= 1'b0; sec <= SEC_HDR; bcnt <= '0; scnt <= '0; sum <= '0;
            hsh <= '0; dsh <= '0; cur <= '0; tx_data <= '0;
        end else begin
            unique case (ps)
            P_IDLE: if (start) begin
                if (fifo_count == 6'd0) done <= 1'b1;           // never an empty packet
                else begin
                    busy <= 1'b1;
                    hsh  <= {2'b00, fifo_count, servo_now_us, servo_target_us, servo_mode, who2, who1,
                             5'b0, fifo_ovf, imu_ok, 8'd1, 8'h4D, 8'h49, 8'h5A, 8'hA5};
                    ovf_clr <= fifo_ovf;
                    scnt <= fifo_count;
                    sec  <= SEC_HDR; bcnt <= '0; sum <= '0;
                    ps   <= P_PREP;
                end
            end
            P_PREP: begin
                ps <= P_SEND;
                unique case (sec)
                SEC_HDR:  cur <= hsh[7:0];
                SEC_DATA: if (bcnt == 5'd0) begin
                              cur <= fifo_q[7:0]; dsh <= fifo_q[207:8]; fifo_pop <= 1'b1;
                          end else begin
                              cur <= dsh[7:0]; dsh <= {8'h00, dsh[199:8]};
                          end
                default:  cur <= (bcnt == 5'd0) ? sum[7:0] : sum[15:8];
                endcase
            end
            P_SEND: if (tx_ready && !tx_valid) begin
                tx_valid <= 1'b1;
                tx_data  <= cur;
                if (!in_sync && sec != SEC_SUM) sum <= sum + 16'(cur);
                ps <= P_PUSH;
            end
            P_PUSH: begin                                       // tx_ready drops next cycle
                ps <= P_PREP;
                unique case (sec)
                SEC_HDR: begin
                    hsh <= {8'h00, hsh[111:8]};
                    if (bcnt == 5'd13) begin sec <= SEC_DATA; bcnt <= '0; end
                    else bcnt <= bcnt + 1'b1;
                end
                SEC_DATA:
                    if (bcnt == 5'd25) begin
                        bcnt <= '0;
                        if (scnt == 6'd1) sec <= SEC_SUM;
                        scnt <= scnt - 6'd1;
                    end else bcnt <= bcnt + 1'b1;
                default:
                    if (bcnt == 5'd1) begin ps <= P_IDLE; busy <= 1'b0; done <= 1'b1; end
                    else bcnt <= bcnt + 1'b1;
                endcase
            end
            default: ps <= P_IDLE;
            endcase
        end
    end
endmodule


module uart_arbiter (
    input  logic       clk,
    input  logic       rst,
    // result packets (uart_packer)
    input  logic       res_req,          // a result packet is ready to go (held until res_start)
    output logic       res_start,
    input  logic       res_done,
    input  logic       res_valid,
    input  logic [7:0] res_data,
    // IMU packets (imu_packer)
    input  logic       imu_req,          // FIFO not empty
    output logic       imu_start,
    input  logic       imu_done,
    input  logic       imu_valid,
    input  logic [7:0] imu_data,
    // to uart_tx
    output logic       tx_valid,
    output logic [7:0] tx_data,
    output logic       sel_imu
);
    typedef enum logic [1:0] {A_IDLE, A_RES, A_IMU} ast_t;
    ast_t st;
    always_ff @(posedge clk) begin
        res_start <= 1'b0;
        imu_start <= 1'b0;
        if (rst) st <= A_IDLE;
        else unique case (st)
            A_IDLE: if (res_req)      begin res_start <= 1'b1; st <= A_RES; end   // results first
                    else if (imu_req) begin imu_start <= 1'b1; st <= A_IMU; end
            A_RES:  if (res_done) st <= A_IDLE;
            A_IMU:  if (imu_done) st <= A_IDLE;
            default: st <= A_IDLE;
        endcase
    end
    assign sel_imu  = (st == A_IMU);
    assign tx_valid = res_valid | imu_valid;
    assign tx_data  = sel_imu ? imu_data : res_data;
endmodule
