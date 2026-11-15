// ============================================================================
// spi_rx.sv - SPI slave receivers (mode 0, MSB first), for the Pmod ports
// Target: Nexys4 DDR, 100 MHz system clock.
//
//   spi_slave_rx : SCLK / MOSI / CS_N sampled with 2-FF synchronisers (no SCLK
//                  clock domain), byte stream + frame start/end. SCLK up to ~10 MHz.
//   cam_rx       : 'CAM1', frame_no u16, 19200 pixels, sum16  ->  double frame buffer
//   tof_rx       : 'TOF1', frame_no u16, 64 x u16 distance, 64 x u8 status, sum16
//
// Frame buffers: two 160x120 buffers. The receiver writes the one that the
// consumer has NOT locked; a frame is "ready" only after a complete packet with
// the right length, magic and checksum. A newer frame replaces an unread one.
// ============================================================================
`timescale 1ns/1ps

module spi_slave_rx (
    input  logic       clk,
    input  logic       rst,
    input  logic       sclk_i,
    input  logic       mosi_i,
    input  logic       csn_i,
    output logic       frame_start,     // CS_N fell
    output logic       frame_end,       // CS_N rose
    output logic       byte_valid,
    output logic [7:0] byte_data
);
    logic [2:0] s_sclk, s_csn;
    logic [1:0] s_mosi;
    always_ff @(posedge clk) begin
        s_sclk <= {s_sclk[1:0], sclk_i};
        s_csn  <= {s_csn[1:0],  csn_i};
        s_mosi <= {s_mosi[0],   mosi_i};
    end
    logic sclk_rise, cs_act;
    assign sclk_rise = (s_sclk[2:1] == 2'b01);
    assign cs_act    = !s_csn[1];

    logic [2:0] nbit;
    logic [7:0] sh;
    always_ff @(posedge clk) begin
        frame_start <= 1'b0;
        frame_end   <= 1'b0;
        byte_valid  <= 1'b0;
        if (rst) begin
            nbit <= '0;
        end else begin
            if (s_csn[2:1] == 2'b10) begin frame_start <= 1'b1; nbit <= '0; end
            if (s_csn[2:1] == 2'b01) frame_end <= 1'b1;
            if (cs_act && sclk_rise) begin
                sh   <= {sh[6:0], s_mosi[1]};
                nbit <= nbit + 3'd1;
                if (nbit == 3'd7) begin
                    byte_valid <= 1'b1;
                    byte_data  <= {sh[6:0], s_mosi[1]};
                end
            end
        end
    end
endmodule


module cam_rx
    import unet_pkg::*;
(
    input  logic           clk,
    input  logic           rst,
    input  logic           sclk_i, mosi_i, csn_i,

    // consumer side
    input  logic           take,            // lock the ready buffer (when ready)
    output logic           ready,           // a complete new frame is waiting
    output logic           cur_buf,         // buffer locked by the consumer
    output logic [15:0]    cur_frame_no,
    input  logic           rd_re,
    input  logic [FAW-1:0] rd_addr,
    output logic [7:0]     rd_data,         // 2-cycle latency, from cur_buf

    output logic [15:0]    n_ok,            // good packets
    output logic [15:0]    n_err            // rejected packets
);
    logic       fs, fe, bv;
    logic [7:0] bd;
    spi_slave_rx u_spi (.clk, .rst, .sclk_i, .mosi_i, .csn_i,
                        .frame_start(fs), .frame_end(fe), .byte_valid(bv), .byte_data(bd));

    localparam int NBYTES = 4 + 2 + IMG_N + 2;      // 19208

    logic        wbuf, ready_buf;
    logic [14:0] cnt;                                // bytes received in this packet
    logic        bad;
    logic [15:0] fno, sum, rx_sum;
    logic [15:0] buf_fno [2];

    // frame buffers: 2 x 19200 x 8 (one RAM of 2*IMG_N, buffer in the MSB region)
    logic        we;
    logic [15:0] waddr;
    logic [7:0]  wdata;
    ram_sdp_ch #(.W(8), .D(2 * IMG_N), .CH(4096), .AW(16), .OREG(1)) u_fb (   // 10 RAMB36, latency 2
        .clk, .we, .waddr, .wdata,
        .re(rd_re), .raddr(cur_buf ? 16'(rd_addr) + 16'(IMG_N) : 16'(rd_addr)), .rdata(rd_data));

    always_ff @(posedge clk) begin
        we <= 1'b0;
        if (rst) begin
            ready <= 1'b0; cur_buf <= 1'b0; ready_buf <= 1'b0; wbuf <= 1'b1;
            cur_frame_no <= '0; fno <= '0; sum <= '0; rx_sum <= '0;
            buf_fno[0] <= '0; buf_fno[1] <= '0;
            cnt <= '0; bad <= 1'b1; n_ok <= '0; n_err <= '0;
        end else begin
            if (take && ready) begin
                cur_buf      <= ready_buf;
                cur_frame_no <= buf_fno[ready_buf];
                ready        <= 1'b0;
            end
            if (fs) begin
                // write the buffer the consumer does not hold; it may be the unread ready one
                logic tgt;
                tgt = (take && ready) ? !ready_buf : !cur_buf;
                wbuf <= tgt;
                if (ready && ready_buf == tgt && !(take && ready)) ready <= 1'b0;
                cnt <= '0; bad <= 1'b0; sum <= '0;
            end else if (bv) begin
                cnt <= cnt + 15'd1;
                unique case (cnt)
                15'd0: if (bd != 8'h43) bad <= 1'b1;         // 'C'
                15'd1: if (bd != 8'h41) bad <= 1'b1;         // 'A'
                15'd2: if (bd != 8'h4D) bad <= 1'b1;         // 'M'
                15'd3: if (bd != 8'h31) bad <= 1'b1;         // '1'
                15'd4: fno[7:0]  <= bd;
                15'd5: fno[15:8] <= bd;
                15'(NBYTES - 2): rx_sum[7:0]  <= bd;
                15'(NBYTES - 1): rx_sum[15:8] <= bd;
                default: begin
                    if (cnt < 15'(NBYTES - 2)) begin
                        we    <= 1'b1;
                        waddr <= wbuf ? 16'(cnt - 15'd6) + 16'(IMG_N) : 16'(cnt - 15'd6);
                        wdata <= bd;
                        sum   <= sum + 16'(bd);
                    end else
                        bad <= 1'b1;                          // too long
                end
                endcase
            end
            if (fe) begin
                if (!bad && cnt == 15'(NBYTES) && rx_sum == sum) begin
                    ready     <= 1'b1;
                    ready_buf <= wbuf;
                    buf_fno[wbuf] <= fno;
                    n_ok      <= n_ok + 16'd1;
                end else if (cnt != 0)
                    n_err <= n_err + 16'd1;
            end
        end
    end
endmodule


module tof_rx (
    input  logic            clk,
    input  logic            rst,
    input  logic            sclk_i, mosi_i, csn_i,
    output logic            valid,           // at least one good frame received
    output logic [15:0]     frame_no,
    output logic [16*64-1:0] dist_mm,        // zone i at [i*16 +: 16]
    output logic [8*64-1:0]  status,         // zone i at [i*8 +: 8]
    output logic [15:0]     n_ok,
    output logic [15:0]     n_err
);
    logic       fs, fe, bv;
    logic [7:0] bd;
    spi_slave_rx u_spi (.clk, .rst, .sclk_i, .mosi_i, .csn_i,
                        .frame_start(fs), .frame_end(fe), .byte_valid(bv), .byte_data(bd));

    localparam int NBYTES = 4 + 2 + 128 + 64 + 2;    // 200
    logic [7:0]  cnt;
    logic        bad;
    logic [15:0] fno, sum, rx_sum;
    logic [16*64-1:0] sd;                            // distance bytes, shifted in (byte k ends at [k*8 +: 8])
    logic [8*64-1:0]  ss;                            // status bytes, shifted in
    logic        in_d, in_s;                         // registered: the NEXT byte is a distance / status byte

    always_ff @(posedge clk) begin
        if (rst) begin
            valid <= 1'b0; cnt <= '0; bad <= 1'b1; n_ok <= '0; n_err <= '0;
            frame_no <= '0; fno <= '0; sum <= '0; rx_sum <= '0;
            dist_mm <= '0; status <= '0; in_d <= 1'b0; in_s <= 1'b0;
        end else begin
            if (fs) begin
                cnt <= '0; bad <= 1'b0; sum <= '0; in_d <= 1'b0; in_s <= 1'b0;
            end else if (bv) begin
                cnt <= cnt + 8'd1;
                // timing: byte-region flags are decoded one byte early, and bytes are
                // shifted in (no cnt-indexed 128:1 write decode in the data path)
                in_d <= (cnt >= 8'd5)   && (cnt < 8'd133);
                in_s <= (cnt >= 8'd133) && (cnt < 8'd197);
                if (cnt == 8'd0 && bd != 8'h54) bad <= 1'b1;      // 'T'
                if (cnt == 8'd1 && bd != 8'h4F) bad <= 1'b1;      // 'O'
                if (cnt == 8'd2 && bd != 8'h46) bad <= 1'b1;      // 'F'
                if (cnt == 8'd3 && bd != 8'h31) bad <= 1'b1;      // '1'
                if (cnt == 8'd4) fno[7:0]  <= bd;
                if (cnt == 8'd5) fno[15:8] <= bd;
                if (in_d) begin                                     // bytes 6..133
                    sd  <= {bd, sd[16*64-1:8]};                     // little-endian u16 stream
                    sum <= sum + 16'(bd);
                end
                if (in_s) begin                                     // bytes 134..197
                    ss  <= {bd, ss[8*64-1:8]};
                    sum <= sum + 16'(bd);
                end
                if (cnt == 8'd198) rx_sum[7:0]  <= bd;
                if (cnt == 8'd199) rx_sum[15:8] <= bd;
                if (cnt >= 8'(NBYTES)) bad <= 1'b1;
            end
            if (fe) begin
                if (!bad && cnt == 8'(NBYTES) && rx_sum == sum) begin
                    dist_mm  <= sd;
                    status   <= ss;
                    frame_no <= fno;
                    valid    <= 1'b1;
                    n_ok     <= n_ok + 16'd1;
                end else if (cnt != 0)
                    n_err <= n_err + 16'd1;
            end
        end
    end
endmodule
