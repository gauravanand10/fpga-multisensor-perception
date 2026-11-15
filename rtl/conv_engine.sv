// ============================================================================
// conv_engine.sv - int8 convolution engine for the Tiny U-Net (Stage 1)
// Target: Nexys4 DDR (Artix-7 XC7A100T), 100 MHz.   SystemVerilog-2012.
//
//   * 3x3 (pad 1) or 1x1 (pad 0) kernels, stride 1 or 2
//   * PO output channels in parallel (one int8 x int8 MAC per lane)
//   * int32 accumulator with bias preload
//   * requant  q = clamp((acc*M + 2^(sh-1)) >>> sh, -127, 127)  (INT8_SPEC.md)
//   * optional ReLU after the clamp
//
// Memory interfaces (synchronous, 1-cycle read latency = plain BRAM):
//
//   Activations: PO banks, one int8 per word.
//     channel c, pixel (y,x) -> bank = c % PO,
//                               addr = base + (c / PO)*H*W + y*W + x
//     ONE byte read per cycle, PO bytes written per cycle (lane mask).
//
//   Weights: one PO*8-bit word per (output group, input channel, tap)
//     addr = w_base + ocg*(Cin*KK) + ic*KK + tap, lane p = oc ocg*PO+p,
//     tap = ky*3+kx (3x3) or 0 (1x1).
//
//   Params: one word per output group, lane p = prm_lane_t {sh, M, bias}.
//
// Loop order: ocg -> oy -> ox -> ic -> tap.  One tap per clock per lane.
// Pipeline: S0 address -> S1 BRAM read -> S2 MAC -> S3 acc*M -> S4 round/clamp
// ============================================================================
`timescale 1ns/1ps

module conv_engine
    import unet_pkg::*;
#(
    parameter int P_PO  = unet_pkg::PO,
    parameter int P_AAW = unet_pkg::AAW,
    parameter int P_WAW = unet_pkg::WAW,
    parameter int P_PAW = unet_pkg::PAW,
    localparam int P_LPO = $clog2(P_PO)
)(
    input  logic                      clk,
    input  logic                      rst,

    // ---- layer configuration (hold stable while busy) ----
    input  logic                      start,
    input  logic [7:0]                cfg_cin,
    input  logic [7:0]                cfg_cout,
    input  logic [8:0]                cfg_hin,
    input  logic [8:0]                cfg_win,
    input  logic [8:0]                cfg_hout,
    input  logic [8:0]                cfg_wout,
    input  logic                      cfg_k3,       // 1 = 3x3 pad 1, 0 = 1x1
    input  logic                      cfg_s2,       // 1 = stride 2
    input  logic                      cfg_relu,
    input  logic [P_AAW-1:0]          cfg_in_base,
    input  logic [P_AAW-1:0]          cfg_out_base,
    input  logic [P_WAW-1:0]          cfg_w_base,
    input  logic [P_PAW-1:0]          cfg_p_base,
    output logic                      busy,
    output logic                      done,         // 1-cycle pulse

    // ---- activation read (1 byte / cycle) ----
    output logic                      act_rd_en,
    output logic [P_LPO-1:0]          act_rd_bank,
    output logic [P_AAW-1:0]          act_rd_addr,
    input  logic [7:0]                act_rd_data,

    // ---- weight read ----
    output logic                      wt_rd_en,
    output logic [P_WAW-1:0]          wt_rd_addr,
    input  logic [P_PO*8-1:0]         wt_rd_data,

    // ---- param read ----
    output logic                      prm_rd_en,
    output logic [P_PAW-1:0]          prm_rd_addr,
    input  logic [P_PO*PRM_LANE_W-1:0] prm_rd_data,

    // ---- activation write (PO bytes / cycle) ----
    output logic                      out_wr_en,
    output logic [P_PO-1:0]           out_wr_mask,
    output logic [P_AAW-1:0]          out_wr_addr,
    output logic [P_PO*8-1:0]         out_wr_data
);

    // ------------------------------------------------------------------
    // derived configuration
    // ------------------------------------------------------------------
    logic [3:0]  kk;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [17:0] plane_in, plane_out;           // top bits unused (max 19200)
    /* verilator lint_on UNUSEDSIGNAL */
    logic [11:0] cin_kk;
    logic [7:0]  n_ocg;

    always_comb begin
        kk        = cfg_k3 ? 4'd9 : 4'd1;
        plane_in  = 18'(cfg_hin)  * 18'(cfg_win);    // max 120*160 = 19200
        plane_out = 18'(cfg_hout) * 18'(cfg_wout);
        cin_kk    = 12'(cfg_cin)  * 12'(kk);
        n_ocg     = 8'((9'(cfg_cout) + 9'(P_PO - 1)) >> P_LPO);
    end

    // ------------------------------------------------------------------
    // control FSM
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {S_IDLE, S_LDP, S_WTP, S_RUN, S_DRN} state_t;
    state_t st;

    logic [7:0]       ocg;
    logic [8:0]       oy, ox;
    logic [7:0]       ic;
    logic [1:0]       ky, kx;
    logic [3:0]       tap;
    logic [P_WAW-1:0] w_grp_base;    // w_base + ocg*cin_kk
    logic [P_WAW-1:0] w_addr;        // running weight address
    logic [P_AAW-1:0] o_grp_base;    // out_base + ocg*plane_out
    logic [P_AAW-1:0] o_pix;         // oy*Wout + ox

    // per-group params, latched in S_WTP
    logic signed [31:0] bias [P_PO];
    logic        [14:0] mult [P_PO];
    logic        [5:0]  shft [P_PO];
    logic [P_PO-1:0]    lane_on;

    logic last_tap, last_ic, last_x, last_y, last_pix, issue;
    always_comb begin
        last_tap = (tap == kk - 4'd1);
        last_ic  = (ic  == cfg_cin  - 8'd1);
        last_x   = (ox  == cfg_wout - 9'd1);
        last_y   = (oy  == cfg_hout - 9'd1);
        last_pix = last_tap && last_ic;
        issue    = (st == S_RUN);
    end

    logic v1, v2, v3, v4;            // pipeline valid bits

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            st   <= S_IDLE;
            busy <= 1'b0;
        end else begin
            unique case (st)
            S_IDLE: if (start) begin
                busy       <= 1'b1;
                ocg        <= '0;
                w_grp_base <= cfg_w_base;
                o_grp_base <= cfg_out_base;
                st         <= S_LDP;
            end
            S_LDP: st <= S_WTP;                      // param address presented
            S_WTP: begin                             // param data valid
                for (int i = 0; i < P_PO; i++) begin
                    prm_lane_t pl;
                    pl         = prm_lane_t'(prm_rd_data[i*PRM_LANE_W +: PRM_LANE_W]);
                    bias[i]    <= pl.bias;
                    mult[i]    <= pl.m[14:0];
                    shft[i]    <= pl.sh[5:0];
                    lane_on[i] <= (16'({ocg, {P_LPO{1'b0}}}) + 16'(i)) < 16'(cfg_cout);
                end
                {oy, ox, ic, ky, kx, tap} <= '0;
                w_addr <= w_grp_base;
                o_pix  <= '0;
                st     <= S_RUN;
            end
            S_RUN: begin
                if (!last_tap) begin
                    tap    <= tap + 4'd1;
                    w_addr <= w_addr + 1'b1;
                    if (kx == 2'd2) begin kx <= '0; ky <= ky + 2'd1; end
                    else              kx <= kx + 2'd1;
                end else begin
                    {tap, kx, ky} <= '0;
                    if (!last_ic) begin
                        ic     <= ic + 8'd1;
                        w_addr <= w_addr + 1'b1;
                    end else begin
                        ic     <= '0;
                        w_addr <= w_grp_base;
                        o_pix  <= o_pix + 1'b1;
                        if (!last_x) ox <= ox + 9'd1;
                        else begin
                            ox <= '0;
                            if (!last_y) oy <= oy + 9'd1;
                            else         st <= S_DRN;
                        end
                    end
                end
            end
            S_DRN: if (!(v1 || v2 || v3 || v4 || out_wr_en)) begin
                if (ocg == n_ocg - 8'd1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    st   <= S_IDLE;
                end else begin
                    ocg        <= ocg + 8'd1;
                    w_grp_base <= w_grp_base + P_WAW'(cin_kk);
                    o_grp_base <= o_grp_base + P_AAW'(plane_out);
                    st         <= S_LDP;
                end
            end
            default: st <= S_IDLE;
            endcase
        end
    end

    assign prm_rd_en   = (st == S_LDP);
    assign prm_rd_addr = cfg_p_base + P_PAW'(ocg);

    // ------------------------------------------------------------------
    // S0: input coordinate, padding test, read addresses
    // ------------------------------------------------------------------
    logic signed [11:0] iy, ix;
    logic               is_pad;
    logic [7:0]         icg;
    always_comb begin
        iy     = $signed({2'b00, (cfg_s2 ? {oy, 1'b0} : {1'b0, oy})}) + $signed(12'(ky)) - (cfg_k3 ? 12'sd1 : 12'sd0);
        ix     = $signed({2'b00, (cfg_s2 ? {ox, 1'b0} : {1'b0, ox})}) + $signed(12'(kx)) - (cfg_k3 ? 12'sd1 : 12'sd0);
        is_pad = (iy < 0) || (ix < 0) ||
                 (iy >= $signed(12'(cfg_hin))) || (ix >= $signed(12'(cfg_win)));
        icg    = ic >> P_LPO;
        act_rd_addr = cfg_in_base + P_AAW'(icg) * P_AAW'(plane_in)
                    + P_AAW'(iy[8:0]) * P_AAW'(cfg_win) + P_AAW'(ix[8:0]);
    end

    assign act_rd_en   = issue && !is_pad;
    assign act_rd_bank = ic[P_LPO-1:0];
    assign wt_rd_en    = issue;
    assign wt_rd_addr  = w_addr;

    // ------------------------------------------------------------------
    // S1: BRAM responds; carry flags
    // ------------------------------------------------------------------
    logic             p1_pad, p1_first, p1_last;
    logic [P_AAW-1:0] p1_oaddr;
    always_ff @(posedge clk) begin
        v1       <= issue && !rst;
        p1_pad   <= is_pad;
        p1_first <= (tap == 4'd0) && (ic == 8'd0);
        p1_last  <= last_pix;
        p1_oaddr <= o_grp_base + o_pix;
    end

    // ------------------------------------------------------------------
    // S2: MAC   acc = bias + sum(x*w)
    // ------------------------------------------------------------------
    logic signed [7:0]  xin;
    logic signed [31:0] acc [P_PO];
    logic [P_AAW-1:0]   p2_oaddr;

    assign xin = p1_pad ? 8'sd0 : $signed(act_rd_data);

    always_ff @(posedge clk) begin
        if (v1)
            for (int g = 0; g < P_PO; g++) begin
                logic signed [15:0] prd;
                prd = xin * $signed(wt_rd_data[g*8 +: 8]);
                acc[g] <= (p1_first ? bias[g] : acc[g]) + 32'(prd);
            end
        v2       <= v1 && p1_last && !rst;
        p2_oaddr <= p1_oaddr;
    end

    // ------------------------------------------------------------------
    // S3: prod = acc * M        (32 x 16 signed -> 48 bit)
    // S4: round, shift, clamp, ReLU, write
    // ------------------------------------------------------------------
    logic signed [47:0] prod [P_PO];
    logic [P_AAW-1:0]   p3_oaddr;

    always_ff @(posedge clk) begin
        if (v2)
            for (int g = 0; g < P_PO; g++)
                prod[g] <= 48'(acc[g]) * $signed({1'b0, mult[g]});
        v3       <= v2 && !rst;
        v4       <= v3 && !rst;
        p3_oaddr <= p2_oaddr;
    end

    logic [P_PO*8-1:0] q_all;
    always_comb begin
        for (int g = 0; g < P_PO; g++) begin
            logic signed [47:0] shd;
            logic signed [7:0]  q;
            shd = (prod[g] + (48'sd1 <<< (shft[g] - 6'd1))) >>> shft[g];
            if      (shd >  48'sd127) q =  8'sd127;
            else if (shd < -48'sd127) q = -8'sd127;
            else                      q = shd[7:0];
            if (cfg_relu && q[7])     q = 8'sd0;
            q_all[g*8 +: 8] = q;
        end
    end

    always_ff @(posedge clk) begin
        out_wr_en   <= v3 && !rst;
        out_wr_addr <= p3_oaddr;
        out_wr_mask <= lane_on;
        if (v3) out_wr_data <= q_all;
    end

endmodule
