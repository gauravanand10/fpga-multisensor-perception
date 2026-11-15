// ============================================================================
// unet_engine.sv - int8 conv engine, PO=16 output x PI=8 input lanes (128 MAC/clk)
// Target: Nexys4 DDR (Artix-7 XC7A100T-1), 100 MHz.   SystemVerilog-2012.
//
// Runs ONE layer described by unet_layers_pkg::layer_t:
//   * 3x3 pad 1 or 1x1, stride 1 or 2
//   * input = camera frame through the 256-entry input LUT (layer 0), or the
//     activation banks; channels < ca come from source A, the rest from source B
//   * on-the-fly nearest x2 upsample of source A (address >> 1)
//   * on-the-fly requant of each source (Concat rescale, identity otherwise):
//         x' = clamp((x*M + 2^(sh-1)) >>> sh, -127, 127)
//   * int32 accumulate, requant (acc*M + 2^(sh-1)) >>> sh, clamp, ReLU
//   * optional argmax over the output lanes -> label write (head layer)
//
// Activation layout: channel c, pixel p of a tensor with plane size HW:
//     bank = c % 16, addr = base + (c / 16)*HW + p
// The 8 channels read together (ic0..ic0+7, ic0 = 8*icg) sit in banks 0-7 or 8-15
// at ONE address, so all 16 banks are read at the same address and half is used.
//
// Loop order: ocg -> oy -> ox -> icg -> tap; one (icg, tap) per clock.
//
// TIMING (100 MHz): every stage does at most one adder / small multiplier / mux
// level between registers. Memories have a 2-cycle read latency (registered
// address in this module, RAM output register + extra output register outside).
//
//   s1  coordinates, padding, source select        s11 128 int8 products
//   s2  address partial products, frame address     s12 two 4-input partial sums
//   s3  address sum  -> act/frame BRAM address      s13 accumulate (int32)
//   s4  (BRAM)                                      s14 acc_hi*M, acc_lo*M (DSP)
//   s5  (BRAM output register)                      s15 hi<<17 + lo
//   s6  lane select / input LUT / padding           s16 + 2^(sh-1)
//   s7  x * M_src                                   s17 >>> sh
//   s8  + 2^(sh_src-1);  weight ROM address         s18 clamp, ReLU -> bank write
//   s9  >>> sh_src                                  s19 argmax 16 -> 4
//   s10 clamp                                       s20 argmax 4 -> 1 -> label write
// ============================================================================
`timescale 1ns/1ps

module unet_engine
    import unet_pkg::*;
    import unet_layers_pkg::*;
#(
    parameter string INLUT_FILE = "unet_inlut.mem"
)(
    input  logic                   clk,
    input  logic                   rst,

    input  logic                   start,
    input  layer_t                 d,              // hold stable while busy
    output logic                   busy,
    output logic                   done,           // 1-cycle pulse

    // activation banks (all 16 at one address), read latency 2
    output logic                   act_re,
    output logic [BAW-1:0]         act_raddr,
    input  logic [NBANK*8-1:0]     act_rdata,      // bank b at [b*8 +: 8]
    output logic                   act_we,
    output logic [NBANK-1:0]       act_wmask,
    output logic [BAW-1:0]         act_waddr,
    output logic [NBANK*8-1:0]     act_wdata,

    // camera frame (layer 0), read latency 2
    output logic                   frm_re,
    output logic [FAW-1:0]         frm_raddr,
    input  logic [7:0]             frm_rdata,

    // weights (read latency 2) / params (read latency 1)
    output logic                   wt_re,
    output logic [WDAW-1:0]        wt_raddr,
    input  logic [WW-1:0]          wt_rdata,       // lane (p,i) at [(p*PI+i)*8 +: 8]
    output logic                   prm_re,
    output logic [PDAW-1:0]        prm_raddr,
    input  logic [PO*PRM_LANE_W-1:0] prm_rdata,

    // label map (argmax layer)
    output logic                   lab_we,
    output logic [LAW-1:0]         lab_waddr,
    output logic [2:0]             lab_wdata
);

    // ------------------------------------------------------------------
    // input LUT  (camera pixel -> int8), 256 x 8 in LUTs
    // ------------------------------------------------------------------
    logic [7:0] inlut [256];
    initial $readmemh(INLUT_FILE, inlut);

    // ------------------------------------------------------------------
    // layer constants, registered when the layer starts
    // ------------------------------------------------------------------
    logic [3:0]         kk_m1, nicg_m1;
    logic [7:0]         wout_m1;
    logic [6:0]         hout_m1;
    logic [2:0]         nocg_m1;
    logic signed [23:0] a_rnd, b_rnd;              // 2^(sh-1) of the two sources

    // ------------------------------------------------------------------
    // control FSM
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {S_IDLE, S_LDP, S_WTP, S_RUN, S_DRN} state_t;
    state_t st;

    logic [2:0]       ocg;
    logic [6:0]       oy;
    logic [7:0]       ox;
    logic [3:0]       icg;
    logic [1:0]       ky, kx;
    logic [3:0]       tap;
    logic [WDAW-1:0]  w_grp_base, w_addr;
    logic [BAW-1:0]   o_grp_base;
    logic [LAW-1:0]   o_pix;

    logic signed [31:0] bias [PO];
    logic        [14:0] mult [PO];
    logic        [5:0]  shft [PO];
    logic signed [47:0] rnd48 [PO];                // 2^(sh-1) per lane
    logic [PO-1:0]      lane_on;

    logic last_tap, last_icg, last_x, last_y, issue;
    assign last_tap = (tap == kk_m1);
    assign last_icg = (icg == nicg_m1);
    assign last_x   = (ox  == wout_m1);
    assign last_y   = (oy  == hout_m1);
    assign issue    = (st == S_RUN);

    logic pipe_busy;                               // any stage still holds data

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
                w_grp_base <= d.w_base;
                o_grp_base <= d.out_base;
                kk_m1      <= d.k3 ? 4'd8 : 4'd0;
                nicg_m1    <= d.n_icg - 4'd1;
                wout_m1    <= d.wout - 8'd1;
                hout_m1    <= d.hout - 7'd1;
                nocg_m1    <= d.n_ocg - 3'd1;
                a_rnd      <= 24'sd1 <<< (d.a_sh - 6'd1);
                b_rnd      <= 24'sd1 <<< (d.b_sh - 6'd1);
                st         <= S_LDP;
            end
            S_LDP: st <= S_WTP;                    // param ROM address presented
            S_WTP: begin                           // param data valid
                for (int i = 0; i < PO; i++) begin
                    prm_lane_t pl;
                    pl         = prm_lane_t'(prm_rdata[i*PRM_LANE_W +: PRM_LANE_W]);
                    bias[i]    <= pl.bias;
                    mult[i]    <= pl.m[14:0];
                    shft[i]    <= pl.sh[5:0];
                    rnd48[i]   <= 48'sd1 <<< (pl.sh[5:0] - 6'd1);
                    lane_on[i] <= (7'({ocg, 4'b0000}) + 7'(i)) < d.cout;
                end
                {oy, ox, icg, ky, kx, tap} <= '0;
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
                    if (!last_icg) begin
                        icg    <= icg + 4'd1;
                        w_addr <= w_addr + 1'b1;
                    end else begin
                        icg    <= '0;
                        w_addr <= w_grp_base;
                        o_pix  <= o_pix + 1'b1;
                        if (!last_x) ox <= ox + 8'd1;
                        else begin
                            ox <= '0;
                            if (!last_y) oy <= oy + 7'd1;
                            else         st <= S_DRN;
                        end
                    end
                end
            end
            S_DRN: if (!pipe_busy) begin
                if (ocg == nocg_m1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    st   <= S_IDLE;
                end else begin
                    ocg        <= ocg + 3'd1;
                    w_grp_base <= w_grp_base + d.w_per_ocg;
                    o_grp_base <= o_grp_base + BAW'(d.out_hw);
                    st         <= S_LDP;
                end
            end
            default: st <= S_IDLE;
            endcase
        end
    end

    assign prm_re    = (st == S_LDP);
    assign prm_raddr = d.p_base + PDAW'(ocg);

    // ==================================================================
    // input side
    // ==================================================================
    // ---- s1: logical input coordinates, padding, source select
    logic               v1, pad1, srcb1, first1, last1;
    logic signed [9:0]  iy1, ix1;
    logic [6:0]         loc1;
    logic [WDAW-1:0]    wa1;
    logic [BAW-1:0]     oa1;
    logic [LAW-1:0]     op1;

    always_ff @(posedge clk) begin
        logic signed [9:0] iy, ix;
        logic [6:0] ic0;
        iy  = $signed({2'b00, (d.s2 ? {oy, 1'b0} : {1'b0, oy})}) + $signed({8'b0, ky}) - (d.k3 ? 10'sd1 : 10'sd0);
        ix  = $signed({1'b0,  (d.s2 ? {ox, 1'b0} : {1'b0, ox})}) + $signed({8'b0, kx}) - (d.k3 ? 10'sd1 : 10'sd0);
        ic0 = {icg[3:0], 3'b000};
        v1     <= issue && !rst;
        iy1    <= iy;
        ix1    <= ix;
        pad1   <= (iy < 0) || (ix < 0) || (iy >= $signed({3'b0, d.hin})) || (ix >= $signed({2'b0, d.win}));
        srcb1  <= (ic0 >= d.ca);
        loc1   <= (ic0 >= d.ca) ? ic0 - d.ca : ic0;
        first1 <= (tap == 4'd0) && (icg == 4'd0);
        last1  <= last_tap && last_icg;
        wa1    <= w_addr;
        oa1    <= o_grp_base + BAW'(o_pix);
        op1    <= o_pix;
    end

    // ---- s2: address partial products, frame address
    logic               v2, pad2, half2, srcb2, first2, last2;
    logic [BAW-1:0]     base2, tg2, ty2;
    logic [7:0]         rx2;
    logic [FAW-1:0]     fa2;
    logic [WDAW-1:0]    wa2;
    logic [BAW-1:0]     oa2;
    logic [LAW-1:0]     op2;

    always_ff @(posedge clk) begin
        logic [6:0] ry;
        logic [7:0] rx;
        ry = (!srcb1 && d.a_up) ? 7'(iy1[7:1]) : iy1[6:0];
        rx = (!srcb1 && d.a_up) ? 8'(ix1[8:1]) : ix1[7:0];
        base2  <= srcb1 ? d.b_base : d.a_base;
        tg2    <= BAW'(loc1[6:4]) * (srcb1 ? BAW'(d.b_hw) : BAW'(d.a_hw));
        ty2    <= BAW'(ry) * (srcb1 ? BAW'(d.b_w) : BAW'(d.a_w));
        rx2    <= rx;
        fa2    <= (FAW'(iy1[6:0]) << 7) + (FAW'(iy1[6:0]) << 5) + FAW'(ix1[7:0]);   // iy*160 + ix
        v2     <= v1 && !rst;
        pad2   <= pad1;
        half2  <= loc1[3];
        srcb2  <= srcb1;
        first2 <= first1;
        last2  <= last1;
        wa2    <= wa1;
        oa2    <= oa1;
        op2    <= op1;
    end

    // ---- s3: address sum -> registered memory addresses
    logic               v3, pad3, half3, srcb3, first3, last3;
    logic [WDAW-1:0]    wa3;
    logic [BAW-1:0]     oa3;
    logic [LAW-1:0]     op3;

    always_ff @(posedge clk) begin
        act_raddr <= base2 + tg2 + ty2 + BAW'(rx2);
        act_re    <= v2 && !pad2 && !d.src_frame && !rst;
        frm_raddr <= fa2;
        frm_re    <= v2 && !pad2 &&  d.src_frame && !rst;
        v3     <= v2 && !rst;
        pad3   <= pad2;
        half3  <= half2;
        srcb3  <= srcb2;
        first3 <= first2;
        last3  <= last2;
        wa3    <= wa2;
        oa3    <= oa2;
        op3    <= op2;
    end

    // ---- s4, s5: memory latency (flags delayed)
    logic               v4, pad4, half4, srcb4, first4, last4;
    logic               v5, pad5, half5, srcb5, first5, last5;
    logic [WDAW-1:0]    wa4, wa5;
    logic [BAW-1:0]     oa4, oa5;
    logic [LAW-1:0]     op4, op5;

    always_ff @(posedge clk) begin
        {v4, pad4, half4, srcb4, first4, last4} <= {v3 && !rst, pad3, half3, srcb3, first3, last3};
        {v5, pad5, half5, srcb5, first5, last5} <= {v4 && !rst, pad4, half4, srcb4, first4, last4};
        wa4 <= wa3;  oa4 <= oa3;  op4 <= op3;
        wa5 <= wa4;  oa5 <= oa4;  op5 <= op4;
    end

    // ---- s6: data -> 8 lanes (or LUT(frame) on lane 0), padding
    logic               v6, srcb6, first6, last6;
    logic signed [7:0]  x6 [PI];
    logic [WDAW-1:0]    wa6;
    logic [BAW-1:0]     oa6;
    logic [LAW-1:0]     op6;

    always_ff @(posedge clk) begin
        for (int i = 0; i < PI; i++) begin
            if (pad5)             x6[i] <= 8'sd0;
            else if (d.src_frame) x6[i] <= (i == 0) ? $signed(inlut[frm_rdata]) : 8'sd0;
            else                  x6[i] <= $signed(act_rdata[(half5 ? (8 + i) : i)*8 +: 8]);
        end
        v6 <= v5 && !rst;  srcb6 <= srcb5;  first6 <= first5;  last6 <= last5;
        wa6 <= wa5;  oa6 <= oa5;  op6 <= op5;
    end

    // ---- s7: x * M_src
    logic               v7, first7, last7;
    logic signed [23:0] xm7 [PI];
    logic signed [23:0] rnd7;
    logic [5:0]         sh7;
    logic [WDAW-1:0]    wa7;
    logic [BAW-1:0]     oa7;
    logic [LAW-1:0]     op7;

    always_ff @(posedge clk) begin
        for (int i = 0; i < PI; i++)
            xm7[i] <= x6[i] * $signed({1'b0, srcb6 ? d.b_m : d.a_m});
        rnd7 <= srcb6 ? b_rnd : a_rnd;
        sh7  <= srcb6 ? d.b_sh : d.a_sh;
        v7 <= v6 && !rst;  first7 <= first6;  last7 <= last6;
        wa7 <= wa6;  oa7 <= oa6;  op7 <= op6;
    end

    // ---- s8: + 2^(sh-1); weight ROM address (data needed in s11)
    logic               v8, first8, last8;
    logic signed [23:0] xr8 [PI];
    logic [5:0]         sh8;
    logic [BAW-1:0]     oa8;
    logic [LAW-1:0]     op8;

    always_ff @(posedge clk) begin
        for (int i = 0; i < PI; i++)
            xr8[i] <= xm7[i] + rnd7;
        sh8 <= sh7;
        wt_raddr <= wa7;
        wt_re    <= v7 && !rst;
        v8 <= v7 && !rst;  first8 <= first7;  last8 <= last7;
        oa8 <= oa7;  op8 <= op7;
    end

    // ---- s9: >>> sh
    logic               v9, first9, last9;
    logic signed [23:0] xs9 [PI];
    logic [BAW-1:0]     oa9;
    logic [LAW-1:0]     op9;

    always_ff @(posedge clk) begin
        for (int i = 0; i < PI; i++)
            xs9[i] <= xr8[i] >>> sh8;
        v9 <= v8 && !rst;  first9 <= first8;  last9 <= last8;
        oa9 <= oa8;  op9 <= op8;
    end

    // ---- s10: clamp to +-127
    logic               v10, first10, last10;
    logic signed [7:0]  x10 [PI];
    logic [BAW-1:0]     oa10;
    logic [LAW-1:0]     op10;

    always_ff @(posedge clk) begin
        for (int i = 0; i < PI; i++) begin
            if      (xs9[i] >  24'sd127) x10[i] <=  8'sd127;
            else if (xs9[i] < -24'sd127) x10[i] <= -8'sd127;
            else                         x10[i] <= xs9[i][7:0];
        end
        v10 <= v9 && !rst;  first10 <= first9;  last10 <= last9;
        oa10 <= oa9;  op10 <= op9;
    end

    // ==================================================================
    // MAC side
    // ==================================================================
    // ---- s11: 128 products (weight word arrives now)
    logic               v11, first11, last11;
    logic signed [15:0] pr11 [PO][PI];
    logic [BAW-1:0]     oa11;
    logic [LAW-1:0]     op11;

    always_ff @(posedge clk) begin
        for (int p = 0; p < PO; p++)
            for (int i = 0; i < PI; i++)
                pr11[p][i] <= x10[i] * $signed(wt_rdata[(p*PI + i)*8 +: 8]);
        v11 <= v10 && !rst;  first11 <= first10;  last11 <= last10;
        oa11 <= oa10;  op11 <= op10;
    end

    // ---- s12: two 4-input partial sums per lane
    logic               v12, first12, last12;
    logic signed [17:0] ps12 [PO][2];
    logic [BAW-1:0]     oa12;
    logic [LAW-1:0]     op12;

    always_ff @(posedge clk) begin
        for (int p = 0; p < PO; p++) begin
            ps12[p][0] <= 18'(pr11[p][0]) + 18'(pr11[p][1]) + 18'(pr11[p][2]) + 18'(pr11[p][3]);
            ps12[p][1] <= 18'(pr11[p][4]) + 18'(pr11[p][5]) + 18'(pr11[p][6]) + 18'(pr11[p][7]);
        end
        v12 <= v11 && !rst;  first12 <= first11;  last12 <= last11;
        oa12 <= oa11;  op12 <= op11;
    end

    // ---- s13: accumulate; v13 = acc holds a finished pixel
    logic signed [31:0] acc [PO];
    logic               v13;
    logic [BAW-1:0]     oa13;
    logic [LAW-1:0]     op13;

    always_ff @(posedge clk) begin
        if (v12)
            for (int p = 0; p < PO; p++)
                acc[p] <= (first12 ? bias[p] : acc[p]) + 32'(ps12[p][0]) + 32'(ps12[p][1]);
        v13  <= v12 && last12 && !rst;
        oa13 <= oa12;
        op13 <= op12;
    end

    // ==================================================================
    // output requant: (acc*M + 2^(sh-1)) >>> sh, clamp, ReLU
    // acc*M = (acc_hi * M) << 17 + acc_lo * M,  acc_hi = acc[31:17] (signed),
    // acc_lo = acc[16:0] (unsigned): two 18x25-sized products -> one DSP48 each.
    // ==================================================================
    // ---- s14: partial products
    logic signed [31:0] phi14 [PO];
    logic signed [32:0] plo14 [PO];
    logic               v14;
    logic [BAW-1:0]     oa14;
    logic [LAW-1:0]     op14;

    always_ff @(posedge clk) begin
        if (v13)
            for (int p = 0; p < PO; p++) begin
                phi14[p] <= $signed(acc[p][31:17]) * $signed({1'b0, mult[p]});
                plo14[p] <= $signed({1'b0, acc[p][16:0]}) * $signed({1'b0, mult[p]});
            end
        v14 <= v13 && !rst;  oa14 <= oa13;  op14 <= op13;
    end

    // ---- s15: combine
    logic signed [47:0] pm15 [PO];
    logic               v15;
    logic [BAW-1:0]     oa15;
    logic [LAW-1:0]     op15;

    always_ff @(posedge clk) begin
        if (v14)
            for (int p = 0; p < PO; p++)
                pm15[p] <= (48'(phi14[p]) <<< 17) + 48'(plo14[p]);
        v15 <= v14 && !rst;  oa15 <= oa14;  op15 <= op14;
    end

    // ---- s16: + 2^(sh-1)
    logic signed [47:0] pr16 [PO];
    logic               v16;
    logic [BAW-1:0]     oa16;
    logic [LAW-1:0]     op16;

    always_ff @(posedge clk) begin
        if (v15)
            for (int p = 0; p < PO; p++)
                pr16[p] <= pm15[p] + rnd48[p];
        v16 <= v15 && !rst;  oa16 <= oa15;  op16 <= op15;
    end

    // ---- s17: >>> sh
    logic signed [47:0] r17 [PO];
    logic               v17;
    logic [BAW-1:0]     oa17;
    logic [LAW-1:0]     op17;

    always_ff @(posedge clk) begin
        if (v16)
            for (int p = 0; p < PO; p++)
                r17[p] <= pr16[p] >>> shft[p];
        v17 <= v16 && !rst;  oa17 <= oa16;  op17 <= op16;
    end

    // ---- s18: clamp, ReLU -> bank write; argmax input (unused lanes = -128)
    logic signed [7:0]  qa18 [PO];
    logic               v18;
    logic [LAW-1:0]     op18;

    always_ff @(posedge clk) begin
        if (v17)
            for (int p = 0; p < PO; p++) begin
                logic signed [7:0] q;
                if      (r17[p] >  48'sd127) q =  8'sd127;
                else if (r17[p] < -48'sd127) q = -8'sd127;
                else                         q = r17[p][7:0];
                if (d.relu && q[7])          q = 8'sd0;
                act_wdata[p*8 +: 8] <= q;
                qa18[p] <= lane_on[p] ? q : -8'sd128;
            end
        act_we    <= v17 && !rst;
        act_waddr <= oa17;
        act_wmask <= lane_on;
        v18       <= v17 && !rst;
        op18      <= op17;
    end

    // ---- s19, s20: argmax tree, first maximum wins (lower index on ties)
    logic signed [7:0]  av19 [4];
    logic [3:0]         ai19 [4];
    logic               v19;
    logic [LAW-1:0]     op19;

    always_ff @(posedge clk) begin
        for (int g = 0; g < 4; g++) begin
            logic signed [7:0] va, vb;
            logic [3:0]        ia, ib;
            // pairs (4g, 4g+1) and (4g+2, 4g+3), then the two winners
            va = qa18[4*g];     ia = 4'(4*g);
            if (qa18[4*g + 1] > va) begin va = qa18[4*g + 1]; ia = 4'(4*g + 1); end
            vb = qa18[4*g + 2]; ib = 4'(4*g + 2);
            if (qa18[4*g + 3] > vb) begin vb = qa18[4*g + 3]; ib = 4'(4*g + 3); end
            if (vb > va) begin va = vb; ia = ib; end
            av19[g] <= va;
            ai19[g] <= ia;
        end
        v19  <= v18 && !rst;
        op19 <= op18;
    end

    always_ff @(posedge clk) begin
        logic signed [7:0] va, vb;
        logic [3:0]        ia, ib;
        va = av19[0]; ia = ai19[0];
        if (av19[1] > va) begin va = av19[1]; ia = ai19[1]; end
        vb = av19[2]; ib = ai19[2];
        if (av19[3] > vb) begin vb = av19[3]; ib = ai19[3]; end
        if (vb > va) begin va = vb; ia = ib; end
        lab_we    <= v19 && d.argmax && !rst;
        lab_waddr <= op19;
        lab_wdata <= ia[2:0];
    end

    assign pipe_busy = v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | v11 | v12 | v13 |
                       v14 | v15 | v16 | v17 | v18 | v19 | act_we | lab_we;

endmodule
