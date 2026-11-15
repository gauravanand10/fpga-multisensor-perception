// ============================================================================
// postproc.sv - objects + ToF distance per object, on the FPGA
// Target: Nexys4 DDR (Artix-7 XC7A100T-1), 100 MHz.
//
// The integer version of what the original PC pipeline / seg_unet.py do on the PC
// (reference: tools/fpga_model.py, bit-exact):
//
//  1. ZONES : the 64 ToF zones are painted on the 80x60 label grid as rectangles
//             (mem/zone_rects.mem, made from your geo/ code), chosen by the zone's
//             distance bin (handles the 50 mm parallax). z-buffer: nearer (or equal,
//             later) zone wins. depth_cam = (d * rz_q15 + 2^14) >> 15.
//             Valid zone = status 5/6/9/10/12 and 20..4000 mm.
//  2. CC    : 8-connected components of equal class, by forward/backward
//             min-label passes until two passes in a row change nothing.
//  3. STATS : pixel count per component; components with >= 24 px numbered 1..63 in
//             raster order of their first pixel; area, box, class, per-zone counts.
//  4. DIST  : per object: zones covering >= 30 % (else the best zone if > 5 %), then
//             the coverage-weighted 35th percentile of their depths
//             (weights = cnt*4096/zone_area).
//
// TIMING (100 MHz): written for timing first, not speed (typ. 2-4 ms per frame):
//   * every memory address / enable / write data is a REGISTER (set by the FSM)
//   * block RAMs have an output register: a read issued in state S is usable in S+3;
//     LUT RAMs: usable in S+2
//   * a 64-entry array lookup always gets its own cycle before arithmetic uses it
//   * multiplies, minimum trees and the percentile test are split over several states
//
// Block RAM: LC (labels / components), ZZ (zone id / z-buffer).
// LUT RAM  : KM (object map), RA (component areas), CN (per-object zone counts),
//            zone rectangles (ROM).
// ============================================================================
`timescale 1ns/1ps

module postproc
    import unet_pkg::*;
#(
    parameter string RECTS_FILE = "zone_rects.mem",
    parameter string RZ_FILE    = "zone_rz.mem",
    parameter string BINLO_FILE = "zone_binlo.mem"
)(
    input  logic               clk,
    input  logic               rst,

    // label map from the U-Net head layer (written before start)
    input  logic               lab_we,
    input  logic [LAW-1:0]     lab_waddr,
    input  logic [2:0]         lab_wdata,

    input  logic               start,
    input  logic [16*64-1:0]   tof_dist,        // zone i at [i*16 +: 16], mm (ST raw order)
    input  logic [8*64-1:0]    tof_status,      // zone i at [i*8 +: 8]
    output logic               busy,
    output logic               done,            // 1-cycle pulse
    output logic [31:0]        cycles,

    // results (stable from done until the next start)
    output logic [5:0]         n_obj,
    output logic [5:0]         n_comp,
    output logic               flag_overflow,   // more than 63 objects
    output logic               flag_noconv,     // CC not converged after CC_MAX_PASS passes
    input  logic [5:0]         obj_k,           // object id 1..63
    output logic               obj_valid,
    output logic [2:0]         obj_cls,
    output logic [12:0]        obj_area,
    output logic [6:0]         obj_x0, obj_x1,
    output logic [5:0]         obj_y0, obj_y1,
    output logic [15:0]        obj_dist,        // mm, 0 = no ToF zone on it
    output logic [6:0]         obj_nsel,

    input  logic               km_re,           // object map, 2-cycle latency, 0 = none
    input  logic [LAW-1:0]     km_raddr,
    output logic [5:0]         km_rdata
);

    localparam logic [12:0] CMAX = 13'h1FFF;     // "no candidate" for the CC minimum

    // ------------------------------------------------------------------
    // memory control registers (all driven from the FSM, all registered)
    // ------------------------------------------------------------------
    logic           lc_re, lc_we;
    logic [LAW-1:0] lc_raddr, lc_waddr;
    logic [15:0]    lc_wdata, lc_rdata;           // {cls[2:0], comp[12:0]}
    logic           zz_re, zz_we;
    logic [LAW-1:0] zz_raddr, zz_waddr;
    logic [22:0]    zz_wdata, zz_rdata;           // {zid[6:0], zbuf[15:0]}
    logic           ra_re, ra_we;
    logic [LAW-1:0] ra_raddr, ra_waddr;
    logic [12:0]    ra_wdata, ra_rdata;
    logic           km_we, kmi_re;
    logic [LAW-1:0] km_waddr, kmi_raddr;
    logic [5:0]     km_wdata, kmi_rdata;
    logic           cn_re, cn_we;
    logic [11:0]    cn_raddr, cn_waddr;
    logic [12:0]    cn_wdata, cn_rdata;
    logic           rc_re;
    logic [10:0]    rc_addr;
    logic [26:0]    rc_data;

    // LC: written by the U-Net head layer (label, initial component = own index) or by the CC passes
    ram_sdp_ch #(.W(16), .D(NPIX), .CH(1024), .AW(LAW), .OREG(1)) u_lc (
        .clk,
        .we(lab_we | lc_we),
        .waddr(lab_we ? lab_waddr : lc_waddr),
        .wdata(lab_we ? {lab_wdata, (lab_wdata != 3'd0) ? lab_waddr : 13'd0} : lc_wdata),
        .re(lc_re), .raddr(lc_raddr), .rdata(lc_rdata));

    ram_sdp_ch #(.W(23), .D(NPIX), .CH(1024), .AW(LAW), .OREG(1)) u_zz (
        .clk, .we(zz_we), .waddr(zz_waddr), .wdata(zz_wdata),
        .re(zz_re), .raddr(zz_raddr), .rdata(zz_rdata));

    ram_sdp_lut #(.W(13), .D(NPIX), .AW(LAW)) u_ra (
        .clk, .we(ra_we), .waddr(ra_waddr), .wdata(ra_wdata),
        .re(ra_re), .raddr(ra_raddr), .rdata(ra_rdata));

    // KM: internal reads while busy, the UART packer's reads otherwise
    ram_sdp_lut #(.W(6), .D(NPIX), .AW(LAW)) u_km (
        .clk, .we(km_we), .waddr(km_waddr), .wdata(km_wdata),
        .re(busy ? kmi_re : km_re), .raddr(busy ? kmi_raddr : km_raddr), .rdata(kmi_rdata));

    ram_sdp_lut #(.W(13), .D(4096), .AW(12)) u_cn (
        .clk, .we(cn_we), .waddr(cn_waddr), .wdata(cn_wdata),
        .re(cn_re), .raddr(cn_raddr), .rdata(cn_rdata));

    rom_lut #(.W(27), .D(2048), .AW(11), .INIT(RECTS_FILE)) u_rect (
        .clk, .re(rc_re), .addr(rc_addr), .data(rc_data));

    logic [15:0] rz_tab    [N_ZONES];
    logic [15:0] binlo_tab [N_BINS];
    initial begin
        $readmemh(RZ_FILE, rz_tab);
        $readmemh(BINLO_FILE, binlo_tab);
    end

    // ------------------------------------------------------------------
    // per-zone / per-object state
    // ------------------------------------------------------------------
    logic [15:0] dcam  [N_ZONES];
    logic [12:0] zarea [N_ZONES];
    logic [12:0] carea [64];
    logic [2:0]  ccls  [64];
    logic [6:0]  cx0 [64], cx1 [64];
    logic [5:0]  cy0 [64], cy1 [64];
    logic [15:0] cdist [64];
    logic [6:0]  cnsel [64];
    logic [12:0] cz [N_ZONES];
    logic [12:0] wz [N_ZONES];
    logic [63:0] sel;
    logic [63:0] valid_k;

    assign obj_valid = valid_k[obj_k];
    assign obj_cls   = ccls[obj_k];
    assign obj_area  = carea[obj_k];
    assign obj_x0    = cx0[obj_k];
    assign obj_x1    = cx1[obj_k];
    assign obj_y0    = cy0[obj_k];
    assign obj_y1    = cy1[obj_k];
    assign obj_dist  = cdist[obj_k];
    assign obj_nsel  = cnsel[obj_k];

    // object map for the packer: LUT RAM output register, then mask (2 cycles)
    always_ff @(posedge clk) km_rdata <= valid_k[kmi_rdata] ? kmi_rdata : 6'd0;

    function automatic logic st_ok(input logic [7:0] s);
        return (s == 8'd5) || (s == 8'd6) || (s == 8'd9) || (s == 8'd10) || (s == 8'd12);
    endfunction

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    typedef enum logic [6:0] {
        P_IDLE, P_CLR,
        P_Z0, P_Z1, P_Z1B, P_Z2, P_Z3, P_Z4, P_ZP0, P_ZP1, P_ZP2, P_ZP3,
        P_A0, P_A1,
        P_C0, P_C1, P_C2, P_C3, P_C4, P_C5, P_C6, P_C7, P_C8, P_C9, P_C10, P_CEND,
        P_R0, P_R1, P_R2, P_R3, P_R4, P_R5, P_RNEXT,
        P_K0, P_K1, P_K2, P_K3, P_K4, P_K5, P_K6, P_K7, P_K8, P_KNEXT,
        P_D0, P_D1, P_D2, P_D3, P_D4, P_D5, P_D6, P_D7, P_D8, P_D9,
        P_D10, P_D11, P_D12, P_D13, P_D14, P_D15,
        P_FIN
    } pst_t;
    pst_t st;

    logic [LAW-1:0] p;                 // pixel index (raster)
    logic [6:0]     x;
    logic [5:0]     y;

    // zones
    logic [6:0]  zi;
    logic [15:0] zd, rzv, zdc;
    logic [7:0]  zs;
    logic        zok;
    logic [4:0]  zbin;
    logic [N_BINS-1:1] zge;            // zd >= bin_lo[i]
    logic [31:0] zprod;
    logic [6:0]  rx0, rx1;
    logic [5:0]  ry1;
    logic [LAW-1:0] rowstep;
    // zone areas (pipelined pass)
    logic [LAW:0]   ap;                // issue counter 0..NPIX
    logic [2:0]     av;                // read valid pipeline
    logic [6:0]     a_zid;
    logic           a_zv;
    logic [LAW:0]   a_done;
    // CC
    logic        dir;
    logic [5:0]  npass;
    logic        changed, prev_nochange;
    logic [2:0]  c_cls;
    logic [12:0] c_comp;
    logic        r_val;
    logic [15:0] r_word;
    logic [15:0] nb1, nb2, nb3;
    logic [12:0] m01, m23, mA, cand3, cc_min;
    logic        v1n, v2n, v3n, rv_ok;
    // stats
    logic [5:0]  nk;
    logic [5:0]  kk_cur;
    logic [6:0]  zid_p;
    logic        is_root;
    logic [12:0] k_area;
    logic [6:0]  k_x0, k_x1;
    logic [5:0]  k_y0, k_y1;
    // distance
    logic [6:0]  kd;                   // 1..64
    logic [6:0]  zz_i, zj;
    logic [12:0] c_r, a_r;
    logic [16:0] c10, a3;
    logic [25:0] pr1, pr2;
    logic        best_ok;
    logic [5:0]  best;
    logic [12:0] cbest, abest;
    logic [18:0] wsum, ssum;
    logic [23:0] t20, t7;
    logic [15:0] dbest, dj, dz;
    logic        dfound, s_bit;
    logic [12:0] wzr;
    logic [6:0]  nsel_c;
    logic [4:0]  dv_i;
    logic [24:0] dv_n;
    logic [12:0] dv_r, dv_d;
    logic [24:0] dv_q;
    logic [13:0] dv_rn;
    assign dv_rn = {dv_r, dv_n[dv_i]};

    always_ff @(posedge clk) begin
        // memory strobes default off
        lc_re <= 1'b0; lc_we <= 1'b0;
        zz_re <= 1'b0; zz_we <= 1'b0;
        ra_re <= 1'b0; ra_we <= 1'b0;
        km_we <= 1'b0; kmi_re <= 1'b0;
        cn_re <= 1'b0; cn_we <= 1'b0;
        rc_re <= 1'b0;
        done  <= 1'b0;
        if (busy) cycles <= cycles + 32'd1;

        if (rst) begin
            st      <= P_IDLE;
            busy    <= 1'b0;
            valid_k <= '0;
            n_obj   <= '0;
            n_comp  <= '0;
            flag_overflow <= 1'b0;
            flag_noconv   <= 1'b0;
        end else begin
            unique case (st)
            // ====================================================== start / clear
            P_IDLE: if (start) begin
                busy <= 1'b1; cycles <= '0;
                p <= '0;
                valid_k <= '0; n_obj <= '0; n_comp <= '0;
                flag_overflow <= 1'b0; flag_noconv <= 1'b0;
                for (int i = 0; i < N_ZONES; i++) begin zarea[i] <= '0; dcam[i] <= '0; end
                for (int k = 0; k < 64; k++) begin
                    carea[k] <= '0; ccls[k] <= '0; cx0[k] <= 7'd127; cy0[k] <= 6'd63;
                    cx1[k] <= '0; cy1[k] <= '0; cdist[k] <= '0; cnsel[k] <= '0;
                end
                st <= P_CLR;
            end
            P_CLR: begin                         // ZZ := no zone, RA := 0, CN := 0
                zz_we <= 1'b1; zz_waddr <= p; zz_wdata <= {7'(NO_ZONE), 16'hFFFF};
                ra_we <= 1'b1; ra_waddr <= p; ra_wdata <= '0;
                if (p < LAW'(4096)) begin cn_we <= 1'b1; cn_waddr <= 12'(p); cn_wdata <= '0; end
                if (p == LAW'(NPIX - 1)) begin zi <= '0; st <= P_Z0; end
                else p <= p + 1'b1;
            end

            // ====================================================== 1. paint zones
            P_Z0: begin                          // fetch zone zi
                if (zi == 7'd64) begin
                    ap <= '0; av <= '0; a_done <= '0; a_zv <= 1'b0;
                    st <= P_A0;
                end else begin
                    zd  <= tof_dist[zi[5:0]*16 +: 16];
                    zs  <= tof_status[zi[5:0]*8 +: 8];
                    rzv <= rz_tab[zi[5:0]];
                    st  <= P_Z1;
                end
            end
            P_Z1: begin                          // validity, bin compares, d * rz
                for (int i = 1; i < N_BINS; i++) zge[i] <= (zd >= binlo_tab[i]);
                zok   <= st_ok(zs) && zd >= 16'(TOF_MIN_MM) && zd <= 16'(TOF_MAX_MM);
                zprod <= 32'(zd) * 32'(rzv);
                st    <= P_Z1B;
            end
            P_Z1B: begin                         // distance bin = highest i with zd >= bin_lo[i]
                logic [4:0] b;
                b = '0;
                for (int i = 1; i < N_BINS; i++)
                    if (zge[i]) b = 5'(i);
                zbin <= b;
                st   <= P_Z2;
            end
            P_Z2: begin                          // depth_cam, rectangle ROM address
                if (!zok) begin
                    zi <= zi + 1'b1; st <= P_Z0;
                end else begin
                    zdc   <= 16'((zprod + 32'h4000) >> 15);
                    rc_re <= 1'b1; rc_addr <= {zi[5:0], zbin};
                    st    <= P_Z3;
                end
            end
            P_Z3: begin dcam[zi[5:0]] <= zdc; st <= P_Z4; end      // ROM reads this cycle
            P_Z4: begin                          // rectangle data valid
                if (!rc_data[26]) begin
                    zi <= zi + 1'b1; st <= P_Z0;
                end else begin
                    rx0 <= rc_data[6:0]; rx1 <= rc_data[19:13]; ry1 <= rc_data[25:20];
                    x   <= rc_data[6:0]; y   <= rc_data[12:7];
                    p   <= (LAW'(rc_data[12:7]) << 6) + (LAW'(rc_data[12:7]) << 4) + LAW'(rc_data[6:0]);
                    rowstep <= LAW'(LAB_W) - (LAW'(rc_data[19:13]) - LAW'(rc_data[6:0]));
                    st  <= P_ZP0;
                end
            end
            P_ZP0: begin zz_re <= 1'b1; zz_raddr <= p; st <= P_ZP1; end
            P_ZP1: st <= P_ZP2;
            P_ZP2: st <= P_ZP3;
            P_ZP3: begin                         // z-buffer data valid
                if (zdc <= zz_rdata[15:0]) begin
                    zz_we <= 1'b1; zz_waddr <= p; zz_wdata <= {zi, zdc};
                end
                if (x == rx1) begin
                    x <= rx0;
                    p <= p + rowstep;
                    if (y == ry1) begin zi <= zi + 1'b1; st <= P_Z0; end
                    else begin y <= y + 1'b1; st <= P_ZP0; end
                end else begin
                    x <= x + 1'b1; p <= p + 1'b1; st <= P_ZP0;
                end
            end

            // ====================================================== zone areas (pipelined)
            P_A0: begin
                // issue one ZZ read per cycle
                if (ap != (LAW+1)'(NPIX)) begin
                    zz_re <= 1'b1; zz_raddr <= LAW'(ap);
                    ap    <= ap + 1'b1;
                end
                av <= {av[1:0], (ap != (LAW+1)'(NPIX))};
                // data of the read issued 3 cycles ago
                a_zv  <= av[2];
                a_zid <= zz_rdata[22:16];
                if (a_zv) begin
                    if (a_zid != 7'(NO_ZONE)) zarea[a_zid[5:0]] <= zarea[a_zid[5:0]] + 1'b1;
                    a_done <= a_done + 1'b1;
                end
                if (a_done == (LAW+1)'(NPIX)) st <= P_A1;
            end
            P_A1: begin
                p <= '0; x <= '0; y <= '0;
                dir <= 1'b0; npass <= '0; changed <= 1'b0; prev_nochange <= 1'b0; r_val <= 1'b0;
                st <= P_C0;
            end

            // ====================================================== 2. connected components
            P_C0: begin lc_re <= 1'b1; lc_raddr <= p; st <= P_C1; end
            P_C1: begin
                lc_re <= 1'b1;
                lc_raddr <= dir ? p + LAW'(LAB_W + 1) : p - LAW'(LAB_W + 1);   // NW / SE
                v1n <= dir ? (x != 7'(LAB_W - 1)) && (y != 6'(LAB_H - 1)) : (x != 0) && (y != 0);
                v2n <= dir ? (y != 6'(LAB_H - 1)) : (y != 0);
                v3n <= dir ? (x != 0) && (y != 6'(LAB_H - 1)) : (x != 7'(LAB_W - 1)) && (y != 0);
                rv_ok <= dir ? (x != 7'(LAB_W - 1)) : (x != 0);
                st <= P_C2;
            end
            P_C2: begin
                lc_re <= 1'b1;
                lc_raddr <= dir ? p + LAW'(LAB_W) : p - LAW'(LAB_W);           // N / S
                st <= P_C3;
            end
            P_C3: begin                          // C data valid; issue n3
                lc_re <= 1'b1;
                lc_raddr <= dir ? p + LAW'(LAB_W - 1) : p - LAW'(LAB_W - 1);   // NE / SW
                c_cls  <= lc_rdata[15:13];
                c_comp <= lc_rdata[12:0];
                if (lc_rdata[15:13] == 3'd0) begin
                    r_val <= 1'b0;
                    st    <= P_CEND;
                end else
                    st <= P_C4;
            end
            P_C4: begin nb1 <= lc_rdata; st <= P_C5; end
            P_C5: begin nb2 <= lc_rdata; st <= P_C6; end
            P_C6: begin nb3 <= lc_rdata; st <= P_C7; end
            P_C7: begin                          // candidate pairs
                logic [12:0] cr, c1, c2;
                cr  = (r_val && rv_ok && r_word[15:13] == c_cls) ? r_word[12:0] : CMAX;
                c1  = (v1n && nb1[15:13] == c_cls) ? nb1[12:0] : CMAX;
                c2  = (v2n && nb2[15:13] == c_cls) ? nb2[12:0] : CMAX;
                m01 <= (cr < c_comp) ? cr : c_comp;
                m23 <= (c2 < c1) ? c2 : c1;
                cand3 <= (v3n && nb3[15:13] == c_cls) ? nb3[12:0] : CMAX;
                st <= P_C8;
            end
            P_C8: begin mA <= (m23 < m01) ? m23 : m01; st <= P_C9; end
            P_C9: begin cc_min <= (cand3 < mA) ? cand3 : mA; st <= P_C10; end
            P_C10: begin
                if (cc_min != c_comp) begin
                    lc_we <= 1'b1; lc_waddr <= p; lc_wdata <= {c_cls, cc_min};
                    changed <= 1'b1;
                end
                r_val  <= 1'b1;
                r_word <= {c_cls, cc_min};
                st     <= P_CEND;
            end
            P_CEND: begin
                if (dir ? (p != '0) : (p != LAW'(NPIX - 1))) begin
                    if (!dir) begin
                        p <= p + 1'b1;
                        if (x == 7'(LAB_W - 1)) begin x <= '0; y <= y + 1'b1; end else x <= x + 1'b1;
                    end else begin
                        p <= p - 1'b1;
                        if (x == 0) begin x <= 7'(LAB_W - 1); y <= y - 1'b1; end else x <= x - 1'b1;
                    end
                    st <= P_C0;
                end else begin                   // end of a pass
                    npass <= npass + 1'b1;
                    r_val <= 1'b0;
                    if ((!changed && prev_nochange) || npass == 6'(CC_MAX_PASS - 1)) begin
                        flag_noconv <= changed;
                        p <= '0;
                        st <= P_R0;
                    end else begin
                        prev_nochange <= !changed;
                        changed <= 1'b0;
                        dir <= !dir;
                        if (!dir) begin p <= LAW'(NPIX - 1); x <= 7'(LAB_W - 1); y <= 6'(LAB_H - 1); end
                        else      begin p <= '0; x <= '0; y <= '0; end
                        st <= P_C0;
                    end
                end
            end

            // ====================================================== 3a. component areas
            P_R0: begin lc_re <= 1'b1; lc_raddr <= p; st <= P_R1; end
            P_R1: st <= P_R2;
            P_R2: st <= P_R3;
            P_R3: begin                          // LC data valid
                c_comp <= lc_rdata[12:0];
                if (lc_rdata[15:13] != 3'd0) begin
                    ra_re <= 1'b1; ra_raddr <= lc_rdata[12:0];
                    st <= P_R4;
                end else
                    st <= P_RNEXT;
            end
            P_R4: st <= P_R5;
            P_R5: begin                          // RA data valid
                ra_we <= 1'b1; ra_waddr <= c_comp; ra_wdata <= ra_rdata + 1'b1;
                st <= P_RNEXT;
            end
            P_RNEXT: begin
                if (p == LAW'(NPIX - 1)) begin
                    p <= '0; x <= '0; y <= '0; nk <= '0;
                    st <= P_K0;
                end else begin
                    p <= p + 1'b1;
                    st <= P_R0;
                end
            end

            // ====================================================== 3b. numbering + stats
            P_K0: begin
                lc_re <= 1'b1; lc_raddr <= p;
                zz_re <= 1'b1; zz_raddr <= p;
                ra_re <= 1'b1; ra_raddr <= p;
                st <= P_K1;
            end
            P_K1: st <= P_K2;
            P_K2: st <= P_K3;
            P_K3: begin                          // LC / ZZ valid (RA of p valid since K2)
                c_cls   <= lc_rdata[15:13];
                c_comp  <= lc_rdata[12:0];
                zid_p   <= zz_rdata[22:16];
                k_area  <= ra_rdata;
                is_root <= (lc_rdata[12:0] == p);
                if (lc_rdata[15:13] == 3'd0) begin
                    km_we <= 1'b1; km_waddr <= p; km_wdata <= '0;
                    st <= P_KNEXT;
                end else begin
                    kmi_re <= 1'b1; kmi_raddr <= lc_rdata[12:0];   // id of the root (non-root case)
                    st <= P_K4;
                end
            end
            P_K4: st <= P_K5;
            P_K5: begin                          // object id of this pixel
                if (is_root) begin
                    if (k_area < 13'(MIN_AREA))  kk_cur <= '0;
                    else if (nk != 6'(K_MAX)) begin kk_cur <= nk + 1'b1; nk <= nk + 1'b1; end
                    else begin kk_cur <= '0; flag_overflow <= 1'b1; end
                end else
                    kk_cur <= kmi_rdata;
                st <= P_K6;
            end
            P_K6: begin                          // write the map, fetch the object's stats
                km_we <= 1'b1; km_waddr <= p; km_wdata <= kk_cur;
                if (kk_cur != '0) begin
                    k_area <= carea[kk_cur];
                    k_x0 <= cx0[kk_cur]; k_x1 <= cx1[kk_cur];
                    k_y0 <= cy0[kk_cur]; k_y1 <= cy1[kk_cur];
                    if (zid_p != 7'(NO_ZONE)) begin cn_re <= 1'b1; cn_raddr <= {kk_cur, zid_p[5:0]}; end
                    st <= P_K7;
                end else
                    st <= P_KNEXT;
            end
            P_K7: st <= P_K8;
            P_K8: begin                          // update stats (count data valid)
                carea[kk_cur] <= k_area + 1'b1;
                ccls[kk_cur]  <= c_cls;
                if (x < k_x0) cx0[kk_cur] <= x;
                if (x > k_x1) cx1[kk_cur] <= x;
                if (y < k_y0) cy0[kk_cur] <= y;
                if (y > k_y1) cy1[kk_cur] <= y;
                if (zid_p != 7'(NO_ZONE)) begin
                    cn_we <= 1'b1; cn_waddr <= {kk_cur, zid_p[5:0]}; cn_wdata <= cn_rdata + 1'b1;
                end
                st <= P_KNEXT;
            end
            P_KNEXT: begin
                if (p == LAW'(NPIX - 1)) begin
                    n_comp <= nk;
                    kd <= 7'd1;
                    st <= P_D0;
                end else begin
                    p <= p + 1'b1;
                    if (x == 7'(LAB_W - 1)) begin x <= '0; y <= y + 1'b1; end else x <= x + 1'b1;
                    st <= P_K0;
                end
            end

            // ====================================================== 4. distance per object
            P_D0: begin
                if (nk == '0 || kd > 7'(nk)) begin
                    st <= P_FIN;
                end else begin
                    valid_k[kd[5:0]] <= 1'b1;
                    n_obj   <= n_obj + 1'b1;
                    zz_i    <= '0; sel <= '0; best_ok <= 1'b0;
                    st      <= P_D1;
                end
            end
            // ---- coverage of every zone: c = pixels of object kd in zone, a = zone area
            P_D1: begin cn_re <= 1'b1; cn_raddr <= {kd[5:0], zz_i[5:0]}; st <= P_D2; end
            P_D2: begin a_r <= zarea[zz_i[5:0]]; st <= P_D3; end
            P_D3: begin c_r <= cn_rdata; st <= P_D4; end
            P_D4: begin
                cz[zz_i[5:0]] <= c_r;
                c10 <= (17'(c_r) << 3) + (17'(c_r) << 1);        // 10 c
                a3  <= (17'(a_r) << 1) + 17'(a_r);               // 3 a
                pr1 <= 26'(c_r) * 26'(abest);
                pr2 <= 26'(cbest) * 26'(a_r);
                st  <= P_D5;
            end
            P_D5: begin
                if (a_r != 0 && c10 >= a3) sel[zz_i[5:0]] <= 1'b1;
                if (a_r != 0 && c_r != 0 && (!best_ok || pr1 > pr2)) begin
                    best_ok <= 1'b1; best <= zz_i[5:0]; cbest <= c_r; abest <= a_r;
                end
                if (zz_i == 7'd63) st <= P_D6;
                else begin zz_i <= zz_i + 1'b1; st <= P_D1; end
            end
            P_D6: begin                          // fallback: best zone if > 5 %
                if (sel == '0 && best_ok && ((19'(cbest) << 4) + (19'(cbest) << 2)) > 19'(abest))
                    sel[best] <= 1'b1;
                zz_i <= '0; wsum <= '0; nsel_c <= '0;
                st <= P_D7;
            end
            // ---- weights w = (c << 12) / a of the selected zones
            P_D7: begin
                if (zz_i == 7'd64) begin
                    zj <= '0; dfound <= 1'b0; dbest <= 16'hFFFF;
                    st <= P_D10;
                end else if (!sel[zz_i[5:0]]) begin
                    zz_i <= zz_i + 1'b1;
                end else begin
                    dv_n <= {cz[zz_i[5:0]], 12'd0};
                    dv_d <= zarea[zz_i[5:0]];
                    dv_r <= '0; dv_q <= '0; dv_i <= 5'd24;
                    st   <= P_D8;
                end
            end
            P_D8: begin                          // restoring division, 1 bit / clock
                if (dv_rn >= 14'(dv_d)) begin
                    dv_r <= 13'(dv_rn - 14'(dv_d));
                    dv_q <= {dv_q[23:0], 1'b1};
                end else begin
                    dv_r <= dv_rn[12:0];
                    dv_q <= {dv_q[23:0], 1'b0};
                end
                if (dv_i == 5'd0) st <= P_D9;
                else dv_i <= dv_i - 1'b1;
            end
            P_D9: begin
                wz[zz_i[5:0]] <= dv_q[12:0];
                wsum   <= wsum + 19'(dv_q[12:0]);
                nsel_c <= nsel_c + 1'b1;
                zz_i   <= zz_i + 1'b1;
                st     <= P_D7;
            end
            // ---- coverage-weighted 35th percentile: smallest selected depth dj with
            //      sum(w of selected zones with depth <= dj) >= 0.35 * W
            P_D10: begin
                if (zj == 7'd64) st <= P_D15;
                else if (!sel[zj[5:0]]) zj <= zj + 1'b1;
                else begin dj <= dcam[zj[5:0]]; ssum <= '0; zz_i <= '0; st <= P_D11; end
            end
            P_D11: begin
                s_bit <= sel[zz_i[5:0]];
                dz    <= dcam[zz_i[5:0]];
                wzr   <= wz[zz_i[5:0]];
                st    <= P_D12;
            end
            P_D12: begin
                if (s_bit && dz <= dj) ssum <= ssum + 19'(wzr);
                if (zz_i == 7'd63) st <= P_D13;
                else begin zz_i <= zz_i + 1'b1; st <= P_D11; end
            end
            P_D13: begin
                t20 <= (24'(ssum) << 4) + (24'(ssum) << 2);      // 20 S
                t7  <= (24'(wsum) << 3) - 24'(wsum);             // 7 W
                st  <= P_D14;
            end
            P_D14: begin
                if (t20 >= t7 && dj < dbest) begin dbest <= dj; dfound <= 1'b1; end
                zj <= zj + 1'b1;
                st <= P_D10;
            end
            P_D15: begin
                cdist[kd[5:0]] <= dfound ? dbest : 16'd0;
                cnsel[kd[5:0]] <= nsel_c;
                kd <= kd + 1'b1;
                st <= P_D0;
            end

            P_FIN: begin
                busy <= 1'b0;
                done <= 1'b1;
                st   <= P_IDLE;
            end
            default: st <= P_IDLE;
            endcase
        end
    end
endmodule
