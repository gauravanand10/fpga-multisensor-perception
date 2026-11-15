// ============================================================================
// unet_system.sv - everything on the FPGA, without board pin names
// Target: Nexys4 DDR (Artix-7 XC7A100T), 100 MHz.
//
//   ESP32-CAM --SPI(JA)--> cam_rx  (double frame buffer, 160x120 grey)
//   ToF ESP32 --SPI(JB)--> tof_rx  (64 distances + 64 status)
//   frame ready -> unet_core (12 layers, 80x60 labels) -> postproc (objects, distances)
//   -> uart_packer --+
//   IMU samples -> imu_fifo -> imu_packer --+-> uart_arbiter -> uart_tx -> USB-UART -> PC
//                                  (sw/fpga_dashboard.py, sw/result_viewer.py)
//
// IMU / servo extension: the result packet (version 1) is unchanged. IMU/servo packets
// (A5 5A 49 4D, rtl/imu_packer.sv) are sent only between result packets.
//
// One frame at a time: lock the newest camera frame + snapshot the newest ToF
// frame, run the network, post-process, send, repeat.
// ============================================================================
`timescale 1ns/1ps

module unet_system
    import unet_pkg::*;
#(
    parameter int    CLKS_PER_BIT = 50,          // 100 MHz / 2 Mbaud
    parameter string MEM_DIR      = ""           // prefix for the .mem init files
)(
    input  logic        clk,
    input  logic        rst,
    // camera SPI (ESP32-CAM is master)
    input  logic        cam_sclk, cam_mosi, cam_csn,
    // ToF SPI (ToF ESP32 is master)
    input  logic        tof_sclk, tof_mosi, tof_csn,
    // UART to the PC
    output logic        uart_txd,
    // switches / LEDs
    input  logic        sw_image,                // 1 = include the camera frame in each packet
    output logic [15:0] led,
    // IMU samples (imu_spi) and servo state (servo_pwm / servo_ctrl), for the IMU packets
    input  logic         imu_sample_valid,
    input  logic [207:0] imu_sample,
    input  logic [1:0]   imu_ok,
    input  logic [7:0]   imu_who1,
    input  logic [7:0]   imu_who2,
    input  logic [7:0]   servo_mode,
    input  logic [15:0]  servo_target_us,
    input  logic [15:0]  servo_now_us
);
    // ---------------- receivers ----------------
    logic           cam_take, cam_ready, cam_buf;
    logic [15:0]    cam_fno, cam_ok, cam_err;
    logic           fb_re;
    logic [FAW-1:0] fb_raddr;
    logic [7:0]     fb_rdata;
    cam_rx u_cam (.clk, .rst, .sclk_i(cam_sclk), .mosi_i(cam_mosi), .csn_i(cam_csn),
                  .take(cam_take), .ready(cam_ready), .cur_buf(cam_buf), .cur_frame_no(cam_fno),
                  .rd_re(fb_re), .rd_addr(fb_raddr), .rd_data(fb_rdata),
                  .n_ok(cam_ok), .n_err(cam_err));

    logic             tof_valid;
    logic [15:0]      tof_fno, tof_ok, tof_err;
    logic [16*64-1:0] tof_dist;
    logic [8*64-1:0]  tof_status;
    tof_rx u_tof (.clk, .rst, .sclk_i(tof_sclk), .mosi_i(tof_mosi), .csn_i(tof_csn),
                  .valid(tof_valid), .frame_no(tof_fno), .dist_mm(tof_dist), .status(tof_status),
                  .n_ok(tof_ok), .n_err(tof_err));

    // ToF snapshot used for this frame
    logic [16*64-1:0] s_dist;
    logic [8*64-1:0]  s_status;
    logic [15:0]      s_tof_no;
    logic             s_tof_ok;

    // ---------------- U-Net ----------------
    logic           un_start, un_busy, un_done;
    logic [31:0]    un_cycles;
    logic           un_frm_re;
    logic [FAW-1:0] un_frm_raddr;
    logic           lab_we;
    logic [LAW-1:0] lab_waddr;
    logic [2:0]     lab_wdata;
    /* verilator lint_off UNUSEDSIGNAL */
    logic           layer_done;
    logic [3:0]     layer_idx;
    logic [NBANK*8-1:0] dbg_rdata;
    /* verilator lint_on UNUSEDSIGNAL */
    unet_core #(
        .WEIGHTS_FILE({MEM_DIR, "unet_weights.mem"}),
        .PARAMS_FILE ({MEM_DIR, "unet_params.mem"}),
        .INLUT_FILE  ({MEM_DIR, "unet_inlut.mem"})
    ) u_core (
        .clk, .rst, .start(un_start), .busy(un_busy), .done(un_done), .cycles(un_cycles),
        .frm_re(un_frm_re), .frm_raddr(un_frm_raddr), .frm_rdata(fb_rdata),
        .lab_we, .lab_waddr, .lab_wdata,
        .dbg_pause(1'b0), .dbg_go(1'b0), .layer_done, .layer_idx,
        .dbg_raddr('0), .dbg_rdata);

    // ---------------- post-processing ----------------
    logic           pp_start, pp_busy, pp_done, pp_ovf, pp_noconv;
    logic [31:0]    pp_cycles;
    logic [5:0]     n_obj, n_comp, obj_k;
    logic           obj_valid;
    logic [2:0]     obj_cls;
    logic [12:0]    obj_area;
    logic [6:0]     obj_x0, obj_x1, obj_nsel;
    logic [5:0]     obj_y0, obj_y1;
    logic [15:0]    obj_dist;
    logic           km_re;
    logic [LAW-1:0] km_raddr;
    logic [5:0]     km_rdata;
    postproc #(
        .RECTS_FILE({MEM_DIR, "zone_rects.mem"}),
        .RZ_FILE   ({MEM_DIR, "zone_rz.mem"}),
        .BINLO_FILE({MEM_DIR, "zone_binlo.mem"})
    ) u_pp (
        .clk, .rst, .lab_we, .lab_waddr, .lab_wdata,
        .start(pp_start), .tof_dist(s_dist), .tof_status(s_status),
        .busy(pp_busy), .done(pp_done), .cycles(pp_cycles),
        .n_obj, .n_comp, .flag_overflow(pp_ovf), .flag_noconv(pp_noconv),
        .obj_k, .obj_valid, .obj_cls, .obj_area, .obj_x0, .obj_x1, .obj_y0, .obj_y1,
        .obj_dist, .obj_nsel, .km_re, .km_raddr, .km_rdata);

    // ---------------- UART ----------------
    logic           pk_start, pk_busy, pk_done, pk_fb_re;
    logic [FAW-1:0] pk_fb_raddr;
    logic           tx_valid, tx_ready;
    logic [7:0]     tx_data;
    logic           pk_tx_valid;
    logic [7:0]     pk_tx_data;
    logic           s_image;
    uart_packer u_pk (
        .clk, .rst, .start(pk_start), .busy(pk_busy), .done(pk_done),
        .with_image(s_image), .flags_in({4'b0, pp_noconv, pp_ovf, s_tof_ok, 1'b0}),
        .cam_no(cam_fno), .tof_no(s_tof_no), .unet_cycles(un_cycles), .pp_cycles(pp_cycles), .n_obj,
        .obj_k, .obj_valid, .obj_cls, .obj_area, .obj_x0, .obj_x1, .obj_y0, .obj_y1, .obj_dist, .obj_nsel,
        .tof_dist(s_dist), .tof_status(s_status),
        .km_re, .km_raddr, .km_rdata, .fb_re(pk_fb_re), .fb_raddr(pk_fb_raddr), .fb_rdata,
        .tx_valid(pk_tx_valid), .tx_data(pk_tx_data), .tx_ready);

    // ---------------- IMU packets + UART arbiter ----------------
    logic           if_pop, if_ovf, if_ovf_clr;
    logic [207:0]   if_q;
    logic [5:0]     if_count;
    imu_fifo u_ififo (.clk, .rst, .wr(imu_sample_valid), .wdata(imu_sample), .pop(if_pop), .q(if_q),
                      .count(if_count), .ovf(if_ovf), .ovf_clr(if_ovf_clr));
    logic           im_start, im_busy, im_done, im_tx_valid;
    logic [7:0]     im_tx_data;
    imu_packer u_ipk (.clk, .rst, .start(im_start), .busy(im_busy), .done(im_done),
        .fifo_count(if_count), .fifo_q(if_q), .fifo_pop(if_pop), .fifo_ovf(if_ovf), .ovf_clr(if_ovf_clr),
        .imu_ok, .who1(imu_who1), .who2(imu_who2), .servo_mode, .servo_target_us, .servo_now_us,
        .tx_valid(im_tx_valid), .tx_data(im_tx_data), .tx_ready);
    logic           res_req, sel_imu;
    uart_arbiter u_arb (.clk, .rst,
        .res_req, .res_start(pk_start), .res_done(pk_done), .res_valid(pk_tx_valid), .res_data(pk_tx_data),
        .imu_req(if_count != 6'd0), .imu_start(im_start), .imu_done(im_done),
        .imu_valid(im_tx_valid), .imu_data(im_tx_data),
        .tx_valid, .tx_data, .sel_imu);

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk, .rst, .valid(tx_valid), .data(tx_data), .ready(tx_ready), .txd(uart_txd));

    assign fb_re    = un_busy ? un_frm_re    : pk_fb_re;
    assign fb_raddr = un_busy ? un_frm_raddr : pk_fb_raddr;

    // ---------------- control ----------------
    typedef enum logic [2:0] {C_IDLE, C_TAKE, C_NET, C_PP, C_SEND, C_REQ} cst_t;
    cst_t cs;
    logic [15:0] n_frames;

    always_ff @(posedge clk) begin
        cam_take <= 1'b0;
        un_start <= 1'b0;
        pp_start <= 1'b0;
        if (rst) begin
            cs <= C_IDLE; n_frames <= '0;
        end else begin
            unique case (cs)
            C_IDLE: if (cam_ready) begin
                cam_take <= 1'b1;
                s_dist   <= tof_dist;
                s_status <= tof_status;
                s_tof_no <= tof_fno;
                s_tof_ok <= tof_valid;
                s_image  <= sw_image;
                cs <= C_TAKE;
            end
            C_TAKE: begin                       // cur_buf switches this cycle
                un_start <= 1'b1;
                cs <= C_NET;
            end
            C_NET: if (un_done) begin pp_start <= 1'b1; cs <= C_PP; end
            C_PP:  if (pp_done) cs <= C_REQ;
            C_REQ: if (pk_start) cs <= C_SEND;  // arbiter: after a running IMU packet
            C_SEND: if (pk_done) begin n_frames <= n_frames + 16'd1; cs <= C_IDLE; end
            default: cs <= C_IDLE;
            endcase
        end
    end

    assign res_req = (cs == C_REQ);

    // ---------------- LEDs ----------------
    //  0 heartbeat (~1.5 Hz)   1 camera packets (toggles)   2 ToF packets (toggles)
    //  3 U-Net running         4 post-processing           5 UART sending
    //  6 camera packet error   7 ToF packet error (sticky until reset)
    //  8..13 objects in the last frame   14 >63 objects   15 frames sent (toggles)
    logic [26:0] hb;
    logic        cam_err_s, tof_err_s;
    logic [5:0]  last_nobj;
    logic        last_ovf;
    always_ff @(posedge clk) begin
        hb <= hb + 27'd1;
        if (rst) begin cam_err_s <= 1'b0; tof_err_s <= 1'b0; last_nobj <= '0; last_ovf <= 1'b0; hb <= '0; end
        else begin
            if (cam_err != 0) cam_err_s <= 1'b1;
            if (tof_err != 0) tof_err_s <= 1'b1;
            if (pp_done) begin last_nobj <= n_obj; last_ovf <= pp_ovf; end
        end
    end
    assign led = {n_frames[0], last_ovf, last_nobj, tof_err_s, cam_err_s, pk_busy, pp_busy, un_busy,
                  tof_ok[0], cam_ok[0], hb[26]};

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused;
    assign unused = ^{cam_buf, n_comp, cam_ok, tof_ok, im_busy, sel_imu};
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
