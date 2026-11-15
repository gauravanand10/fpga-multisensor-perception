// ============================================================================
// uart_tx.sv - UART transmitter (8N1) + the result packet builder
// Target: Nexys4 DDR, USB-UART (FT2232HQ) on pin D4 (UART_RXD_OUT).
//
// uart_tx     : CLKS_PER_BIT = 100 MHz / baud (50 -> 2 Mbaud).
// uart_packer : sends one result packet (format: tools/fpga_model.py uart_packet):
//   A5 5A 46 50 | ver flags cam_no tof_no unet_cycles pp_cycles n_obj |
//   n_obj x 12 B objects | 64 x u16 ToF dist | 64 x u8 ToF status |
//   4800 B object map | [19200 B camera frame] | u16 sum16 (all bytes after the sync)
// ============================================================================
`timescale 1ns/1ps

module uart_tx #(
    parameter int CLKS_PER_BIT = 50
)(
    input  logic       clk,
    input  logic       rst,
    input  logic       valid,
    input  logic [7:0] data,
    output logic       ready,
    output logic       txd
);
    localparam int CW = $clog2(CLKS_PER_BIT + 1);
    logic [CW-1:0] ccnt;
    logic [3:0]    bitn;
    logic [9:0]    shreg;
    logic          active;

    assign ready = !active;

    always_ff @(posedge clk) begin
        if (rst) begin
            active <= 1'b0;
            txd    <= 1'b1;
        end else if (!active) begin
            txd <= 1'b1;
            if (valid) begin
                shreg  <= {1'b1, data, 1'b0};         // stop, data LSB first, start
                active <= 1'b1;
                ccnt   <= '0;
                bitn   <= '0;
                txd    <= 1'b0;
            end
        end else begin
            if (ccnt == CW'(CLKS_PER_BIT - 1)) begin
                ccnt <= '0;
                if (bitn == 4'd9) begin
                    active <= 1'b0;
                    txd    <= 1'b1;
                end else begin
                    bitn <= bitn + 4'd1;
                    txd  <= shreg[bitn + 4'd1];
                end
            end else
                ccnt <= ccnt + 1'b1;
        end
    end
endmodule


module uart_packer
    import unet_pkg::*;
(
    input  logic             clk,
    input  logic             rst,
    input  logic             start,
    output logic             busy,
    output logic             done,

    // header fields (stable while busy)
    input  logic             with_image,
    input  logic [7:0]       flags_in,        // bits 1..3 (bit 0 = image is set here)
    input  logic [15:0]      cam_no,
    input  logic [15:0]      tof_no,
    input  logic [31:0]      unet_cycles,
    input  logic [31:0]      pp_cycles,
    input  logic [5:0]       n_obj,

    // object table (postproc, combinational)
    output logic [5:0]       obj_k,
    input  logic             obj_valid,
    input  logic [2:0]       obj_cls,
    input  logic [12:0]      obj_area,
    input  logic [6:0]       obj_x0, obj_x1,
    input  logic [5:0]       obj_y0, obj_y1,
    input  logic [15:0]      obj_dist,
    input  logic [6:0]       obj_nsel,

    input  logic [16*64-1:0] tof_dist,
    input  logic [8*64-1:0]  tof_status,

    output logic             km_re,           // object map, 2-cycle latency
    output logic [LAW-1:0]   km_raddr,
    input  logic [5:0]       km_rdata,
    output logic             fb_re,           // camera frame, 2-cycle latency
    output logic [FAW-1:0]   fb_raddr,
    input  logic [7:0]       fb_rdata,

    // byte stream to uart_tx
    output logic             tx_valid,
    output logic [7:0]       tx_data,
    input  logic             tx_ready
);
    typedef enum logic [2:0] {SEC_SYNC, SEC_HDR, SEC_OBJ, SEC_TOF, SEC_MAP, SEC_IMG, SEC_SUM} sec_t;
    typedef enum logic [2:0] {B_IDLE, B_PREP, B_FIND, B_WAIT, B_SEND, B_PUSH} bst_t;
    sec_t        sec;
    bst_t        bs;
    logic [14:0] idx;
    logic [3:0]  ob;                  // byte inside an object
    logic [15:0] sum;
    logic [7:0]  cur;
    logic [1:0]  wcnt;

    // header / object / ToF bytes (combinational)
    logic [7:0] hdr_b, obj_b, tof_b;
    always_comb begin
        unique case (idx[3:0])
        4'd0:  hdr_b = 8'd1;                                  // version
        4'd1:  hdr_b = {flags_in[7:1], with_image};
        4'd2:  hdr_b = cam_no[7:0];
        4'd3:  hdr_b = cam_no[15:8];
        4'd4:  hdr_b = tof_no[7:0];
        4'd5:  hdr_b = tof_no[15:8];
        4'd6:  hdr_b = unet_cycles[7:0];
        4'd7:  hdr_b = unet_cycles[15:8];
        4'd8:  hdr_b = unet_cycles[23:16];
        4'd9:  hdr_b = unet_cycles[31:24];
        4'd10: hdr_b = pp_cycles[7:0];
        4'd11: hdr_b = pp_cycles[15:8];
        4'd12: hdr_b = pp_cycles[23:16];
        4'd13: hdr_b = pp_cycles[31:24];
        default: hdr_b = {2'b00, n_obj};
        endcase
        unique case (ob)
        4'd0:  obj_b = {2'b00, obj_k};
        4'd1:  obj_b = {5'b0, obj_cls};
        4'd2:  obj_b = obj_area[7:0];
        4'd3:  obj_b = {3'b0, obj_area[12:8]};
        4'd4:  obj_b = {1'b0, obj_x0};
        4'd5:  obj_b = {2'b0, obj_y0};
        4'd6:  obj_b = {1'b0, obj_x1};
        4'd7:  obj_b = {2'b0, obj_y1};
        4'd8:  obj_b = obj_dist[7:0];
        4'd9:  obj_b = obj_dist[15:8];
        4'd10: obj_b = {1'b0, obj_nsel};
        default: obj_b = 8'd0;
        endcase
        if (idx < 15'd128) tof_b = tof_dist[idx[6:0]*8 +: 8];
        else               tof_b = tof_status[(idx[6:0])*8 +: 8];
    end

    assign km_raddr = LAW'(idx);
    assign fb_raddr = FAW'(idx);
    assign km_re    = (bs == B_PREP) && (sec == SEC_MAP);
    assign fb_re    = (bs == B_PREP) && (sec == SEC_IMG);

    always_ff @(posedge clk) begin
        done     <= 1'b0;
        tx_valid <= 1'b0;
        if (rst) begin
            bs <= B_IDLE; busy <= 1'b0;
        end else begin
            unique case (bs)
            B_IDLE: if (start) begin
                busy <= 1'b1; sec <= SEC_SYNC; idx <= '0; sum <= '0; obj_k <= 6'd1; ob <= '0;
                bs <= B_PREP;
            end
            B_PREP: begin
                unique case (sec)
                SEC_SYNC: begin
                    cur <= (idx == 0) ? 8'hA5 : (idx == 1) ? 8'h5A : (idx == 2) ? 8'h46 : 8'h50;
                    bs  <= B_SEND;
                end
                SEC_HDR: begin cur <= hdr_b; bs <= B_SEND; end
                SEC_OBJ: begin
                    if (ob == 4'd0) bs <= B_FIND;            // find the next valid object
                    else begin cur <= obj_b; bs <= B_SEND; end
                end
                SEC_TOF: begin cur <= tof_b; bs <= B_SEND; end
                SEC_MAP: begin wcnt <= 2'd2; bs <= B_WAIT; end
                SEC_IMG: begin wcnt <= 2'd2; bs <= B_WAIT; end   // frame buffer latency 2
                default: begin                               // SEC_SUM
                    cur <= (idx == 0) ? sum[7:0] : sum[15:8];
                    bs  <= B_SEND;
                end
                endcase
            end
            B_FIND: begin
                if (obj_k == 6'd63 && !obj_valid) begin
                    sec <= SEC_TOF; idx <= '0; bs <= B_PREP;   // no more objects
                end else if (obj_valid) begin
                    cur <= obj_b; bs <= B_SEND;
                end else
                    obj_k <= obj_k + 6'd1;
            end
            B_WAIT: begin
                if (wcnt == 2'd1) begin
                    cur <= (sec == SEC_MAP) ? {2'b00, km_rdata} : fb_rdata;
                    bs  <= B_SEND;
                end
                wcnt <= wcnt - 2'd1;
            end
            B_SEND: if (tx_ready && !tx_valid) begin
                tx_valid <= 1'b1;
                tx_data  <= cur;
                if (sec != SEC_SYNC && sec != SEC_SUM) sum <= sum + 16'(cur);
                bs <= B_PUSH;
            end
            B_PUSH: begin                                    // tx_ready drops next cycle
                bs <= B_PREP;
                unique case (sec)
                SEC_SYNC: if (idx == 15'd3)  begin sec <= SEC_HDR; idx <= '0; end else idx <= idx + 1'b1;
                SEC_HDR:  if (idx == 15'd14) begin
                              sec <= (n_obj != 0) ? SEC_OBJ : SEC_TOF; idx <= '0; ob <= '0; obj_k <= 6'd1;
                          end else idx <= idx + 1'b1;
                SEC_OBJ:  if (ob == 4'd11) begin
                              ob <= '0;
                              if (obj_k == 6'd63) begin sec <= SEC_TOF; idx <= '0; end
                              else obj_k <= obj_k + 6'd1;
                          end else ob <= ob + 4'd1;
                SEC_TOF:  if (idx == 15'd191) begin sec <= SEC_MAP; idx <= '0; end else idx <= idx + 1'b1;
                SEC_MAP:  if (idx == 15'(NPIX - 1)) begin
                              sec <= with_image ? SEC_IMG : SEC_SUM; idx <= '0;
                          end else idx <= idx + 1'b1;
                SEC_IMG:  if (idx == 15'(IMG_N - 1)) begin sec <= SEC_SUM; idx <= '0; end else idx <= idx + 1'b1;
                default:  if (idx == 15'd1) begin bs <= B_IDLE; busy <= 1'b0; done <= 1'b1; end
                          else idx <= idx + 1'b1;
                endcase
            end
            default: bs <= B_IDLE;
            endcase
        end
    end
endmodule
