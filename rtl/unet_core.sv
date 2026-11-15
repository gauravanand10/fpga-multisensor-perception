// ============================================================================
// unet_core.sv - the whole Tiny U-Net: layer sequencer + engine + on-chip memories
// Target: Nexys4 DDR (Artix-7 XC7A100T), 100 MHz.
//
//   start -> runs the 12 conv layers of unet_layers_pkg (12 ->  ~1.4 M cycles)
//   input : camera frame through frm_* (external frame buffer, 1-cycle read)
//   output: 80x60 label map through lab_* (written by the head layer, raster order)
//
// Memories inside:
//   16 activation banks  12000 x 8     (48 RAMB36, 3 x 4K x 9 each)
//   weights              2048 x 1024   (~58 RAMB36, init unet_weights.mem)
//   params               32 x 896      (LUT ROM,    init unet_params.mem)
//   input LUT            256 x 8       (LUT ROM,    init unet_inlut.mem)
//
// Debug (simulation): with dbg_pause = 1 the sequencer stops after each layer
// (layer_done pulse, layer_idx) until dbg_go; meanwhile dbg_raddr reads all 16
// banks (dbg_rdata, 2-cycle latency). Tie dbg_pause = 0 on the board.
// ============================================================================
`timescale 1ns/1ps

module unet_core
    import unet_pkg::*;
    import unet_layers_pkg::*;
#(
    parameter string WEIGHTS_FILE = "unet_weights.mem",
    parameter string PARAMS_FILE  = "unet_params.mem",
    parameter string INLUT_FILE   = "unet_inlut.mem"
)(
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 start,
    output logic                 busy,
    output logic                 done,              // 1-cycle pulse
    output logic [31:0]          cycles,            // cycles of the last run

    output logic                 frm_re,
    output logic [FAW-1:0]       frm_raddr,
    input  logic [7:0]           frm_rdata,

    output logic                 lab_we,
    output logic [LAW-1:0]       lab_waddr,
    output logic [2:0]           lab_wdata,

    input  logic                 dbg_pause,
    input  logic                 dbg_go,
    output logic                 layer_done,
    output logic [3:0]           layer_idx,
    input  logic [BAW-1:0]       dbg_raddr,
    output logic [NBANK*8-1:0]   dbg_rdata
);

    // ---------------- sequencer ----------------
    typedef enum logic [2:0] {Q_IDLE, Q_START, Q_RUN, Q_PAUSE, Q_NEXT} qst_t;
    qst_t   q;
    logic [3:0] li;
    layer_t     desc;
    logic       e_start, e_busy, e_done;

    always_ff @(posedge clk) begin
        e_start    <= 1'b0;
        done       <= 1'b0;
        layer_done <= 1'b0;
        if (busy && q != Q_PAUSE) cycles <= cycles + 32'd1;
        if (rst) begin
            q    <= Q_IDLE;
            busy <= 1'b0;
            li   <= '0;
        end else begin
            unique case (q)
            Q_IDLE: if (start) begin
                busy   <= 1'b1;
                cycles <= '0;
                li     <= '0;
                q      <= Q_START;
            end
            Q_START: begin
                desc    <= layer_desc(li);
                q       <= Q_RUN;
                e_start <= 1'b1;            // desc is registered this cycle; engine samples next cycle
            end
            Q_RUN: if (e_done) begin
                layer_done <= 1'b1;
                q          <= dbg_pause ? Q_PAUSE : Q_NEXT;
            end
            Q_PAUSE: if (dbg_go) q <= Q_NEXT;
            Q_NEXT: begin
                if (li == 4'(N_LAYERS - 1)) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    q    <= Q_IDLE;
                end else begin
                    li <= li + 4'd1;
                    q  <= Q_START;
                end
            end
            default: q <= Q_IDLE;
            endcase
        end
    end
    assign layer_idx = li;

    // engine start one cycle after desc is loaded
    logic e_start_d;
    always_ff @(posedge clk) e_start_d <= e_start && !rst;

    // ---------------- engine ----------------
    logic                    act_re, act_we;
    logic [BAW-1:0]          act_raddr, act_waddr;
    logic [NBANK*8-1:0]      act_rdata, act_wdata;
    logic [NBANK-1:0]        act_wmask;
    logic                    wt_re, prm_re;
    logic [WDAW-1:0]         wt_raddr;
    logic [PDAW-1:0]         prm_raddr;
    logic [WW-1:0]           wt_rdata;
    logic [PO*PRM_LANE_W-1:0] prm_rdata;

    unet_engine #(.INLUT_FILE(INLUT_FILE)) u_eng (
        .clk, .rst,
        .start(e_start_d), .d(desc), .busy(e_busy), .done(e_done),
        .act_re, .act_raddr, .act_rdata, .act_we, .act_wmask, .act_waddr, .act_wdata,
        .frm_re, .frm_raddr, .frm_rdata,
        .wt_re, .wt_raddr, .wt_rdata, .prm_re, .prm_raddr, .prm_rdata,
        .lab_we, .lab_waddr, .lab_wdata);

    // ---------------- activation banks ----------------
    logic           bank_re;
    logic [BAW-1:0] bank_raddr;
    assign bank_re    = e_busy ? act_re    : 1'b1;
    assign bank_raddr = e_busy ? act_raddr : dbg_raddr;

    for (genvar b = 0; b < NBANK; b++) begin : g_bank
        ram_sdp_ch #(.W(8), .D(BANK_DEPTH), .CH(4096), .AW(BAW), .OREG(1)) u_ram (   // 3 RAMB36, latency 2
            .clk,
            .we(act_we && act_wmask[b]), .waddr(act_waddr), .wdata(act_wdata[b*8 +: 8]),
            .re(bank_re), .raddr(bank_raddr), .rdata(act_rdata[b*8 +: 8]));
    end
    assign dbg_rdata = act_rdata;

    // ---------------- weights / params ----------------
    rom_sp  #(.W(WW), .D(WDEPTH), .AW(WDAW), .INIT(WEIGHTS_FILE), .OREG(1)) u_wrom (   // latency 2
        .clk, .re(wt_re), .addr(wt_raddr), .data(wt_rdata));
    rom_lut #(.W(PO*PRM_LANE_W), .D(PDEPTH), .AW(PDAW), .INIT(PARAMS_FILE)) u_prom (
        .clk, .re(prm_re), .addr(prm_raddr), .data(prm_rdata));

endmodule
