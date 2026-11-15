// ============================================================================
// unet_pkg.sv - shared constants for the Tiny U-Net FPGA design
//
// Target board : Digilent Nexys4 DDR
// FPGA         : Xilinx Artix-7 XC7A100T-1CSG324C  (Vivado part xc7a100tcsg324-1)
// Clock        : 100 MHz on-board oscillator (pin E3)
// BRAM         : 135 x RAMB36 = 4 860 Kb (607.5 KB)
// DSP48E1      : 240
// ============================================================================
`timescale 1ns/1ps

// package constants are shared; not every module uses all of them
/* verilator lint_off UNUSEDPARAM */
package unet_pkg;

    localparam string FPGA_PART   = "xc7a100tcsg324-1";
    localparam int    CLK_HZ      = 100_000_000;

    // ---- Stage-1 single-lane engine (conv_engine.sv) ----
    localparam int PO   = 16;                  // parallel output-channel lanes
    localparam int LPO  = $clog2(PO);
    localparam int AAW  = 16;                  // activation bank address bits (stage-1 tb)
    localparam int WAW  = 14;
    localparam int PAW  = 8;

    // ---- full accelerator (unet_engine.sv / unet_core.sv) ----
    localparam int PI         = 8;             // parallel input-channel lanes
    localparam int NBANK      = 16;            // activation banks (= PO)
    localparam int BANK_DEPTH = 12000;         // bytes per bank
    localparam int BAW        = 14;            // bank address bits
    localparam int WW         = PO * PI * 8;   // weight word = 1024 bit
    localparam int WDEPTH     = 2048;          // weight words (1613 used)
    localparam int WDAW       = 11;
    localparam int PDEPTH     = 32;            // param words (28 used)
    localparam int PDAW       = 5;

    // ---- camera / label geometry ----
    localparam int IMG_W  = 160;
    localparam int IMG_H  = 120;
    localparam int IMG_N  = IMG_W * IMG_H;     // 19200
    localparam int FAW    = 15;                // frame buffer address bits
    localparam int LAB_W  = 80;
    localparam int LAB_H  = 60;
    localparam int NPIX   = LAB_W * LAB_H;     // 4800
    localparam int LAW    = 13;                // label address bits

    // ---- int8 arithmetic (INT8_SPEC.md) ----
    localparam int QMAX = 127;

    // one lane of a param word: {sh, M, bias}  = 56 bits
    localparam int PRM_LANE_W = 56;
    typedef struct packed {
        logic        [7:0]  sh;                // requant shift
        logic        [15:0] m;                 // requant multiplier (15 bits used)
        logic signed [31:0] bias;              // int32 bias
    } prm_lane_t;

    // ---- post-processing ----
    localparam int N_CLS       = 6;
    localparam int K_MAX       = 63;           // connected components (ids 1..63)
    localparam int MIN_AREA    = 24;           // 0.005 * 4800 (seg_unet min_area)
    localparam int CC_MAX_PASS = 32;
    localparam int N_ZONES     = 64;
    localparam int N_BINS      = 32;
    localparam int NO_ZONE     = 127;
    localparam int TOF_MIN_MM  = 20;
    localparam int TOF_MAX_MM  = 4000;

endpackage
/* verilator lint_on UNUSEDPARAM */
