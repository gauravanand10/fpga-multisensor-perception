// ============================================================================
// tb_unet_core.sv - full Tiny U-Net on the accelerator vs the golden vectors
//
// Feeds the golden camera frame (mem/tb_frame.mem, 160x120 as the ESP32-CAM sends
// it), runs all 12 conv layers, and after EVERY layer pauses the core and compares
// its output region in the 16 activation banks with the fpga_ref golden tensor
// (post-ReLU where the network has a ReLU). Finally the 80x60 label map is compared
// with argmax(golden logits) (mem/tb_labels.mem).
//
// Run directory must be build/<anything> (the scripts do this):
//   RTL init files  : ../../mem/
//   golden vectors  : ../../../golden_model/data/golden_vectors/mem/   (+MEMDIR= overrides)
// ============================================================================
`timescale 1ns/1ps

module tb_unet_core;
    import unet_pkg::*;
    import unet_layers_pkg::*;

    parameter string RTL_MEM = "../../hw/mem/";

    logic clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;                              // 100 MHz

    // ---------------- frame buffer model ----------------
    logic [7:0] frame [IMG_N];
    logic       frm_re;
    logic [FAW-1:0] frm_raddr;
    logic [7:0] frm_rdata;
    logic [7:0] frm_q;                                  // 2-cycle latency, like cam_rx
    always @(posedge clk) begin
        if (frm_re) frm_q <= frame[frm_raddr];
        frm_rdata <= frm_q;
    end

    // ---------------- label capture ----------------
    logic [2:0] labels [NPIX];
    logic       lab_we;
    logic [LAW-1:0] lab_waddr;
    logic [2:0] lab_wdata;
    int         n_lab = 0;
    always @(posedge clk) if (lab_we) begin labels[lab_waddr] <= lab_wdata; n_lab <= n_lab + 1; end

    // ---------------- DUT ----------------
    logic start = 1'b0, busy, done, dbg_go = 1'b0, layer_done;
    logic [31:0] cycles;
    logic [3:0]  layer_idx;
    logic [BAW-1:0] dbg_raddr = '0;
    logic [NBANK*8-1:0] dbg_rdata;

    unet_core #(
        .WEIGHTS_FILE({RTL_MEM, "unet_weights.mem"}),
        .PARAMS_FILE ({RTL_MEM, "unet_params.mem"}),
        .INLUT_FILE  ({RTL_MEM, "unet_inlut.mem"})
    ) dut (
        .clk, .rst, .start, .busy, .done, .cycles,
        .frm_re, .frm_raddr, .frm_rdata,
        .lab_we, .lab_waddr, .lab_wdata,
        .dbg_pause(1'b1), .dbg_go, .layer_done, .layer_idx, .dbg_raddr, .dbg_rdata);

    // ---------------- golden ----------------
    string memdir;
    string check_name [N_LAYERS] = '{
        "L001_Relu_out0", "L003_Relu_out0", "L005_Relu_out0", "L007_Relu_out0",
        "L009_Relu_out0", "L011_Relu_out0", "L013_Relu_out0", "L017_Relu_out0",
        "L019_Relu_out0", "L023_Relu_out0", "L025_Relu_out0", "L026_Conv_out0"};
    logic [7:0] gold [80*60*16];
    logic [3:0] gold_lab [NPIX];
    int  total_err = 0, layers_ok = 0;
    int  t_start, t_layer;

    function automatic string find_golden();
        string cand [3];
        int fd;
        cand[0] = "../../../golden_model/data/golden_vectors/mem/";
        cand[1] = "../../../../../../golden_model/data/golden_vectors/mem/";
        cand[2] = "../../golden_model/data/golden_vectors/mem/";
        for (int k = 0; k < 3; k++) begin
            fd = $fopen({cand[k], "input_q.mem"}, "r");
            if (fd != 0) begin $fclose(fd); return cand[k]; end
        end
        return "";
    endfunction

    task automatic check_layer(input int li);
        layer_t L;
        int err = 0, n;
        logic signed [7:0] got, want;
        L = layer_desc(4'(li));
        n = int'(L.cout) * int'(L.out_hw);
        $readmemh({memdir, check_name[li], ".mem"}, gold, 0, n - 1);
        for (int g = 0; g < int'(L.n_ocg); g++)
            for (int p = 0; p < int'(L.out_hw); p++) begin
                @(negedge clk) dbg_raddr = BAW'(int'(L.out_base) + g*int'(L.out_hw) + p);
                @(negedge clk);
                @(negedge clk);                              // bank read latency 2
                for (int b = 0; b < NBANK; b++) begin
                    int oc;
                    oc = g*NBANK + b;
                    if (oc < int'(L.cout)) begin
                        got  = dbg_rdata[b*8 +: 8];
                        want = gold[oc*int'(L.out_hw) + p];
                        if (got !== want) begin
                            if (err < 5) $display("   MISMATCH oc=%0d pix=%0d got=%0d want=%0d", oc, p, got, want);
                            err++;
                        end
                    end
                end
            end
        total_err += err;
        if (err == 0) layers_ok++;
        $display("layer %2d  %-16s %3dx%2dx%2d  cycles=%7d  %s", li, check_name[li],
                 L.cout, L.hout, L.wout, cycles - t_layer, (err == 0) ? "BIT-EXACT" : "FAIL");
        t_layer = cycles;
    endtask

    initial begin
        int lab_err;
        lab_err = 0;
        memdir = "";
        if (!$value$plusargs("MEMDIR=%s", memdir)) memdir = find_golden();
        if (memdir == "") begin
            $display("ERROR: golden .mem folder not found (pass +MEMDIR=<path>/)");
            $finish;
        end
        $readmemh({RTL_MEM, "tb_frame.mem"}, frame);
        $readmemh({RTL_MEM, "tb_labels.mem"}, gold_lab);
        $display("target     : Nexys4 DDR, %s, %0d MHz", FPGA_PART, CLK_HZ / 1_000_000);
        $display("golden dir : %s", memdir);
        $display("Tiny U-Net on the accelerator (%0d x %0d MAC lanes)", PO, PI);
        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (3) @(posedge clk);
        @(posedge clk) start <= 1'b1;
        @(posedge clk) start <= 1'b0;
        t_layer = 0;
        for (int li = 0; li < N_LAYERS; li++) begin
            @(posedge layer_done);
            check_layer(li);
            @(negedge clk) dbg_go = 1'b1;
            @(negedge clk) dbg_go = 1'b0;
        end
        @(posedge done);
        @(posedge clk);
        for (int p = 0; p < NPIX; p++)
            if ({1'b0, labels[p]} !== gold_lab[p]) lab_err++;
        $display("label map 80x60: %0d writes, %0d mismatches  %s", n_lab, lab_err,
                 (lab_err == 0 && n_lab == NPIX) ? "BIT-EXACT" : "FAIL");
        $display("U-Net cycles (incl. debug pauses excluded): %0d  -> %0.2f ms at 100 MHz",
                 cycles, real'(cycles) / 100000.0);
        $display("--------------------------------------------------");
        if (total_err == 0 && lab_err == 0 && n_lab == NPIX && layers_ok == N_LAYERS)
            $display("PASS: %0d/%0d layers + label map bit-exact", layers_ok, N_LAYERS);
        else
            $display("FAIL: %0d layer mismatches, %0d label mismatches", total_err, lab_err);
        $finish;
    end

    initial begin
        #1s;
        $display("TIMEOUT");
        $finish;
    end
endmodule
