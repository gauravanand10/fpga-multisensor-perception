// ============================================================================
// tb_conv_engine.sv - Stage 1 testbench (SystemVerilog)
//
// Loads the golden $readmemh vectors written by fpga_ref\run_ref.py
// (golden_model\data\golden_vectors\mem), runs conv_engine on single
// layers and compares every output byte with the golden _out0.mem.
//
//   T1  input_q        L000 3x3 s2  1->16, no ReLU   vs L000_Conv_out0
//   T2  input_q        L000 3x3 s2  1->16, ReLU      vs L001_Relu_out0
//   T3  L001_Relu_out0 L002 3x3 s1 16->16, ReLU      vs L003_Relu_out0
//   T4  L003_Relu_out0 L004 3x3 s2 16->32, no ReLU   vs L004_Conv_out0
//   T5  L025_Relu_out0 L026 1x1 s1 16->6 (head)      vs L026_Conv_out0
//   N1  negative control (ReLU output vs pre-ReLU golden) - must FAIL
//
// Golden directory: found automatically relative to the run directory,
// or override with +MEMDIR=<path ending in />
// ============================================================================
`timescale 1ns/1ps

module tb_conv_engine;
    import unet_pkg::*;

    localparam int DEPTH   = 1 << AAW;
    localparam int OUTBASE = 32768;

    logic clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;                      // 100 MHz, as on the Nexys4 DDR

    // ---------------- behavioural BRAMs (1-cycle read) ----------------
    logic [7:0]               act  [PO*DEPTH];    // bank*DEPTH + addr
    logic [PO*8-1:0]          wmem [1<<WAW];
    logic [PO*PRM_LANE_W-1:0] pmem [1<<PAW];

    // flat C-order arrays filled by $readmemh
    logic [7:0]  f_in  [131072];
    logic [7:0]  f_exp [131072];
    logic [7:0]  f_w   [131072];
    logic [31:0] f_b   [256];
    logic [31:0] f_M   [256];
    logic [31:0] f_sh  [256];

    // ---------------- DUT ----------------
    logic        start = 1'b0;
    logic [7:0]  cfg_cin, cfg_cout;
    logic [8:0]  cfg_hin, cfg_win, cfg_hout, cfg_wout;
    logic        cfg_k3, cfg_s2, cfg_relu;
    logic        busy, done;
    logic        act_rd_en, wt_rd_en, prm_rd_en, out_wr_en;
    logic [LPO-1:0]           act_rd_bank;
    logic [AAW-1:0]           act_rd_addr, out_wr_addr;
    logic [WAW-1:0]           wt_rd_addr;
    logic [PAW-1:0]           prm_rd_addr;
    logic [PO-1:0]            out_wr_mask;
    logic [PO*8-1:0]          out_wr_data;
    logic [7:0]               act_rd_data;
    logic [PO*8-1:0]          wt_rd_data;
    logic [PO*PRM_LANE_W-1:0] prm_rd_data;

    conv_engine dut (
        .clk, .rst, .start,
        .cfg_cin, .cfg_cout, .cfg_hin, .cfg_win, .cfg_hout, .cfg_wout,
        .cfg_k3, .cfg_s2, .cfg_relu,
        .cfg_in_base ('0),
        .cfg_out_base(AAW'(OUTBASE)),
        .cfg_w_base  ('0),
        .cfg_p_base  ('0),
        .busy, .done,
        .act_rd_en, .act_rd_bank, .act_rd_addr, .act_rd_data,
        .wt_rd_en,  .wt_rd_addr,  .wt_rd_data,
        .prm_rd_en, .prm_rd_addr, .prm_rd_data,
        .out_wr_en, .out_wr_mask, .out_wr_addr, .out_wr_data);

    // plain 'always' (not always_ff): the test task also writes act[] directly
    always @(posedge clk) begin
        act_rd_data <= act_rd_en ? act[int'(act_rd_bank)*DEPTH + int'(act_rd_addr)]
                                 : 8'hxx;      // padding: engine must ignore it
        if (wt_rd_en)  wt_rd_data  <= wmem[wt_rd_addr];
        if (prm_rd_en) prm_rd_data <= pmem[prm_rd_addr];
        if (out_wr_en)
            for (int p = 0; p < PO; p++)
                if (out_wr_mask[p]) act[p*DEPTH + int'(out_wr_addr)] <= out_wr_data[p*8 +: 8];
    end

    // ---------------- test bookkeeping ----------------
    string memdir;
    int    total_err = 0, n_tests = 0;
    bit    neg_ok    = 1'b0;
    int    cyc       = 0;
    always @(posedge clk) if (busy) cyc <= cyc + 1;   // task resets cyc too

    task automatic run_layer(
        input string tname, input string in_file, input string layer, input string exp_file,
        input int cin, input int cout, input int hin, input int win,
        input bit k3,  input bit s2,   input bit relu);

        int kk   = k3 ? 9 : 1;
        int hout = s2 ? hin / 2 : hin;
        int wout = s2 ? win / 2 : win;
        int err  = 0;
        logic signed [7:0] got, want;
        prm_lane_t pl;

        // --- load golden files ---
        $readmemh({memdir, in_file,  ".mem"},    f_in,  0, cin*hin*win - 1);
        $readmemh({memdir, exp_file, ".mem"},    f_exp, 0, cout*hout*wout - 1);
        $readmemh({memdir, layer,    "_w.mem"},  f_w,   0, cout*cin*kk - 1);
        $readmemh({memdir, layer,    "_b.mem"},  f_b,   0, cout - 1);
        $readmemh({memdir, layer,    "_M.mem"},  f_M,   0, cout - 1);
        $readmemh({memdir, layer,    "_sh.mem"}, f_sh,  0, cout - 1);

        // --- activations: CHW -> PO banks ---
        for (int n = 0; n < PO*DEPTH; n++) act[n] = 8'h00;
        for (int c = 0; c < cin; c++)
            for (int n = 0; n < hin*win; n++)
                act[(c % PO)*DEPTH + (c / PO)*hin*win + n] = f_in[c*hin*win + n];

        // --- weights: [O][I][kh][kw] -> one wide word per (ocg, ic, tap) ---
        for (int n = 0; n < (1 << WAW); n++) wmem[n] = '0;
        for (int o = 0; o < cout; o++)
            for (int c = 0; c < cin; c++)
                for (int t = 0; t < kk; t++)
                    wmem[(o / PO)*cin*kk + c*kk + t][(o % PO)*8 +: 8] = f_w[(o*cin + c)*kk + t];

        // --- params: one word per output group ---
        for (int n = 0; n < (1 << PAW); n++) pmem[n] = '0;
        for (int o = 0; o < cout; o++) begin
            pl.bias = f_b[o];
            pl.m    = f_M[o][15:0];
            pl.sh   = f_sh[o][7:0];
            pmem[o / PO][(o % PO)*PRM_LANE_W +: PRM_LANE_W] = pl;
        end

        // --- run ---
        cfg_cin  = 8'(cin);  cfg_cout = 8'(cout);
        cfg_hin  = 9'(hin);  cfg_win  = 9'(win);
        cfg_hout = 9'(hout); cfg_wout = 9'(wout);
        cfg_k3   = k3; cfg_s2 = s2; cfg_relu = relu;
        cyc = 0;
        @(posedge clk) start <= 1'b1;
        @(posedge clk) start <= 1'b0;
        @(posedge done);
        @(posedge clk);

        // --- compare ---
        for (int o = 0; o < cout; o++)
            for (int n = 0; n < hout*wout; n++) begin
                got  = act[(o % PO)*DEPTH + OUTBASE + (o / PO)*hout*wout + n];
                want = f_exp[o*hout*wout + n];
                if (got !== want) begin
                    if (err < 5)
                        $display("   MISMATCH oc=%0d y=%0d x=%0d got=%0d want=%0d",
                                 o, n / wout, n % wout, got, want);
                    err++;
                end
            end
        n_tests++;
        total_err += err;
        $display("%-26s %0dx%0dx%0d -> %0dx%0dx%0d  k%0d s%0d relu%0d  cycles=%0d  %s",
                 tname, cin, hin, win, cout, hout, wout, k3 ? 3 : 1, s2 ? 2 : 1, relu,
                 cyc, (err == 0) ? "BIT-EXACT" : "FAIL     ");
        if (err != 0) $display("   %0d of %0d values differ", err, cout*hout*wout);
    endtask

    initial begin
        int err_prev;
        if (!$value$plusargs("MEMDIR=%s", memdir)) begin
            // No plusarg: search relative to the simulator's run directory.
            //   [0] older layout
            //   [1] Vivado GUI run dir
            //   [2] batch script run dir      build/xsim_<test>
            string cand [3];
            int    fd;
            cand[0] = "../../../golden_model/data/golden_vectors/mem/";
            cand[1] = "../../../../../../golden_model/data/golden_vectors/mem/";
            cand[2] = "../../golden_model/data/golden_vectors/mem/";
            memdir = "";
            for (int k = 0; k < 3; k++) begin
                fd = $fopen({cand[k], "input_q.mem"}, "r");
                if (fd != 0) begin
                    $fclose(fd);
                    if (memdir == "") memdir = cand[k];
                end
            end
            if (memdir == "") begin
                $display("ERROR: golden .mem folder not found. Pass +MEMDIR=<path>/ (ending in /).");
                $finish;
            end
        end
        $display("target     : Nexys4 DDR, %s, %0d MHz", FPGA_PART, CLK_HZ / 1_000_000);
        $display("golden dir : %s", memdir);
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);

        run_layer("T1 L000 conv",      "input_q",        "L000_Conv", "L000_Conv_out0",  1, 16, 120, 160, 1, 1, 0);
        run_layer("T2 L000 conv+relu", "input_q",        "L000_Conv", "L001_Relu_out0",  1, 16, 120, 160, 1, 1, 1);
        run_layer("T3 L002 conv+relu", "L001_Relu_out0", "L002_Conv", "L003_Relu_out0", 16, 16,  60,  80, 1, 0, 1);
        run_layer("T4 L004 conv s2",   "L003_Relu_out0", "L004_Conv", "L004_Conv_out0", 16, 32,  60,  80, 1, 1, 0);
        run_layer("T5 L026 head 1x1",  "L025_Relu_out0", "L026_Conv", "L026_Conv_out0", 16,  6,  60,  80, 0, 0, 0);

        // negative control: ReLU on, compared with the PRE-ReLU golden -> must differ
        err_prev = total_err;
        $display("(negative control - expected to FAIL)");
        run_layer("N1 relu vs pre-relu", "input_q", "L000_Conv", "L000_Conv_out0", 1, 16, 120, 160, 1, 1, 1);
        n_tests--;
        neg_ok    = (total_err > err_prev);
        total_err = err_prev;

        $display("--------------------------------------------------");
        if (neg_ok) $display("negative control detected mismatches: yes (good)");
        else        $display("negative control detected mismatches: NO - testbench is broken");
        if (total_err == 0 && neg_ok) $display("PASS: %0d/%0d layer tests bit-exact", n_tests, n_tests);
        else                          $display("FAIL: %0d mismatching values", total_err);
        $finish;
    end

    initial begin
        #2s;
        $display("TIMEOUT");
        $finish;
    end
endmodule
