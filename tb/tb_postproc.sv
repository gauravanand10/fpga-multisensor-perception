// ============================================================================
// tb_postproc.sv - connected components + ToF zones + per-object distance
//
// Test cases from tools/gen_pp_vectors.py (expected values from tools/fpga_model.py):
//   case 0 = label map of the golden frame, 1..4 synthetic shapes, 5 = >63 objects.
// Checks object count, flags, every object's class/area/box/distance/zones and
// the whole 80x60 object map.  Run from build/<dir> (hw/scripts/run_sim.ps1 does this).
// ============================================================================
`timescale 1ns/1ps

module tb_postproc;
    import unet_pkg::*;

    parameter string RTL_MEM = "../../hw/mem/";

    logic clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;

    logic           lab_we = 1'b0;
    logic [LAW-1:0] lab_waddr = '0;
    logic [2:0]     lab_wdata = '0;
    logic           start = 1'b0, busy, done;
    logic [31:0]    cycles;
    logic [16*64-1:0] tof_dist;
    logic [8*64-1:0]  tof_status;
    logic [5:0]     n_obj, n_comp, obj_k = 6'd1;
    logic           flag_overflow, flag_noconv, obj_valid;
    logic [2:0]     obj_cls;
    logic [12:0]    obj_area;
    logic [6:0]     obj_x0, obj_x1, obj_nsel;
    logic [5:0]     obj_y0, obj_y1;
    logic [15:0]    obj_dist;
    logic           km_re = 1'b0;
    logic [LAW-1:0] km_raddr = '0;
    logic [5:0]     km_rdata;

    postproc #(
        .RECTS_FILE({RTL_MEM, "zone_rects.mem"}),
        .RZ_FILE   ({RTL_MEM, "zone_rz.mem"}),
        .BINLO_FILE({RTL_MEM, "zone_binlo.mem"})
    ) dut (.*);

    logic [3:0]  f_lab [NPIX];
    logic [15:0] f_tof [128];
    logic [15:0] f_exp [3 + 63*9 + NPIX];
    logic [3:0]  f_n [1];

    int total_err = 0, cases_ok = 0;

    task automatic run_case(input int ci);
        int err = 0, base;
        string tag;
        tag = $sformatf("%stb_pp%0d", RTL_MEM, ci);
        $readmemh({tag, "_lab.mem"}, f_lab);
        $readmemh({tag, "_tof.mem"}, f_tof);
        $readmemh({tag, "_exp.mem"}, f_exp);
        for (int i = 0; i < 64; i++) begin
            tof_dist[i*16 +: 16]  = f_tof[i];
            tof_status[i*8 +: 8]  = f_tof[64 + i][7:0];
        end
        // write the label map as the U-Net head layer does
        for (int p = 0; p < NPIX; p++) begin
            @(negedge clk);
            lab_we = 1'b1; lab_waddr = LAW'(p); lab_wdata = f_lab[p][2:0];
        end
        @(negedge clk) lab_we = 1'b0;
        @(negedge clk) start = 1'b1;
        @(negedge clk) start = 1'b0;
        @(posedge done);
        @(negedge clk);
        if (n_obj !== f_exp[0][5:0]) begin err++; $display("   n_obj %0d expected %0d", n_obj, f_exp[0]); end
        if (n_comp !== f_exp[1][5:0]) begin err++; $display("   n_comp %0d expected %0d", n_comp, f_exp[1]); end
        if (flag_overflow !== f_exp[2][0]) begin err++; $display("   overflow %0d expected %0d", flag_overflow, f_exp[2]); end
        if (flag_noconv) begin err++; $display("   CC did not converge"); end
        for (int k = 1; k < 64; k++) begin
            base = 3 + (k - 1) * 9;
            obj_k = 6'(k);
            #1;
            if (obj_valid !== f_exp[base][0]) begin
                err++; $display("   obj %0d valid %0d expected %0d", k, obj_valid, f_exp[base]);
            end else if (obj_valid) begin
                if (obj_cls !== f_exp[base+1][2:0] || obj_area !== f_exp[base+2][12:0] ||
                    obj_x0 !== f_exp[base+3][6:0] || obj_y0 !== f_exp[base+4][5:0] ||
                    obj_x1 !== f_exp[base+5][6:0] || obj_y1 !== f_exp[base+6][5:0] ||
                    obj_dist !== f_exp[base+7] || obj_nsel !== f_exp[base+8][6:0]) begin
                    err++;
                    $display("   obj %0d got cls %0d area %0d box %0d,%0d-%0d,%0d dist %0d nsel %0d", k,
                             obj_cls, obj_area, obj_x0, obj_y0, obj_x1, obj_y1, obj_dist, obj_nsel);
                    $display("          want cls %0d area %0d box %0d,%0d-%0d,%0d dist %0d nsel %0d",
                             f_exp[base+1], f_exp[base+2], f_exp[base+3], f_exp[base+4], f_exp[base+5],
                             f_exp[base+6], f_exp[base+7], f_exp[base+8]);
                end
            end
        end
        begin : kmap
            int kerr = 0;
            for (int p = 0; p < NPIX; p++) begin
                @(negedge clk) km_re = 1'b1; km_raddr = LAW'(p);
                @(negedge clk) km_re = 1'b0;
                @(negedge clk);
                if (km_rdata !== f_exp[3 + 63*9 + p][5:0]) kerr++;
            end
            if (kerr) begin err++; $display("   object map: %0d pixels differ", kerr); end
        end
        total_err += err;
        if (err == 0) cases_ok++;
        $display("case %0d: %2d objects, %2d numbered, overflow=%0d  cycles=%6d  %s",
                 ci, n_obj, n_comp, flag_overflow, cycles, (err == 0) ? "BIT-EXACT" : "FAIL");
    endtask

    initial begin
        int nc;
        $readmemh({RTL_MEM, "tb_pp_ncases.mem"}, f_n);
        nc = int'(f_n[0]);
        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (3) @(posedge clk);
        for (int c = 0; c < nc; c++) run_case(c);
        $display("--------------------------------------------------");
        if (total_err == 0) $display("PASS: %0d/%0d post-processing cases bit-exact", cases_ok, nc);
        else                $display("FAIL: %0d/%0d cases", nc - cases_ok, nc);
        $finish;
    end

    initial begin #1s; $display("TIMEOUT"); $finish; end
endmodule
