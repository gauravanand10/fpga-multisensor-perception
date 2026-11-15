// ============================================================================
// tb_imu_packer.sv - imu_fifo + imu_packer + uart_arbiter + uart_tx
//
// Part 1 (tools/gen_imu_vectors.py): 4 IMU packets, UART bytes compared with the
//   Python model (fpga_model.imu_packet), incl. checksum, n = 32 with 2 samples
//   dropped -> overflow flag, flag cleared in the next packet.
// Part 2 (arbitration): a stand-in for uart_packer sends 304-byte "result packets"
//   (A5 5A 46 50 + known bytes) while IMU samples arrive every 6 us. The stream must be
//   a clean sequence of whole packets: no IMU byte inside a result packet, a result
//   packet waits for a running IMU packet, every IMU packet has n >= 1 and a good
//   checksum, no sample lost (sample_no continuous), result packets byte-exact.
// Part 3: FIFO empty for a while -> no bytes at all (no empty packets).
// ============================================================================
`timescale 1ns/1ps

module fake_res_src #(parameter int NB = 304) (
    input  logic       clk, rst, start,
    output logic       busy, done,
    output logic       tx_valid,
    output logic [7:0] tx_data,
    input  logic       tx_ready
);
    int  k;
    logic [1:0] ph;
    always_ff @(posedge clk) begin
        done <= 1'b0; tx_valid <= 1'b0;
        if (rst) begin busy <= 1'b0; ph <= 0; k <= 0; end
        else if (!busy) begin
            if (start) begin busy <= 1'b1; k <= 0; ph <= 0; end
        end else unique case (ph)
            2'd0: if (tx_ready && !tx_valid) begin
                tx_valid <= 1'b1;
                tx_data  <= (k == 0) ? 8'hA5 : (k == 1) ? 8'h5A : (k == 2) ? 8'h46 : (k == 3) ? 8'h50 : 8'(k * 3 + 1);
                ph <= 2'd1;
            end
            default: begin
                ph <= 2'd0;
                if (k == NB - 1) begin busy <= 1'b0; done <= 1'b1; end
                k <= k + 1;
            end
        endcase
    end
endmodule


module tb_imu_packer;
    parameter string RTL_MEM = "../../hw/mem/";
    localparam int CPB = 4;
    localparam int NRES = 304;

    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst = 1'b1;

    // FIFO
    logic         wr = 1'b0;
    logic [207:0] wdata = '0;
    logic         pop, ovf, ovf_clr;
    logic [207:0] q;
    logic [5:0]   count;
    imu_fifo u_fifo (.clk, .rst, .wr, .wdata, .pop, .q, .count, .ovf, .ovf_clr);

    // status
    logic [1:0]  imu_ok = '0;
    logic [7:0]  who1 = '0, who2 = '0, smode = '0;
    logic [15:0] tgt = '0, now = '0;

    logic im_start, im_busy, im_done, im_valid, im_ready;
    logic [7:0] im_data;
    imu_packer u_pk (.clk, .rst, .start(im_start), .busy(im_busy), .done(im_done),
        .fifo_count(count), .fifo_q(q), .fifo_pop(pop), .fifo_ovf(ovf), .ovf_clr,
        .imu_ok, .who1, .who2, .servo_mode(smode), .servo_target_us(tgt), .servo_now_us(now),
        .tx_valid(im_valid), .tx_data(im_data), .tx_ready(im_ready));

    logic res_req = 1'b0, res_start, res_busy, res_done, res_valid;
    logic [7:0] res_data;
    fake_res_src #(.NB(NRES)) u_res (.clk, .rst, .start(res_start), .busy(res_busy), .done(res_done),
        .tx_valid(res_valid), .tx_data(res_data), .tx_ready(im_ready));

    logic hold = 1'b1;
    logic tx_valid, sel_imu, tx_ready, txd;
    logic [7:0] tx_data;
    uart_arbiter u_arb (.clk, .rst, .res_req, .res_start, .res_done, .res_valid, .res_data,
        .imu_req(count != 6'd0 && !hold), .imu_start(im_start), .imu_done(im_done),
        .imu_valid(im_valid), .imu_data(im_data), .tx_valid, .tx_data, .sel_imu);
    uart_tx #(.CLKS_PER_BIT(CPB)) u_tx (.clk, .rst, .valid(tx_valid), .data(tx_data), .ready(tx_ready), .txd);
    assign im_ready = tx_ready;

    int errs = 0;

    // ---------------- arbitration monitors ----------------
    always @(posedge clk) if (!rst) begin
        if (res_start && im_busy) begin errs++; $display("[%t] result packet started inside an IMU packet", $realtime); end
        if (im_start && res_busy) begin errs++; $display("[%t] IMU packet started inside a result packet", $realtime); end
        if (im_valid && res_valid) begin errs++; $display("[%t] both packers sent a byte", $realtime); end
        if (im_start && count == 6'd0) begin errs++; $display("[%t] IMU packet started with an empty FIFO", $realtime); end
    end

    // ---------------- UART receiver ----------------
    logic [7:0] rx [8192];
    int nrx = 0;
    initial forever begin
        logic [7:0] v;
        @(negedge txd);
        #(CPB * 10 * 1.5);
        for (int b = 0; b < 8; b++) begin v[b] = txd; #(CPB * 10); end
        rx[nrx] = v; nrx++;
    end

    task automatic wait_rx(input int n, input int max_cyc);
        int c = 0;
        while (nrx < n && c < max_cyc) begin @(posedge clk); c++; end
        if (nrx < n) begin errs++; $display("[%t] only %0d of %0d bytes", $realtime, nrx, n); end
    endtask

    task automatic push(input logic [207:0] w);
        @(posedge clk); wr <= 1'b1; wdata <= w;
        @(posedge clk); wr <= 1'b0;
        repeat (3) @(posedge clk);
    endtask

    // ---------------- vectors ----------------
    logic [207:0] smp [64];
    logic [63:0]  st [4];
    logic [7:0]   npush [4];
    logic [15:0]  lens [4];
    logic [7:0]   exp [2048];

    // ---------------- part 2 stream walker ----------------
    int n_res = 0, n_imu = 0, n_imu_smp = 0;
    logic [15:0] next_sno;
    task automatic walk(input int from, input int to);
        int p = from;
        while (p < to) begin
            if (p + 4 > to) begin errs++; $display("stream ends inside a sync at %0d", p); return; end
            if (rx[p] == 8'hA5 && rx[p+1] == 8'h5A && rx[p+2] == 8'h46 && rx[p+3] == 8'h50) begin
                for (int k = 4; k < NRES; k++)
                    if (rx[p + k] !== 8'(k * 3 + 1)) begin
                        errs++; $display("result packet at %0d: byte %0d = %h (IMU byte inside?)", p, k, rx[p + k]); break;
                    end
                n_res++; p += NRES;
            end else if (rx[p] == 8'hA5 && rx[p+1] == 8'h5A && rx[p+2] == 8'h49 && rx[p+3] == 8'h4D) begin
                int n, len;
                logic [15:0] s, got, sno;
                n = rx[p + 13]; len = 14 + 26 * n + 2;
                if (n == 0 || n > 32) begin errs++; $display("IMU packet at %0d with n = %0d", p, n); return; end
                s = '0;
                for (int k = 4; k < len - 2; k++) s += 16'(rx[p + k]);
                got = {rx[p + len - 1], rx[p + len - 2]};
                if (s != got) begin errs++; $display("IMU packet at %0d: checksum %h, packet says %h", p, s, got); end
                if (rx[p + 5][2]) begin errs++; $display("IMU packet at %0d: overflow flag set", p); end
                for (int j = 0; j < n; j++) begin
                    sno = {rx[p + 15 + 26 * j], rx[p + 14 + 26 * j]};
                    if (sno != next_sno) begin errs++; $display("IMU sample_no %0d, expected %0d", sno, next_sno); end
                    next_sno = sno + 16'd1;
                    if ({rx[p + 17 + 26 * j], rx[p + 16 + 26 * j]} != 16'(sno * 7 + 1)) begin
                        errs++; $display("IMU sample %0d: ax1 wrong", sno);
                    end
                end
                n_imu++; n_imu_smp += n; p += len;
            end else begin
                errs++; $display("stream: no packet sync at byte %0d (%h %h %h %h)", p, rx[p], rx[p+1], rx[p+2], rx[p+3]);
                return;
            end
        end
    endtask

    initial begin
        int base, si, pushed;
        $timeformat(-6, 1, " us", 10);
        $readmemh({RTL_MEM, "tb_imu_samples.mem"}, smp);
        $readmemh({RTL_MEM, "tb_imu_status.mem"}, st);
        $readmemh({RTL_MEM, "tb_imu_npush.mem"}, npush);
        $readmemh({RTL_MEM, "tb_imu_len.mem"}, lens);
        $readmemh({RTL_MEM, "tb_imu_exp.mem"}, exp);
        $display("imu_packer: FIFO 32 x 208 bit (LUT-RAM), packets vs fpga_model.imu_packet, UART arbitration");
        repeat (10) @(posedge clk);
        rst = 1'b0;

        // ---------------- part 1 ----------------
        base = 0; si = 0;
        for (int c = 0; c < 4; c++) begin
            int e = 0;
            hold = 1'b1;
            {imu_ok, who1, who2, smode, tgt, now} = {st[c][57:56], st[c][55:0]};
            for (int i = 0; i < npush[c]; i++) begin push(smp[si]); si++; end
            if (c == 2 && ovf !== 1'b1) begin errs++; $display("overflow flag not set after 34 pushes"); end
            hold = 1'b0;
            wait_rx(base + lens[c], 200_000);
            repeat (100) @(posedge clk);
            for (int i = 0; i < lens[c]; i++)
                if (rx[base + i] !== exp[base + i]) begin
                    if (e < 5) $display("   packet %0d byte %0d: got %h expected %h", c, i, rx[base + i], exp[base + i]);
                    e++;
                end
            if (nrx != base + lens[c]) begin e++; $display("   packet %0d: %0d extra bytes", c, nrx - base - lens[c]); end
            errs += e;
            $display("[%t] IMU packet %0d: %0d pushed, n = %0d, flags %h, %0d bytes  %s", $realtime, c, npush[c],
                     rx[base + 13], rx[base + 5], lens[c], e ? "FAIL" : "BYTE-EXACT (incl. sum16)");
            base += lens[c];
        end
        if (ovf !== 1'b0) begin errs++; $display("overflow flag not cleared"); end

        // ---------------- part 2 ----------------
        next_sno = 16'd0; pushed = 0;
        fork
            begin                                    // IMU samples every 30 us (UART: ~11 us per sample)
                for (int k = 0; k < 120; k++) begin
                    logic [207:0] w;
                    w = '0;
                    w[15:0] = 16'(k);
                    for (int j = 1; j < 13; j++) w[16 * j +: 16] = 16'(k * 7 + j);
                    push(w); pushed++;
                    repeat (2995) @(posedge clk);
                end
            end
            begin                                    // result packets: one while idle, two during IMU packets
                repeat (2000) @(posedge clk);
                for (int r = 0; r < 3; r++) begin
                    if (r > 0) while (!sel_imu) @(posedge clk);     // request while an IMU packet runs
                    res_req = 1'b1;
                    while (!res_start) @(posedge clk);
                    @(posedge clk); res_req = 1'b0;
                    while (!res_done) @(posedge clk);
                    repeat (3000) @(posedge clk);
                end
            end
        join
        while (count != 0 || im_busy || res_busy || !tx_ready) @(posedge clk);
        repeat (CPB * 20) @(posedge clk);
        walk(base, nrx);
        $display("[%t] arbitration: %0d result packets, %0d IMU packets with %0d of %0d samples, stream clean",
                 $realtime, n_res, n_imu, n_imu_smp, pushed);
        if (n_res != 3 || n_imu_smp != pushed) begin errs++; $display("expected 3 result packets and all samples"); end

        // ---------------- part 3 ----------------
        base = nrx;
        repeat (50_000) @(posedge clk);
        if (nrx != base) begin errs++; $display("%0d bytes sent with an empty FIFO", nrx - base); end
        else $display("[%t] FIFO empty for 500 us: no bytes (no empty packets)", $realtime);

        $display("--------------------------------------------------");
        if (errs == 0) $display("PASS: 4/4 IMU packets byte-exact vs Python, overflow flag, arbitration clean, no empty packets");
        else           $display("FAIL: %0d errors", errs);
        $finish;
    end

    initial begin #50ms; $display("TIMEOUT"); $finish; end
endmodule
