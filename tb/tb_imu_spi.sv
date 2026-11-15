// ============================================================================
// tb_imu_spi.sv - imu_spi (real parameters) against two MPU-6500/9250 SPI slave models
//
// One shared bus, two chip selects, pull-up on MISO (missing IMU reads 0xFF).
//   IMU1 = MPU-6500 (WHO_AM_I 0x70), present from the start
//   IMU2 = MPU-9250 (WHO_AM_I 0x71), ABSENT at the start, plugged in at 400 ms
// Checks:
//   * init writes per IMU, exact order and values, H_RESET first, >= 100 ms before the rest
//   * SCLK period 2.000 us (500 kHz, <= 1 MHz), mode 0 / CS framing / whole bytes (model)
//   * never both chip selects low
//   * every sample: 12 values = the model registers of that moment for an IMU that is OK,
//     0 for one that is not; sample_no +1 each time; one sample every 10 ms (+-1 ms)
//     also while an IMU is missing (no lock-up)
//   * IMU2 absent -> flag 0, WHO_AM_I 0xFF; after plugging in -> recovered (retry ~1 s)
//   * IMU1 unplugged -> flag 0 / zeros on the next sample; plugged back -> recovered
// ============================================================================
`timescale 1ns/1ps

module tb_imu_spi;
    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst = 1'b1;

    logic sclk, mosi, cs1_n, cs2_n;
    tri1  miso;                                   // pull-up (XDC PULLUP on JC2)
    logic sample_valid;
    logic [207:0] sample;
    logic [1:0] imu_ok;
    logic [7:0] who1, who2;
    logic p1 = 1'b1, p2 = 1'b0;

    imu_spi dut (.clk, .rst, .sclk, .mosi, .cs1_n, .cs2_n, .miso_i(miso),
                 .sample_valid, .sample, .imu_ok, .who1, .who2);
    mpu6500_model #(.WHO(8'h70), .NAME("IMU1")) m1 (.present(p1), .sclk, .mosi, .cs_n(cs1_n), .miso);
    mpu6500_model #(.WHO(8'h71), .NAME("IMU2")) m2 (.present(p2), .sclk, .mosi, .cs_n(cs2_n), .miso);

    int errs = 0;

    // ---------------- both chip selects low? ----------------
    always @(posedge clk) if (!rst && !cs1_n && !cs2_n) begin
        errs++; $display("[%t] both chip selects low", $realtime);
    end

    // ---------------- SCLK period ----------------
    realtime t_sr = 0;
    int n_per = 0;
    always @(posedge sclk) begin
        if (t_sr > 0 && $realtime - t_sr < 3000.0) begin       // inside one byte/transaction
            if ($realtime - t_sr != 2000.0) begin
                errs++; $display("[%t] SCLK period %0.1f ns (expected 2000)", $realtime, $realtime - t_sr);
            end
            n_per++;
        end
        t_sr = $realtime;
    end

    // ---------------- model values ----------------
    logic [15:0] v1 [7], v2 [7];
    int          nset = 0;
    task automatic new_values();
        for (int k = 0; k < 7; k++) begin
            v1[k] = 16'(nset * 37 + k * 1111 - 7000);
            v2[k] = 16'(-(nset * 53) + k * 2222 + 900);
        end
        v1[2] = 16'(16384 + nset);                 // az ~ 1 g
        v2[2] = 16'(-16384 - nset);
        m1.set_values(v1);
        m2.set_values(v2);
        nset++;
    endtask

    function automatic logic [95:0] pack(input logic [15:0] v [7], input logic ok);
        // ax ay az gx gy gz (temperature v[3] is not sent)
        if (!ok) return 96'd0;
        return {v[6], v[5], v[4], v[2], v[1], v[0]};
    endfunction

    // ---------------- every sample ----------------
    int          n_samp = 0, n_ok1 = 0, n_ok2 = 0;
    logic [15:0] last_sno;
    realtime     t_last = 0;
    logic [95:0] e1, e2;
    always @(posedge clk) if (sample_valid) begin
        e1 = pack(v1, imu_ok[0]);
        e2 = pack(v2, imu_ok[1]);
        if (sample[111:16] !== e1) begin
            errs++; if (errs < 10) $display("[%t] sample %0d IMU1 values %h expected %h", $realtime, n_samp, sample[111:16], e1);
        end
        if (sample[207:112] !== e2) begin
            errs++; if (errs < 10) $display("[%t] sample %0d IMU2 values %h expected %h", $realtime, n_samp, sample[207:112], e2);
        end
        if (imu_ok[0] && !p1) begin errs++; $display("[%t] IMU1 OK while unplugged", $realtime); end
        if (imu_ok[1] && !p2) begin errs++; $display("[%t] IMU2 OK while unplugged", $realtime); end
        if (n_samp > 0) begin
            if (sample[15:0] != last_sno + 16'd1) begin errs++; $display("[%t] sample_no %0d after %0d", $realtime, sample[15:0], last_sno); end
            if ($realtime - t_last < 9.0e6 || $realtime - t_last > 11.0e6) begin
                errs++; $display("[%t] %0.3f ms between samples", $realtime, ($realtime - t_last) / 1.0e6);
            end
        end
        last_sno = sample[15:0]; t_last = $realtime;
        n_samp++;
        if (imu_ok[0]) n_ok1++;
        if (imu_ok[1]) n_ok2++;
        new_values();                              // the next sample must show these
    end

    // ---------------- init write log ----------------
    logic [7:0] ea [7] = '{8'h6B, 8'h6B, 8'h6A, 8'h1A, 8'h1B, 8'h1C, 8'h1D};
    logic [7:0] ed [7] = '{8'h80, 8'h01, 8'h10, 8'h03, 8'h00, 8'h00, 8'h03};
    task automatic check_init(input string name, input int first, input int nwr,
                              input logic [7:0] wa [256], input logic [7:0] wd [256], input realtime wt [256]);
        int e;
        e = 0;
        if (nwr < first + 7) begin errs++; $display("%s: only %0d register writes", name, nwr); return; end
        for (int i = 0; i < 7; i++)
            if (wa[first + i] !== ea[i] || wd[first + i] !== ed[i]) begin
                e++; $display("%s: write %0d = %h:%h, expected %h:%h", name, i, wa[first + i], wd[first + i], ea[i], ed[i]);
            end
        if (wt[first + 1] - wt[first] < 100.0e6) begin
            e++; $display("%s: only %0.3f ms between H_RESET and the next write", name, (wt[first + 1] - wt[first]) / 1.0e6);
        end
        errs += e;
        $display("[%t] %s init writes %0d..%0d: 6B=80 | %0.1f ms | 6B=01 6A=10 1A=03 1B=00 1C=00 1D=03  %s", $realtime, name,
                 first, first + 6, (wt[first + 1] - wt[first]) / 1.0e6, e ? "WRONG" : "ok");
    endtask

    task automatic wait_ms(input real ms);
        #(ms * 1.0e6);
    endtask

    task automatic wait_ok(input int i, input logic want, input real max_ms, input string what);
        realtime t0;
        t0 = $realtime;
        while (imu_ok[i] !== want && $realtime - t0 < max_ms * 1.0e6) @(posedge clk);
        if (imu_ok[i] !== want) begin errs++; $display("FAIL %s: not reached after %0.0f ms", what, max_ms); end
        else $display("[%t] %s (after %0.1f ms)", $realtime, what, ($realtime - t0) / 1.0e6);
    endtask

    initial begin
        $timeformat(-3, 2, " ms", 10);
        $display("imu_spi: SCLK 500 kHz, 100 Hz samples, IMU1 present, IMU2 absent until 400 ms");
        new_values();
        repeat (20) @(posedge clk);
        rst = 1'b0;

        wait_ok(0, 1'b1, 400.0, "IMU1 OK (WHO_AM_I 0x70)");
        if (m1.wr_time[0] < 100.0e6) begin errs++; $display("H_RESET %0.3f ms after reset (< 100)", m1.wr_time[0] / 1.0e6); end
        $display("[%t] first H_RESET at %0.1f ms after start", $realtime, m1.wr_time[0] / 1.0e6);
        check_init("IMU1", 0, m1.nwr, m1.wr_addr, m1.wr_data, m1.wr_time);
        wait_ms(400.0 - $realtime / 1.0e6);
        if (imu_ok[1] !== 1'b0 || who2 !== 8'hFF) begin errs++; $display("IMU2 absent: ok %b, WHO_AM_I %h (expected 0, FF)", imu_ok[1], who2); end
        else $display("[%t] IMU2 absent: flag 0, WHO_AM_I %h, values 0, %0d samples so far", $realtime, who2, n_samp);

        p2 = 1'b1;                                         // plug in IMU2
        $display("[%t] IMU2 plugged in", $realtime);
        wait_ok(1, 1'b1, 1600.0, "IMU2 recovered (WHO_AM_I 0x71)");
        if (who2 !== 8'h71) begin errs++; $display("who2 %h", who2); end
        check_init("IMU2", 0, m2.nwr, m2.wr_addr, m2.wr_data, m2.wr_time);
        wait_ms(50.0);

        while (!sample_valid) @(posedge clk);              // unplug between two rounds
        p1 = 1'b0;                                         // unplug IMU1
        $display("[%t] IMU1 unplugged", $realtime);
        wait_ok(0, 1'b0, 25.0, "IMU1 flagged missing");
        if (imu_ok[1] !== 1'b1) begin errs++; $display("IMU2 lost while IMU1 was unplugged"); end
        wait_ms(300.0);
        p1 = 1'b1;
        $display("[%t] IMU1 plugged back", $realtime);
        wait_ok(0, 1'b1, 1600.0, "IMU1 recovered");
        check_init("IMU1 (2nd)", 7, m1.nwr, m1.wr_addr, m1.wr_data, m1.wr_time);
        wait_ms(50.0);

        if (m1.errs || m2.errs) begin errs += m1.errs + m2.errs; $display("model errors: IMU1 %0d, IMU2 %0d", m1.errs, m2.errs); end
        if (n_per < 100) begin errs++; $display("too few SCLK periods measured (%0d)", n_per); end
        $display("--------------------------------------------------");
        $display("%0d samples (IMU1 OK in %0d, IMU2 OK in %0d), %0d SCLK periods of 2.000 us, IMU1 %0d / IMU2 %0d CS frames, %0d / %0d bursts",
                 n_samp, n_ok1, n_ok2, n_per, m1.n_frames, m2.n_frames, m1.n_bursts, m2.n_bursts);
        if (errs == 0)
            $display("PASS: init order, SCLK 500 kHz, CS framing, 12 values exact, IMU absent flagged without lock-up, recovery");
        else
            $display("FAIL: %0d errors", errs);
        $finish;
    end

    initial begin #5s; $display("TIMEOUT"); $finish; end
endmodule
