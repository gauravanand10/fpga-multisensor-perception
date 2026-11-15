// ============================================================================
// tb_nexys4_top.sv - the whole FPGA, pin to pin
//
// SPI masters (like the two ESP32s) drive Pmod JA (camera) and JB (ToF); an
// MPU-6500 model answers on JC (IMU1), IMU2 is NOT connected (pull-up on MISO);
// the servo output JD1 is measured; a UART receiver listens on UART_RXD_OUT.
// Sequence (tools/gen_top_vectors.py, same vectors as the base design):
//   ToF 0 + camera 0 (golden frame, SW0=1)  -> result packet 0 (with image)
//   camera 1 with a bad checksum            -> must be rejected, no result packet, LED6 on
//   ToF 1 + camera 2 (mirrored, SW0=0)      -> result packet 1 (no image)
// Checks:
//   * the UART stream is a clean sequence of whole packets (result A5 5A 46 50 or
//     IMU A5 5A 49 4D), nothing in between, so no IMU byte is inside a result packet
//   * both result packets byte-identical to the golden packets (except the two
//     cycle counters, offsets 10..17), checksum checked separately
//   * IMU packets: checksum, version, n >= 1, IMU2 missing (flag 0, WHO_AM_I FF or 00 before
//     the first attempt, zeros),
//     IMU1 WHO_AM_I 0x70 and its 6 values exact once initialised, sample_no continuous,
//     no FIFO overflow, servo fields (manual, target 1556 us for SW[15:8] = 100 deg)
//   * IMU packets keep coming while no camera frame arrives
//   * JD1: 20.000 ms period, pulse widths 1500 -> 1556 us in <= 20 us steps
// The received bytes are written to uart_capture.bin; scripts\run_sim.ps1 then runs
// the legacy result-packet reader and the full stream parser over it (tools\check_capture.py).
//
// UART at CLKS_PER_BIT = 8 and the IMU tick at 1 ms here (fast simulation); the board
// uses 50 (2 Mbaud) and 10 ms (100 Hz). SCLK is the real 500 kHz.
// ============================================================================
`timescale 1ns/1ps

module tb_nexys4_top;
    parameter string RTL_MEM = "../../hw/mem/";
    localparam int CPB      = 8;           // UART clocks per bit in this test
    localparam real SPI_HALF = 50.0;       // ns -> 10 MHz SCLK

    logic        clk = 1'b0;
    always #5 clk = ~clk;
    logic        resetn = 1'b0;
    logic [15:0] sw = 16'h6401;            // SW[15:8] = 100 deg, SW1 = 0 (manual), SW0 = 1 (image)
    logic [15:0] led;
    logic        txd;
    logic [4:1]  ja = 4'b0001, jb = 4'b0001;     // CS_N high, SCLK low
    logic        jc_cs1_n, jc_cs2_n, jc_mosi, jc_sclk, jd_servo;
    tri1         jc_miso;                         // XDC: PULLUP on JC2

    nexys4_top #(.CLKS_PER_BIT(CPB), .MEM_DIR(RTL_MEM),
                 .IMU_TICK_CYC(100_000), .IMU_PWRUP_TICKS(2), .IMU_RESET_TICKS(11), .IMU_RETRY_TICKS(20)) dut (
        .CLK100MHZ(clk), .CPU_RESETN(resetn), .SW(sw), .LED(led), .UART_RXD_OUT(txd),
        .JA(ja), .JB(jb),
        .JC_CS1_N(jc_cs1_n), .JC_MISO(jc_miso), .JC_MOSI(jc_mosi), .JC_SCLK(jc_sclk), .JC_CS2_N(jc_cs2_n),
        .JD_SERVO(jd_servo));

    // ---------------- IMUs: IMU1 present, IMU2 not connected ----------------
    mpu6500_model #(.WHO(8'h70), .RESET_NS(10.0e6), .NAME("IMU1")) m1 (
        .present(1'b1), .sclk(jc_sclk), .mosi(jc_mosi), .cs_n(jc_cs1_n), .miso(jc_miso));
    mpu6500_model #(.WHO(8'h71), .RESET_NS(10.0e6), .NAME("IMU2")) m2 (
        .present(1'b0), .sclk(jc_sclk), .mosi(jc_mosi), .cs_n(jc_cs2_n), .miso(jc_miso));
    logic [15:0] mv [7];
    initial begin
        mv = '{16'd100, -16'sd200, 16'd16384, 16'd0, 16'd5, -16'sd7, 16'd3};   // ax ay az temp gx gy gz
        m1.set_values(mv);
        m2.set_values(mv);
    end

    // ---------------- SPI master (mode 0, MSB first) ----------------
    logic [7:0] pkt [24576];

    task automatic spi_send_cam(input int n);
        ja[1] = 1'b0;
        #(4 * SPI_HALF);
        for (int i = 0; i < n; i++)
            for (int b = 7; b >= 0; b--) begin
                ja[2] = pkt[i][b];
                #(SPI_HALF) ja[4] = 1'b1;
                #(SPI_HALF) ja[4] = 1'b0;
            end
        #(4 * SPI_HALF);
        ja[1] = 1'b1;
        #(20 * SPI_HALF);
    endtask

    task automatic spi_send_tof(input int n);
        jb[1] = 1'b0;
        #(4 * SPI_HALF);
        for (int i = 0; i < n; i++)
            for (int b = 7; b >= 0; b--) begin
                jb[2] = pkt[i][b];
                #(SPI_HALF) jb[4] = 1'b1;
                #(SPI_HALF) jb[4] = 1'b0;
            end
        #(4 * SPI_HALF);
        jb[1] = 1'b1;
        #(20 * SPI_HALF);
    endtask

    // ---------------- UART receiver ----------------
    logic [7:0] rx [262144];
    int         nrx = 0;
    int         fcap;
    initial begin
        fcap = $fopen("uart_capture.bin", "wb");
        forever begin
            logic [7:0] v;
            @(negedge txd);
            #(CPB * 10 * 1.5);
            for (int b = 0; b < 8; b++) begin
                v[b] = txd;
                #(CPB * 10);
            end
            rx[nrx] = v;
            nrx++;
            $fwrite(fcap, "%c", v);
        end
    end

    int total_err = 0, pk_ok = 0;

    // ---------------- stream walker: whole packets only ----------------
    int          n_res = 0, res_pos [8], res_len [8];
    int          n_imu = 0, n_imu_ok1 = 0, n_imu_smp = 0, n_imu_valid = 0, n_imu_while_cam_idle = 0;
    logic [15:0] next_sno;
    bit          sno_init = 0, seen_vals = 0;
    int          last_now = 1500;
    bit          cam_idle = 0;

    task automatic need(input int n);
        while (nrx < n) @(posedge clk);
    endtask

    task automatic check_imu(input int p, input int n, input int len);
        logic [15:0] s, got, sno;
        logic [15:0] tgt, now, v;
        int e = 0;
        bit vals, zero;
        s = '0;
        for (int k = 4; k < len - 2; k++) s += 16'(rx[p + k]);
        got = {rx[p + len - 1], rx[p + len - 2]};
        if (s != got) begin e++; $display("[%t] IMU packet at %0d: checksum %h, packet says %h", $realtime, p, s, got); end
        if (rx[p + 4] != 8'd1) begin e++; $display("IMU packet at %0d: version %0d", p, rx[p + 4]); end
        if (rx[p + 5][1] || (rx[p + 7] != 8'hFF && rx[p + 7] != 8'h00)) begin e++; $display("IMU packet at %0d: IMU2 not shown as missing (flags %h, WHO %h)", p, rx[p + 5], rx[p + 7]); end
        if (rx[p + 5][2]) begin e++; $display("IMU packet at %0d: FIFO overflow", p); end
        if (rx[p + 5][0]) begin
            n_imu_ok1++;
            if (rx[p + 6] != 8'h70) begin e++; $display("IMU packet at %0d: WHO_AM_I 1 = %h", p, rx[p + 6]); end
        end
        tgt = {rx[p + 10], rx[p + 9]};
        now = {rx[p + 12], rx[p + 11]};
        if (rx[p + 8] != 8'd0 || tgt != 16'd1556) begin e++; $display("IMU packet at %0d: servo mode %0d target %0d (expected 0, 1556)", p, rx[p + 8], tgt); end
        if (int'(now) < last_now || int'(now) - last_now > 20 || int'(now) > 1556) begin
            e++; $display("IMU packet at %0d: servo now %0d us after %0d", p, now, last_now);
        end
        last_now = int'(now);
        for (int j = 0; j < n; j++) begin
            int b;
            b = p + 14 + 26 * j;
            sno = {rx[b + 1], rx[b]};
            if (sno_init && sno != next_sno) begin e++; $display("IMU sample_no %0d, expected %0d", sno, next_sno); end
            next_sno = sno + 16'd1; sno_init = 1;
            vals = 1; zero = 1;
            for (int k = 0; k < 6; k++) begin
                v = {rx[b + 3 + 2 * k], rx[b + 2 + 2 * k]};
                if (v != mv[(k < 3) ? k : k + 1]) vals = 0;
                if (v != 16'd0) zero = 0;
            end
            for (int k = 6; k < 12; k++)
                if ({rx[b + 3 + 2 * k], rx[b + 2 + 2 * k]} != 16'd0) begin e++; $display("IMU sample %0d: IMU2 value %0d not 0", sno, k - 6); end
            if (vals) begin seen_vals = 1; n_imu_valid++; end
            else if (!zero || seen_vals) begin e++; $display("IMU sample %0d: IMU1 values wrong", sno); end
        end
        n_imu++; n_imu_smp += n;
        if (cam_idle) n_imu_while_cam_idle++;
        total_err += e;
    endtask

    initial begin : walker
        int pos, len, n;
        pos = 0;
        forever begin
            need(pos + 4);
            if (rx[pos] == 8'hA5 && rx[pos+1] == 8'h5A && rx[pos+2] == 8'h46 && rx[pos+3] == 8'h50) begin
                need(pos + 19);
                len = 19 + 12 * rx[pos + 18] + 192 + 4800 + (rx[pos + 5][0] ? 19200 : 0) + 2;
                need(pos + len);
                if (n_res < 8) begin res_pos[n_res] = pos; res_len[n_res] = len; end
                n_res++;
                $display("[%t] result packet %0d at byte %0d, %0d bytes", $realtime, n_res - 1, pos, len);
                pos += len;
            end else if (rx[pos] == 8'hA5 && rx[pos+1] == 8'h5A && rx[pos+2] == 8'h49 && rx[pos+3] == 8'h4D) begin
                need(pos + 14);
                n = rx[pos + 13];
                if (n == 0 || n > 32) begin
                    total_err++; $display("[%t] IMU packet at %0d with n = %0d", $realtime, pos, n);
                    pos++;
                end else begin
                    len = 14 + 26 * n + 2;
                    need(pos + len);
                    check_imu(pos, n, len);
                    pos += len;
                end
            end else begin
                total_err++;
                $display("[%t] stream: byte %0d (%h) is not the start of a packet", $realtime, pos, rx[pos]);
                pos++;
            end
        end
    end

    // ---------------- servo output ----------------
    realtime t_sr = 0;
    int      n_servo = 0, servo_last = 1500;
    always @(posedge jd_servo) begin
        if (t_sr > 0 && $realtime - t_sr != 20.0e6) begin
            total_err++; $display("[%t] servo period %0.3f ms", $realtime, ($realtime - t_sr) / 1.0e6);
        end
        t_sr = $realtime;
    end
    always @(negedge jd_servo) if (t_sr > 0) begin
        int w;
        w = int'(($realtime - t_sr) / 1000.0);
        if (w < servo_last || w - servo_last > 20 || w > 1556) begin
            total_err++; $display("[%t] servo pulse %0d us after %0d", $realtime, w, servo_last);
        end
        $display("[%t] servo pulse %0d us", $realtime, w);
        servo_last = w; n_servo++;
    end

    // ---------------- compare result packets ----------------
    logic [7:0] exp [24576];
    logic [15:0] lens [2];

    task automatic wait_res(input int target, input int max_ms);
        int cyc = 0;
        while (n_res < target) begin
            @(posedge clk);
            cyc++;
            if (cyc % 200_000 == 0)
                $display("[%t]   ... %0d UART bytes, %0d result / %0d IMU packets, FSM state %0d", $realtime, nrx,
                         n_res, n_imu, dut.u_sys.cs);
            if (cyc >= max_ms * 100_000) begin
                $display("[%t] ERROR: only %0d of %0d result packets after %0d ms", $realtime, n_res, target, max_ms);
                total_err++;
                return;
            end
        end
    endtask

    task automatic check_packet(input int j, input int start_idx);
        int n, err = 0;
        logic [15:0] s, got_sum;
        n = int'(lens[j]);
        if (res_len[j] != n) begin err++; $display("   result packet %0d: %0d bytes, golden %0d", j, res_len[j], n); end
        $readmemh($sformatf("%stb_uart%0d.mem", RTL_MEM, j), exp, 0, n - 1);
        s = '0;
        for (int i = 4; i < n - 2; i++) s = s + 16'(rx[start_idx + i]);
        got_sum = {rx[start_idx + n - 1], rx[start_idx + n - 2]};
        if (s != got_sum) begin err++; $display("   checksum: computed %h, packet says %h", s, got_sum); end
        for (int i = 0; i < n - 2; i++) begin
            if ((i < 10 || i >= 18) && rx[start_idx + i] !== exp[i]) begin   // skip the cycle counters
                if (err < 6) $display("   byte %0d: got %h expected %h", i, rx[start_idx + i], exp[i]);
                err++;
            end
        end
        total_err += err;
        if (err == 0) pk_ok++;
        $display("UART result packet %0d at byte %0d: %0d bytes, U-Net %0d cycles, post-proc %0d cycles, %0d objects  %s",
                 j, start_idx, n, {rx[start_idx+13], rx[start_idx+12], rx[start_idx+11], rx[start_idx+10]},
                 {rx[start_idx+17], rx[start_idx+16], rx[start_idx+15], rx[start_idx+14]},
                 rx[start_idx+18], (err == 0) ? "BIT-EXACT" : "FAIL");
    endtask

    initial begin
        int imu_before;
        $timeformat(-3, 2, " ms", 10);
        for (int i = 0; i < 262144; i++) rx[i] = 8'h00;
        $readmemh({RTL_MEM, "tb_uart_len.mem"}, lens);
        $display("Nexys4 DDR top: SPI camera + SPI ToF -> U-Net -> objects -> UART, + IMU (JC) + servo (JD1)");
        #(1000) resetn = 1'b1;
        while (dut.u_sys.rst !== 1'b0) @(posedge clk);
        $display("[%t] reset released", $realtime);

        $readmemh({RTL_MEM, "tb_tof0.mem"}, pkt, 0, 199);  spi_send_tof(200);
        $readmemh({RTL_MEM, "tb_cam0.mem"}, pkt, 0, 19207); spi_send_cam(19208);
        $display("[%t] camera frame 0 sent", $realtime);
        wait_res(1, 60);
        check_packet(0, res_pos[0]);

        sw = 16'h6400;                                        // no image from now on
        $readmemh({RTL_MEM, "tb_cam1.mem"}, pkt, 0, 19207); spi_send_cam(19208);
        imu_before = n_imu; cam_idle = 1;
        #(200_000);
        if (n_res != 1 || led[6] !== 1'b1) begin
            total_err++; $display("bad camera packet: NOT rejected (result packets %0d, LED6 %b)", n_res, led[6]);
        end else
            $display("[%t] camera frame 1 (bad checksum) rejected, LED6 on; %0d IMU packets meanwhile", $realtime, n_imu - imu_before);
        #(5_000_000);                                         // 5 ms without camera frames
        cam_idle = 0;
        if (n_imu - imu_before < 4) begin total_err++; $display("only %0d IMU packets in 5.2 ms without camera frames", n_imu - imu_before); end

        $readmemh({RTL_MEM, "tb_tof1.mem"}, pkt, 0, 199);  spi_send_tof(200);
        $readmemh({RTL_MEM, "tb_cam2.mem"}, pkt, 0, 19207); spi_send_cam(19208);
        $display("[%t] camera frame 2 sent", $realtime);
        wait_res(2, 60);
        check_packet(1, res_pos[1]);
        while (n_servo < 4) @(posedge clk);                   // servo reaches 1556 us
        #(2_000_000);

        $fclose(fcap);
        if (m1.errs) begin total_err += m1.errs; $display("IMU1 model: %0d SPI errors", m1.errs); end
        if (n_imu_ok1 == 0 || n_imu_valid < 20) begin total_err++; $display("IMU1 never delivered values (%0d)", n_imu_valid); end
        if (servo_last != 1556) begin total_err++; $display("servo pulse %0d us at the end (expected 1556)", servo_last); end
        if (n_res != 2) begin total_err++; $display("%0d result packets (expected 2)", n_res); end
        $display("LEDs: %b  (8..13 = objects in the last frame)", led);
        $display("UART: %0d bytes = %0d result packets + %0d IMU packets (%0d samples, %0d with IMU1 values, IMU1 ok in %0d packets, %0d while no camera frame)",
                 nrx, n_res, n_imu, n_imu_smp, n_imu_valid, n_imu_ok1, n_imu_while_cam_idle);
        $display("servo: %0d pulses, last %0d us", n_servo, servo_last);
        $display("--------------------------------------------------");
        if (total_err == 0 && pk_ok == 2)
            $display("PASS: 2/2 result packets bit-exact, bad packet rejected, %0d IMU packets ok, stream clean, servo ok", n_imu);
        else
            $display("FAIL: %0d errors", total_err);
        $finish;
    end

    initial begin #2s; $display("TIMEOUT"); $finish; end
endmodule
