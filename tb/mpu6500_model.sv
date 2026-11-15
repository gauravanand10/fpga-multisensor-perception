// ============================================================================
// mpu6500_model.sv - behavioural SPI slave of an MPU-6500 / MPU-9250 (simulation only)
//
// SPI mode 0, MSB first, first byte = R/W (bit 7) + address, then data bytes with
// address auto-increment. MISO driven only while NCS is low (otherwise 'z, so a
// pull-up on the shared line gives 0xFF for a missing IMU). present = 0: the chip is
// not there (no answer, no register changes).
//
// Checks (counted in errs, printed):
//   * SCLK low when NCS falls / rises (mode 0), whole bytes per NCS-low frame
//   * SCLK high and low phases >= 500 ns (SCLK <= 1 MHz, the MPU limit for all registers)
//   * MOSI stable >= 100 ns before each rising SCLK edge
//   * no access during RESET_NS after a H_RESET write (PWR_MGMT_1 bit 7)
// Log: every register write (address, data, time) in wr_addr / wr_data / wr_time.
// ============================================================================
`timescale 1ns/1ps

module mpu6500_model #(
    parameter logic [7:0] WHO      = 8'h70,
    parameter real        RESET_NS = 100_000_000.0,     // 100 ms
    parameter string      NAME     = "IMU"
)(
    input  logic present,
    input  logic sclk,
    input  logic mosi,
    input  logic cs_n,
    output wire  miso
);
    logic [7:0] regs [128];
    int         errs = 0;
    int         n_frames = 0, n_bursts = 0;
    int         nwr = 0;
    logic [7:0] wr_addr [256];
    logic [7:0] wr_data [256];
    realtime    wr_time [256];
    realtime    t_reset = -1.0e12;

    logic       miso_r = 1'b0;
    assign miso = (present && !cs_n) ? miso_r : 1'bz;

    task automatic hw_reset();
        for (int i = 0; i < 128; i++)              // sensor data (0x3B..0x48) keeps being measured
            if (i < 8'h3B || i > 8'h48) regs[i] = 8'h00;
        regs[7'h6B] = 8'h01;
        regs[7'h75] = WHO;
    endtask

    // accel x,y,z, temperature, gyro x,y,z (signed 16 bit) -> 0x3B..0x48
    task automatic set_values(input logic [15:0] v [7]);
        for (int i = 0; i < 7; i++) begin
            regs[7'h3B + 2 * i]     = v[i][15:8];
            regs[7'h3B + 2 * i + 1] = v[i][7:0];
        end
    endtask

    initial hw_reset();

    int         nbit;
    logic [7:0] sh;
    logic [7:0] addr;
    logic       rd;
    realtime    t_rise = 0, t_fall = 0, t_mosi = 0;
    bit         in_frame = 0;

    always @(mosi) t_mosi = $realtime;

    always @(negedge cs_n) if (present) begin
        nbit = 0; in_frame = 1;
        if (sclk !== 1'b0) begin errs++; $display("[%t] %s: NCS fell with SCLK high", $realtime, NAME); end
        if ($realtime - t_reset < RESET_NS) begin
            errs++; $display("[%t] %s: accessed %0.3f ms after H_RESET (< %0.1f ms)", $realtime, NAME,
                             ($realtime - t_reset) / 1.0e6, RESET_NS / 1.0e6);
        end
        t_fall = $realtime;
    end

    always @(posedge cs_n) if (present && in_frame) begin
        in_frame = 0;
        if (sclk !== 1'b0) begin errs++; $display("[%t] %s: NCS rose with SCLK high", $realtime, NAME); end
        if (nbit % 8 != 0 || nbit == 0) begin errs++; $display("[%t] %s: frame of %0d bits", $realtime, NAME, nbit); end
        n_frames++;
        if (nbit == 120 && addr == 8'h3B + 14) n_bursts++;
    end

    always @(posedge sclk) if (present && !cs_n) begin
        if ($realtime - t_fall < 500.0) begin
            errs++; $display("[%t] %s: SCLK low phase %0.1f ns (< 500)", $realtime, NAME, $realtime - t_fall);
        end
        if ($realtime - t_mosi < 100.0) begin
            errs++; $display("[%t] %s: MOSI changed %0.1f ns before SCLK rose", $realtime, NAME, $realtime - t_mosi);
        end
        t_rise = $realtime;
        sh = {sh[6:0], mosi};
        nbit++;
        if (nbit == 8) begin
            rd   = sh[7];
            addr = {1'b0, sh[6:0]};
        end else if (nbit % 8 == 0) begin
            if (!rd) begin
                regs[addr[6:0]] = sh;
                if (nwr < 256) begin wr_addr[nwr] = addr; wr_data[nwr] = sh; wr_time[nwr] = $realtime; end
                nwr++;
                if (addr == 8'h6B && sh[7]) begin
                    hw_reset();
                    t_reset = $realtime;
                end
            end
            addr = addr + 8'd1;
        end
    end

    always @(negedge sclk) if (present && !cs_n) begin
        if ($realtime - t_rise < 500.0) begin
            errs++; $display("[%t] %s: SCLK high phase %0.1f ns (< 500)", $realtime, NAME, $realtime - t_rise);
        end
        t_fall = $realtime;
        if (nbit >= 8 && rd)                       // next bit of the byte being read
            miso_r = regs[addr[6:0]][7 - (nbit % 8)];
    end
endmodule
