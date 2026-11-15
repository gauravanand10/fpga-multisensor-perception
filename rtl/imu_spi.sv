// ============================================================================
// imu_spi.sv - SPI master for two MPU-6500 / MPU-9250 IMUs on one bus (Pmod JC)
// Target: Nexys4 DDR, 100 MHz.
//
//   JC1 CS1_N  JC2 MISO (AD0/SDO of both)  JC3 MOSI (SDA/SDI)  JC4 SCLK (SCL)  JC7 CS2_N
//
// SPI mode 0: SCLK idles low, MOSI changes while SCLK is low, MISO is sampled on
// the rising edge (through a 2-FF synchroniser; at 500 kHz the data has been
// stable for ~1 us when it is sampled). MSB first. Read = address | 0x80.
// All outputs (SCLK, MOSI, CS_N) come straight from flip-flops.
//
// Every 10 ms (TICK_CYC) one "round" runs:  IMU1 job, IMU2 job, latch sample.
// Job of one IMU, by its state:
//   NEED  : wait (power-up 100 ms, or the 1 s retry time), then
//           write PWR_MGMT_1 (0x6B) = 0x80 (H_RESET)                -> RWAIT
//   RWAIT : wait 110 ms (>= 100 ms after the reset write, with margin), then
//           0x6B = 0x01 (clock: auto PLL), USER_CTRL 0x6A = 0x10 (I2C_IF_DIS: SPI only),
//           CONFIG 0x1A = 0x03 (gyro DLPF 41 Hz), GYRO_CONFIG 0x1B = 0x00 (+-250 dps),
//           ACCEL_CONFIG 0x1C = 0x00 (+-2 g), ACCEL_CONFIG2 0x1D = 0x03 (accel DLPF 41 Hz),
//           read WHO_AM_I 0x75: 0x70 MPU-6500, 0x71 MPU-9250, 0x73 MPU-9255, 0x68 MPU-6000
//           -> OK (and burst-read at once), else -> NEED (retry after RETRY_TICKS)
//   OK    : read WHO_AM_I (detects an unplugged IMU), then burst-read 14 bytes from
//           ACCEL_XOUT_H 0x3B (accel 6, temperature 2, gyro 6, high byte first).
//           Wrong WHO_AM_I -> NEED (values become 0).
// A missing IMU reads 0xFF (pull-up on MISO): it is reported as not OK and never
// blocks anything; the other IMU and the sample rate are not affected.
//
// sample (208 bit, all latched together, sample_valid = 1-cycle pulse, 100 Hz):
//   [15:0] sample_no, then ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2 (16 bit each,
//   ax1 in [31:16] ... gz2 in [207:192]); values of an IMU that is not OK are 0.
// ============================================================================
`timescale 1ns/1ps

module imu_spi #(
    parameter int HALF_CYC    = 100,          // SCLK half period: 100 -> 500 kHz
    parameter int TICK_CYC    = 1_000_000,    // 10 ms -> 100 Hz samples
    parameter int PWRUP_TICKS = 10,           // ~100 ms after configuration before the first reset
    parameter int RESET_TICKS = 11,           // 110 ms after H_RESET (datasheet: >= 100 ms)
    parameter int RETRY_TICKS = 100           // ~1 s between attempts on a missing IMU
)(
    input  logic         clk,
    input  logic         rst,
    output logic         sclk,
    output logic         mosi,
    output logic         cs1_n,
    output logic         cs2_n,
    input  logic         miso_i,
    output logic         sample_valid,
    output logic [207:0] sample,
    output logic [1:0]   imu_ok,              // latched with the sample
    output logic [7:0]   who1,                // last WHO_AM_I read (0xFF = nothing answers)
    output logic [7:0]   who2
);
    // ------------------------------------------------------------------
    // MISO synchroniser
    // ------------------------------------------------------------------
    logic [1:0] miso_s;
    always_ff @(posedge clk) miso_s <= {miso_s[0], miso_i};

    // ------------------------------------------------------------------
    // SPI byte engine: one CS-low transaction of len bytes
    //   byte 0 = b0 (address), byte 1 = b1 (write data), then 0x00
    //   received bits shift into rxbuf (last 14 bytes kept)
    // ------------------------------------------------------------------
    localparam int CNTW = $clog2(2 * HALF_CYC + 1);
    typedef enum logic [2:0] {E_IDLE, E_LO, E_HI, E_CSH, E_GAP} est_t;
    est_t          es;
    logic [CNTW-1:0] ecnt;
    logic [6:0]    ebits;            // bits still to transfer
    logic [15:0]   txsh;
    logic [111:0]  rxbuf;
    logic          e_go, e_sel, e_done;
    logic [3:0]    e_len;
    logic [7:0]    e_b0, e_b1;

    always_ff @(posedge clk) begin
        e_done <= 1'b0;
        if (rst) begin
            es <= E_IDLE; sclk <= 1'b0; mosi <= 1'b0; cs1_n <= 1'b1; cs2_n <= 1'b1;
            ecnt <= '0; ebits <= '0; txsh <= '0; rxbuf <= '0;
        end else begin
            unique case (es)
            E_IDLE: if (e_go) begin
                cs1_n <= e_sel;                   // sel 0 -> IMU1
                cs2_n <= !e_sel;
                txsh  <= {e_b0[6:0], e_b1, 1'b0};
                mosi  <= e_b0[7];
                ebits <= {e_len, 3'b000};
                ecnt  <= '0;
                es    <= E_LO;                    // CS set-up = first low phase
            end
            E_LO: if (ecnt == CNTW'(HALF_CYC - 1)) begin
                ecnt  <= '0;
                sclk  <= 1'b1;
                rxbuf <= {rxbuf[110:0], miso_s[1]};
                es    <= E_HI;
            end else ecnt <= ecnt + 1'b1;
            E_HI: if (ecnt == CNTW'(HALF_CYC - 1)) begin
                ecnt  <= '0;
                sclk  <= 1'b0;
                ebits <= ebits - 7'd1;
                if (ebits == 7'd1) es <= E_CSH;
                else begin
                    mosi <= txsh[15];
                    txsh <= {txsh[14:0], 1'b0};
                    es   <= E_LO;
                end
            end else ecnt <= ecnt + 1'b1;
            E_CSH: if (ecnt == CNTW'(HALF_CYC - 1)) begin   // hold after the last edge
                ecnt <= '0; cs1_n <= 1'b1; cs2_n <= 1'b1; mosi <= 1'b0;
                es <= E_GAP;
            end else ecnt <= ecnt + 1'b1;
            E_GAP: if (ecnt == CNTW'(2 * HALF_CYC - 1)) begin   // CS high between transactions
                ecnt <= '0; e_done <= 1'b1; es <= E_IDLE;
            end else ecnt <= ecnt + 1'b1;
            default: es <= E_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // transaction table
    // ------------------------------------------------------------------
    localparam logic [3:0] STEP_RESET = 4'd0, STEP_CFG0 = 4'd1, STEP_ID_INIT = 4'd7,
                           STEP_ID = 4'd8, STEP_BURST = 4'd9;
    logic [3:0] step;
    always_comb begin
        e_b1 = 8'h00; e_len = 4'd2;
        unique case (step)
        4'd0:    begin e_b0 = 8'h6B; e_b1 = 8'h80; end     // PWR_MGMT_1: H_RESET
        4'd1:    begin e_b0 = 8'h6B; e_b1 = 8'h01; end     // PWR_MGMT_1: CLKSEL auto
        4'd2:    begin e_b0 = 8'h6A; e_b1 = 8'h10; end     // USER_CTRL: I2C_IF_DIS
        4'd3:    begin e_b0 = 8'h1A; e_b1 = 8'h03; end     // CONFIG: DLPF 41 Hz
        4'd4:    begin e_b0 = 8'h1B; e_b1 = 8'h00; end     // GYRO_CONFIG: +-250 dps
        4'd5:    begin e_b0 = 8'h1C; e_b1 = 8'h00; end     // ACCEL_CONFIG: +-2 g
        4'd6:    begin e_b0 = 8'h1D; e_b1 = 8'h03; end     // ACCEL_CONFIG2: DLPF 41 Hz
        4'd7, 4'd8: e_b0 = 8'hF5;                          // read WHO_AM_I (0x75 | 0x80)
        default: begin e_b0 = 8'hBB; e_len = 4'd15; end    // read 0x3B.. 14 bytes
        endcase
    end

    // ------------------------------------------------------------------
    // sequencer
    // ------------------------------------------------------------------
    localparam int TW = $clog2(TICK_CYC);
    logic [TW-1:0] tcnt;
    logic          tick_pend;

    typedef enum logic [1:0] {ST_NEED, ST_RWAIT, ST_OK} ist_t;
    ist_t       ist [2];
    logic [6:0] wcnt [2];
    logic [95:0] raw [2];
    logic [7:0] who [2];

    typedef enum logic [2:0] {Q_IDLE, Q_DISP, Q_RUN, Q_WAIT, Q_AFTER, Q_NEXT, Q_LATCH} qst_t;
    qst_t       qs;
    logic       cur;                 // IMU being served (0 = IMU1)
    logic [15:0] sno;
    logic       id_good;
    logic [7:0] rx_id;

    assign rx_id   = rxbuf[7:0];
    assign id_good = (rx_id == 8'h70) || (rx_id == 8'h71) || (rx_id == 8'h73) || (rx_id == 8'h68);

    always_ff @(posedge clk) begin
        e_go         <= 1'b0;
        sample_valid <= 1'b0;
        if (rst) begin
            tcnt <= '0; tick_pend <= 1'b0; qs <= Q_IDLE; cur <= 1'b0; step <= '0; e_sel <= 1'b0;
            sno <= '0; sample <= '0; imu_ok <= '0; who1 <= '0; who2 <= '0;
            for (int i = 0; i < 2; i++) begin
                ist[i] <= ST_NEED; wcnt[i] <= 7'(PWRUP_TICKS); raw[i] <= '0; who[i] <= '0;
            end
        end else begin
            if (tcnt == TW'(TICK_CYC - 1)) begin tcnt <= '0; tick_pend <= 1'b1; end
            else tcnt <= tcnt + 1'b1;

            unique case (qs)
            Q_IDLE: if (tick_pend) begin
                tick_pend <= 1'b0;              // (a tick in this same cycle is not lost:
                cur <= 1'b0;                    //  rounds take < 1 ms, ticks are 10 ms apart)
                qs  <= Q_DISP;
            end
            Q_DISP: begin
                e_sel <= cur;
                unique case (ist[cur])
                ST_OK: begin step <= STEP_ID; qs <= Q_RUN; end
                ST_NEED:
                    if (wcnt[cur] == 7'd0) begin step <= STEP_RESET; qs <= Q_RUN; end
                    else begin wcnt[cur] <= wcnt[cur] - 7'd1; qs <= Q_NEXT; end
                default:                        // ST_RWAIT
                    if (wcnt[cur] == 7'd0) begin step <= STEP_CFG0; qs <= Q_RUN; end
                    else begin wcnt[cur] <= wcnt[cur] - 7'd1; qs <= Q_NEXT; end
                endcase
            end
            Q_RUN: begin e_go <= 1'b1; qs <= Q_WAIT; end
            Q_WAIT: if (e_done) qs <= Q_AFTER;
            Q_AFTER: begin
                qs <= Q_NEXT;
                if (step == STEP_RESET) begin
                    ist[cur] <= ST_RWAIT; wcnt[cur] <= 7'(RESET_TICKS - 1);
                end else if (step < STEP_ID_INIT) begin
                    step <= step + 4'd1; qs <= Q_RUN;
                end else if (step == STEP_ID_INIT || step == STEP_ID) begin
                    who[cur] <= rx_id;
                    if (id_good) begin                  // read the values in this round too, so
                        ist[cur] <= ST_OK;              // an OK flag never comes with old values
                        step <= STEP_BURST; qs <= Q_RUN;
                    end else begin
                        ist[cur] <= ST_NEED; wcnt[cur] <= 7'(RETRY_TICKS - 1);
                    end
                end else                        // STEP_BURST
                    raw[cur] <= {rxbuf[15:0], rxbuf[31:16], rxbuf[47:32],       // gz gy gx
                                 rxbuf[79:64], rxbuf[95:80], rxbuf[111:96]};    // az ay ax
            end
            Q_NEXT: if (cur == 1'b0) begin cur <= 1'b1; qs <= Q_DISP; end
                    else qs <= Q_LATCH;
            Q_LATCH: begin
                sample <= {(ist[1] == ST_OK) ? raw[1] : 96'd0,
                           (ist[0] == ST_OK) ? raw[0] : 96'd0, sno};
                imu_ok <= {ist[1] == ST_OK, ist[0] == ST_OK};
                who1 <= who[0]; who2 <= who[1];
                sno  <= sno + 16'd1;
                sample_valid <= 1'b1;
                qs <= Q_IDLE;
            end
            default: qs <= Q_IDLE;
            endcase
        end
    end
endmodule
