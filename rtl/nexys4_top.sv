// ============================================================================
// nexys4_top.sv - board top level for the Digilent Nexys4 DDR (xc7a100tcsg324-1)
// Pin names = Digilent Nexys-4-DDR-Master.xdc (constr/nexys4_imu_servo.xdc).
//
//   JA (camera SPI from the ESP32-CAM)   JA1 CS_N   JA2 MOSI   JA4 SCLK   (JA5/JA11 GND)
//   JB (ToF SPI from the ToF ESP32)      JB1 CS_N   JB2 MOSI   JB4 SCLK   (JB5/JB11 GND)
//   JC (two MPU-6500/9250, FPGA = SPI master, 500 kHz)
//        JC1 CS1_N -> IMU1 NCS     JC2 MISO <- AD0 of both    JC3 MOSI -> SDA of both
//        JC4 SCLK  -> SCL of both  JC5 GND   JC6 3V3          JC7 CS2_N -> IMU2 NCS
//   JD1 servo signal (50 Hz); servo powered from a separate 5 V supply, GND common (JD5)
//   UART_RXD_OUT (D4) -> USB-UART -> PC  2 Mbaud 8N1 (result packets + IMU packets)
//   SW0      = include the camera image in each result packet
//   SW1      = servo sweep between the limits (off = manual)
//   SW[15:8] = manual servo angle in degrees (binary, clamped to 45..135; 0 = centre 90)
//   CPU_RESETN = reset
// ============================================================================
`timescale 1ns/1ps

module nexys4_top #(
    parameter int    CLKS_PER_BIT    = 50,
    parameter string MEM_DIR         = "",
    // IMU timing (simulation may shorten the tick and the waits; the board uses the defaults)
    parameter int    IMU_HALF_CYC    = 100,          // 500 kHz SCLK
    parameter int    IMU_TICK_CYC    = 1_000_000,    // 100 Hz
    parameter int    IMU_PWRUP_TICKS = 10,
    parameter int    IMU_RESET_TICKS = 11,
    parameter int    IMU_RETRY_TICKS = 100,
    // servo
    parameter int    SERVO_ANGLE_MIN = 45,
    parameter int    SERVO_ANGLE_MAX = 135
)(
    input  logic        CLK100MHZ,
    input  logic        CPU_RESETN,
    input  logic [15:0] SW,
    output logic [15:0] LED,
    output logic        UART_RXD_OUT,
    input  logic [4:1]  JA,
    input  logic [4:1]  JB,
    output logic        JC_CS1_N,     // JC1
    input  logic        JC_MISO,      // JC2
    output logic        JC_MOSI,      // JC3
    output logic        JC_SCLK,      // JC4
    output logic        JC_CS2_N,     // JC7
    output logic        JD_SERVO      // JD1
);
    // reset: asynchronous button -> synchronous, active-high, ~1 ms stretched
    logic [2:0]  rsync = '0;               // power-up values (FPGA INIT)
    logic [16:0] rcnt  = '0;
    logic        rst   = 1'b1;
    always_ff @(posedge CLK100MHZ) begin
        rsync <= {rsync[1:0], CPU_RESETN};
        if (!rsync[2]) begin rcnt <= '0; rst <= 1'b1; end
        else if (rcnt != '1) begin rcnt <= rcnt + 17'd1; rst <= 1'b1; end
        else rst <= 1'b0;
    end

    logic [15:0] sw_s0, sw_s1;
    always_ff @(posedge CLK100MHZ) begin sw_s0 <= SW; sw_s1 <= sw_s0; end

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused;
    assign unused = ^{JA[3], JB[3], sw_s1[7:2]};    // JA3/JB3 = MISO, not driven by the FPGA
    /* verilator lint_on UNUSEDSIGNAL */

    // ---------------- IMUs on JC ----------------
    logic         imu_valid;
    logic [207:0] imu_sample;
    logic [1:0]   imu_ok;
    logic [7:0]   imu_who1, imu_who2;
    imu_spi #(.HALF_CYC(IMU_HALF_CYC), .TICK_CYC(IMU_TICK_CYC), .PWRUP_TICKS(IMU_PWRUP_TICKS),
              .RESET_TICKS(IMU_RESET_TICKS), .RETRY_TICKS(IMU_RETRY_TICKS)) u_imu (
        .clk(CLK100MHZ), .rst,
        .sclk(JC_SCLK), .mosi(JC_MOSI), .cs1_n(JC_CS1_N), .cs2_n(JC_CS2_N), .miso_i(JC_MISO),
        .sample_valid(imu_valid), .sample(imu_sample), .imu_ok, .who1(imu_who1), .who2(imu_who2));

    // ---------------- servo on JD1 ----------------
    logic [7:0]  servo_angle, servo_mode;
    logic        servo_pstart;
    logic [15:0] servo_target_us, servo_now_us;
    servo_ctrl #(.ANGLE_MIN(SERVO_ANGLE_MIN), .ANGLE_MAX(SERVO_ANGLE_MAX)) u_sctl (
        .clk(CLK100MHZ), .rst, .sweep(sw_s1[1]), .sw_angle(sw_s1[15:8]),
        .ext_en(1'b0), .ext_angle(8'd90),               // later: PC command / IMU angle
        .period_start(servo_pstart), .angle(servo_angle), .mode(servo_mode));
    servo_pwm #(.ANGLE_MIN(SERVO_ANGLE_MIN), .ANGLE_MAX(SERVO_ANGLE_MAX)) u_servo (
        .clk(CLK100MHZ), .rst, .angle_deg(servo_angle), .pwm(JD_SERVO), .period_start(servo_pstart),
        .target_us(servo_target_us), .now_us(servo_now_us));

    // ---------------- depth-map system (+ IMU packets on the same UART) ----------------
    unet_system #(.CLKS_PER_BIT(CLKS_PER_BIT), .MEM_DIR(MEM_DIR)) u_sys (
        .clk(CLK100MHZ), .rst,
        .cam_sclk(JA[4]), .cam_mosi(JA[2]), .cam_csn(JA[1]),
        .tof_sclk(JB[4]), .tof_mosi(JB[2]), .tof_csn(JB[1]),
        .uart_txd(UART_RXD_OUT),
        .sw_image(sw_s1[0]),
        .led(LED),
        .imu_sample_valid(imu_valid), .imu_sample, .imu_ok, .imu_who1, .imu_who2,
        .servo_mode, .servo_target_us, .servo_now_us);
endmodule
