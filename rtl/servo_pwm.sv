// ============================================================================
// servo_pwm.sv - hobby-servo PWM (50 Hz) + angle source selection
// Target: Nexys4 DDR, 100 MHz.  Output on Pmod JD1.
//
// servo_pwm  : 20.000 ms period, pulse = 1000 + angle*1000/180 us (0..180 deg).
//              angle -> us with a constant multiply (x22756 >> 12, exact rounding
//              for 0..180), no divider. The pulse width (now_us) changes only at
//              the start of a period and by at most SLEW_US per period.
//              Angle clamped to [ANGLE_MIN, ANGLE_MAX]; centre (90 deg) after reset.
// servo_ctrl : picks the angle: manual (switches), sweep (between the limits),
//              or an external source (ext_en / ext_angle, for a later PC command or
//              an IMU-based angle; tied off for now).
// ============================================================================
`timescale 1ns/1ps

module servo_pwm #(
    parameter int CLK_PER_US = 100,           // 100 MHz
    parameter int PERIOD_US  = 20000,         // 20.000 ms
    parameter int ANGLE_MIN  = 45,            // safe travel (camera PCB mounted on the horn)
    parameter int ANGLE_MAX  = 135,
    parameter int SLEW_US    = 20             // max pulse change per period (~3.6 deg / 20 ms)
)(
    input  logic        clk,
    input  logic        rst,
    input  logic [7:0]  angle_deg,            // wanted angle in degrees (any value, clamped)
    output logic        pwm,
    output logic        period_start,         // 1-cycle pulse when a new period starts
    output logic [15:0] target_us,            // pulse width for the wanted angle
    output logic [15:0] now_us                // pulse width in the current period
);
    localparam int CW = $clog2(CLK_PER_US);
    localparam int UW = $clog2(PERIOD_US);

    // ---- angle -> target pulse width: clamp | multiply | shift+offset (3 stages) ----
    logic [7:0]  a_c;
    logic [22:0] prod;
    always_ff @(posedge clk) begin
        if (rst) begin
            a_c <= 8'd90; prod <= 23'(90 * 22756); target_us <= 16'd1500;
        end else begin
            if      (angle_deg < 8'(ANGLE_MIN)) a_c <= 8'(ANGLE_MIN);
            else if (angle_deg > 8'(ANGLE_MAX)) a_c <= 8'(ANGLE_MAX);
            else                                a_c <= angle_deg;
            prod      <= 23'(a_c) * 23'd22756;
            target_us <= 16'd1000 + 16'((prod + 23'd2048) >> 12);
        end
    end

    // ---- period: 1 us prescaler + us counter ----
    logic [CW-1:0] pc;
    logic [UW-1:0] us;
    logic          wrap;
    assign wrap = (pc == CW'(CLK_PER_US - 1)) && (us == UW'(PERIOD_US - 1));

    // slew limits, registered (now_us is constant for a whole period)
    logic [15:0] up_r, dn_r;
    always_ff @(posedge clk) begin
        if (rst) begin
            pc <= '0; us <= '0; now_us <= 16'd1500; pwm <= 1'b0; period_start <= 1'b0;
            up_r <= 16'(1500 + SLEW_US); dn_r <= 16'(1500 - SLEW_US);
        end else begin
            period_start <= wrap;
            up_r <= now_us + 16'(SLEW_US);
            dn_r <= now_us - 16'(SLEW_US);
            if (pc == CW'(CLK_PER_US - 1)) begin
                pc <= '0;
                if (us == UW'(PERIOD_US - 1)) us <= '0;
                else                          us <= us + 1'b1;
            end else
                pc <= pc + 1'b1;
            if (wrap) begin                    // new pulse width only here
                if      (target_us > up_r) now_us <= up_r;
                else if (target_us < dn_r) now_us <= dn_r;
                else                       now_us <= target_us;
            end
            pwm <= (16'(us) < now_us) && !wrap;  // high for now_us microseconds from us = 0
        end
    end
endmodule


module servo_ctrl #(
    parameter int ANGLE_MIN     = 45,
    parameter int ANGLE_MAX     = 135,
    parameter int SWEEP_PERIODS = 2           // periods per 1-degree step (2 -> 25 deg/s)
)(
    input  logic       clk,
    input  logic       rst,
    input  logic       sweep,                 // SW1: 1 = sweep between the limits
    input  logic [7:0] sw_angle,              // SW[15:8], degrees; 0 = centre (90)
    input  logic       ext_en,                // external angle source (future), wins over both
    input  logic [7:0] ext_angle,
    input  logic       period_start,          // from servo_pwm
    output logic [7:0] angle,                 // to servo_pwm
    output logic [7:0] mode                   // 0 manual, 1 sweep, 2 external
);
    localparam int DW = (SWEEP_PERIODS > 1) ? $clog2(SWEEP_PERIODS) : 1;

    logic [7:0] man;                          // manual angle, clamped
    always_ff @(posedge clk) begin
        if (rst)                              man <= 8'd90;
        else if (sw_angle == 8'd0)            man <= 8'd90;
        else if (sw_angle < 8'(ANGLE_MIN))    man <= 8'(ANGLE_MIN);
        else if (sw_angle > 8'(ANGLE_MAX))    man <= 8'(ANGLE_MAX);
        else                                  man <= sw_angle;
    end

    logic [7:0]    swp;
    logic          dir_up;
    logic [DW-1:0] div;
    always_ff @(posedge clk) begin
        if (rst) begin
            swp <= 8'd90; dir_up <= 1'b1; div <= '0;
        end else if (!sweep || ext_en) begin
            swp <= man; div <= '0;            // sweep starts from where the servo is
        end else if (period_start) begin
            if (div == DW'(SWEEP_PERIODS - 1)) begin
                div <= '0;
                if (dir_up) begin
                    if (swp >= 8'(ANGLE_MAX)) begin dir_up <= 1'b0; swp <= swp - 8'd1; end
                    else swp <= swp + 8'd1;
                end else begin
                    if (swp <= 8'(ANGLE_MIN)) begin dir_up <= 1'b1; swp <= swp + 8'd1; end
                    else swp <= swp - 8'd1;
                end
            end else
                div <= div + 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin angle <= 8'd90; mode <= 8'd0; end
        else if (ext_en) begin angle <= ext_angle; mode <= 8'd2; end
        else if (sweep)  begin angle <= swp;       mode <= 8'd1; end
        else             begin angle <= man;       mode <= 8'd0; end
    end
endmodule
