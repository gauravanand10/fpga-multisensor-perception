// ============================================================================
// tb_servo_pwm.sv - servo_ctrl + servo_pwm with the real parameters (100 MHz,
// 20 ms period, 45..135 deg, slew 20 us / period, sweep 1 deg / 2 periods)
//
// Every pulse is measured (rising edge -> falling edge, rising -> rising) and
// compared with a model: at every period start  now = slew(now, target(angle)).
//   1 after reset: 20.000 ms period, 1500 us (90 deg)
//   2 manual 45 / 135 / 90 deg; switches below / above the limits (clamped); 0 = centre
//   3 slew limit: no pulse changes by more than 20 us
//   4 angle changed in the middle of a pulse and in the middle of a period:
//     the running pulse is not cut or stretched (checked against the model)
//   5 sweep: turns exactly at 1000 / 2000 us, never outside, small steps
//   6 external source (mode 2)
// ============================================================================
`timescale 1ns/1ps

module tb_servo_pwm;
    localparam int AMIN = 45, AMAX = 135, SLEW = 20;
    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst = 1'b1;
    logic sweep = 1'b0, ext_en = 1'b0;
    logic [7:0] sw_angle = 8'd0, ext_angle = 8'd0;
    logic [7:0] angle, mode;
    logic pwm, pstart;
    logic [15:0] target_us, now_us;

    servo_ctrl #(.ANGLE_MIN(AMIN), .ANGLE_MAX(AMAX), .SWEEP_PERIODS(2)) u_ctrl (
        .clk, .rst, .sweep, .sw_angle, .ext_en, .ext_angle, .period_start(pstart), .angle, .mode);
    servo_pwm #(.ANGLE_MIN(AMIN), .ANGLE_MAX(AMAX), .SLEW_US(SLEW)) u_pwm (
        .clk, .rst, .angle_deg(angle), .pwm, .period_start(pstart), .target_us, .now_us);

    int errs = 0;

    // second instance, full travel 0..180 deg, to measure the 1000 / 2000 us ends
    logic [7:0]  ang_full = 8'd90;
    logic        pwm_full, ps_full;
    logic [15:0] tg_full, now_full;
    servo_pwm #(.ANGLE_MIN(0), .ANGLE_MAX(180), .SLEW_US(SLEW)) u_full (
        .clk, .rst, .angle_deg(ang_full), .pwm(pwm_full), .period_start(ps_full),
        .target_us(tg_full), .now_us(now_full));
    int full_ok = 0;
    task automatic full_check(input int a, input int us);
        realtime tr, tp, tw;
        ang_full = 8'(a);
        repeat (60) @(posedge pwm_full);
        @(posedge pwm_full); tp = $realtime;
        @(negedge pwm_full); tw = $realtime - tp;
        @(posedge pwm_full); tr = $realtime - tp;
        if (tw != real'(us) * 1000.0 || tr != 20_000_000.0) begin
            errs++; $display("FAIL full-range %0d deg: pulse %0.3f us, period %0.3f ms", a, tw / 1000.0, tr / 1.0e6);
        end else begin
            full_ok++; $display("[%t] full-range instance %0d deg: pulse %0.3f us, period %0.3f ms", $realtime, a, tw / 1000.0, tr / 1.0e6);
        end
    endtask
    initial begin
        @(negedge rst);
        full_check(0, 1000);
        full_check(180, 2000);
        full_check(90, 1500);
    end

    // ---------------- model ----------------
    function automatic int tgt_of(input int a, input bit ext);
        int c;
        if (!ext && a == 0) c = 90; else c = a;
        if (c < AMIN) c = AMIN;
        if (c > AMAX) c = AMAX;
        return 1000 + (c * 1000 + 90) / 180;          // = round(c * 1000 / 180)
    endfunction

    int  m_now = 1500;          // model pulse width
    bit  model_on = 1'b1;       // off in sweep mode (property checks there)
    int  ang_cmd = 0;           // what the testbench set
    bit  ang_ext = 1'b0;

    // ---------------- measurement ----------------
    realtime t_rise = 0, t_prev = 0;
    int      n_pulse = 0, last_w = 1500, w;
    int      w_min = 99999, w_max = 0, max_step = 0;
    int      t_m, dstep;
    bit      seen_max = 0, seen_min = 0;

    always @(posedge pwm) if (!rst) begin
        t_prev = t_rise; t_rise = $realtime;
        if (n_pulse > 0) begin
            if (t_rise - t_prev != 20_000_000.0) begin
                errs++; $display("[%t] period %0.1f ns (expected 20 000 000)", $realtime, t_rise - t_prev);
            end
        end
        if (model_on) begin
            t_m = tgt_of(ang_cmd, ang_ext);
            if      (t_m > m_now + SLEW) m_now = m_now + SLEW;
            else if (t_m < m_now - SLEW) m_now = m_now - SLEW;
            else                         m_now = t_m;
        end
    end
    always @(negedge pwm) if (!rst) begin
        w = int'(($realtime - t_rise) / 1000.0);
        if ((($realtime - t_rise) / 1000.0) != real'(w)) begin
            errs++; $display("[%t] pulse %0.3f us is not a whole number of us", $realtime, ($realtime - t_rise) / 1000.0);
        end
        if (model_on && w != m_now) begin
            errs++; $display("[%t] pulse %0d us, model %0d us", $realtime, w, m_now);
        end
        dstep = w - last_w;
        if (dstep < 0) dstep = -dstep;
        if (n_pulse > 0 && dstep > SLEW) begin
            errs++; $display("[%t] pulse changed by %0d us (> %0d)", $realtime, dstep, SLEW);
        end
        if (n_pulse > 0 && dstep > max_step) max_step = dstep;
        if (w < 1250 || w > 1750) begin errs++; $display("[%t] pulse %0d us outside 1250..1750 (45..135 deg)", $realtime, w); end
        if (w < w_min) w_min = w;
        if (w > w_max) w_max = w;
        if (w == 1750) seen_max = 1;
        if (w == 1250) seen_min = 1;
        last_w = w;
        n_pulse++;
    end

    task automatic periods(input int n);
        repeat (n) @(posedge pwm);
    endtask

    task automatic set_angle(input int a);
        sw_angle = 8'(a); ang_cmd = a;
    endtask

    // waits until the pulse has settled, then checks it
    task automatic expect_settled(input string what, input int us);
        int guard = 0;
        while (last_w != us && guard < 80) begin @(negedge pwm); guard++; end
        @(negedge pwm); @(negedge pwm);
        if (last_w != us) begin errs++; $display("FAIL %s: pulse %0d us, expected %0d", what, last_w, us); end
        else $display("[%t] %-34s pulse %0d us, period 20.000 ms, after %0d periods", $realtime, what, last_w, guard);
    endtask

    initial begin
        $timeformat(-3, 2, " ms", 10);
        $display("servo_pwm: 100 MHz, 20 ms period, limits %0d..%0d deg, slew %0d us/period", AMIN, AMAX, SLEW);
        repeat (20) @(posedge clk);
        rst = 1'b0;
        // 1. centre after reset (switches all 0 = centre)
        periods(1);
        expect_settled("after reset (centre)", 1500);
        // 2. manual
        set_angle(45);   expect_settled("manual 45 deg", 1250);
        set_angle(135);  expect_settled("manual 135 deg (slew 25 periods)", 1750);
        set_angle(90);   expect_settled("manual 90 deg", 1500);
        set_angle(10);   expect_settled("switch 10 deg -> clamp 45", 1250);
        set_angle(250);  expect_settled("switch 250 deg -> clamp 135", 1750);
        set_angle(100);  expect_settled("manual 100 deg", 1556);
        // 4. glitch tests: change in the middle of a pulse and in the middle of a period
        @(posedge pwm); #(500_000);     set_angle(60);    // 0.5 ms into a 1.556 ms pulse
        @(negedge pwm); #(5_000_000);   set_angle(120);   // middle of the low part
        @(posedge pwm); #(10);          set_angle(80);    // just after the rising edge
        expect_settled("mid-pulse / mid-period changes", 1444);
        set_angle(0);    expect_settled("switches 0 -> centre", 1500);
        if (mode != 8'd0) begin errs++; $display("mode %0d, expected 0 (manual)", mode); end
        // 5. sweep
        $display("[%t] sweep on", $realtime);
        model_on = 0; seen_max = 0; seen_min = 0; w_min = 99999; w_max = 0; max_step = 0;
        sweep = 1'b1;
        repeat (3) @(posedge clk);
        if (mode != 8'd1) begin errs++; $display("mode %0d, expected 1 (sweep)", mode); end
        while (!seen_max) @(negedge pwm);
        $display("[%t] sweep reached 1750 us (135 deg)", $realtime);
        while (!seen_min) @(negedge pwm);
        $display("[%t] sweep reached 1250 us (45 deg)", $realtime);
        periods(20);
        $display("[%t] sweep: min %0d us, max %0d us, largest step %0d us", $realtime, w_min, w_max, max_step);
        if (w_min != 1250 || w_max != 1750 || max_step > 6) begin errs++; $display("sweep range/steps wrong"); end
        sweep = 1'b0;
        // back to manual from wherever the sweep was
        @(negedge pwm); m_now = last_w; model_on = 1;
        set_angle(90);   expect_settled("sweep off -> manual 90", 1500);
        // 6. external source wins
        ext_angle = 8'd100; ext_en = 1'b1; ang_ext = 1'b1; ang_cmd = 100;
        expect_settled("external source 100 deg", 1556);
        if (mode != 8'd2) begin errs++; $display("mode %0d, expected 2 (external)", mode); end
        $display("--------------------------------------------------");
        $display("%0d pulses measured", n_pulse);
        if (full_ok != 3) begin errs++; $display("full-range checks: %0d of 3", full_ok); end
        if (errs == 0) $display("PASS: servo period 20.000 ms, pulses 1250/1500/1750 us (45/90/135 deg), 1000/2000 us (0/180 deg), clamp, slew, no glitch, sweep, ext");
        else           $display("FAIL: %0d errors", errs);
        $finish;
    end

    initial begin #30s; $display("TIMEOUT"); $finish; end
endmodule
