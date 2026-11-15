"""
check_capture.py - run both PC parsers over a captured UART stream (e.g. the top-level simulation)

 1. the legacy result-packet reader: the PacketReader class of sw/result_viewer.py (imported unchanged,
    together with hw/tools/fpga_model.py)
    -> must give the expected number of result packets and 0 bad packets; the IMU packets
    in between must be skipped silently
 2. the new parser (tools\\fpga_model.py parse_stream / parse_imu_packet)
    -> every byte belongs to a complete packet, 0 bad, result packets identical to (1)

    python hw\\tools\\check_capture.py build\\xsim_top\\uart_capture.bin
Exit code 0 = PASS. The last line printed is the PASS / FAIL summary.
"""
import argparse, importlib.util, os, sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
LEGACY_VIEWER = os.path.join(REPO, "sw", "result_viewer.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--expect-results", type=int, default=2)
    a = ap.parse_args()
    data = open(a.capture, "rb").read()
    print(f"{a.capture}: {len(data)} bytes")
    errs = []

    # ---- 1. legacy result-packet reader ----
    old_viewer = load("old_fpga_viewer", LEGACY_VIEWER)
    seen = []

    class Rec(old_viewer.PacketReader):                       # only records each packet; run() is unchanged
        @property
        def latest(self):
            return self._latest

        @latest.setter
        def latest(self, v):
            self._latest = v
            if v is not None:
                seen.append(v)

    with open(a.capture, "rb") as f:
        rd = Rec(f, replay_fps=1e9)
        rd._latest = None
        rd.run()                                              # synchronous, reads to EOF
    print(f"legacy result-packet reader: {rd.n_ok} ok, {rd.n_bad} bad "
          f"(cam_no {[p['cam_no'] for p in seen]}, objects {[len(p['objs']) for p in seen]})")
    if rd.n_ok != a.expect_results or rd.n_bad != 0:
        errs.append(f"old reader {rd.n_ok} ok / {rd.n_bad} bad, expected {a.expect_results} / 0")

    # ---- 2. new parser ----
    fm = load("fpga_model_new", os.path.join(HERE, "fpga_model.py"))
    items, consumed, n_bad = fm.parse_stream(bytearray(data))
    res = [p for k, p in items if k == "FP"]
    imu = [p for k, p in items if k == "IM"]
    # strict walk: packets must follow each other with nothing in between
    pos, gaps = 0, 0
    while pos < len(data):
        sync = data[pos:pos + 4]
        if sync == fm.UART_SYNC:
            _, n = fm.parse_uart_packet(data[pos:])
        elif sync == fm.IMU_SYNC:
            _, n = fm.parse_imu_packet(data[pos:])
        else:
            gaps += 1; pos += 1; continue
        if _ is None:
            break                                             # incomplete packet at the end
        pos += n
    tail = len(data) - pos
    nsmp = sum(len(p["samples"]) for p in imu)
    snos = [s[0] for p in imu for s in p["samples"]]
    lost = sum(((b - a_) & 0xFFFF) - 1 for a_, b in zip(snos, snos[1:]))
    ok1 = sum(1 for p in imu if p["imu_ok"] & 1)
    print(f"new parser: {len(res)} result + {len(imu)} IMU packets ({nsmp} samples, {lost} lost, IMU1 ok in {ok1}), "
          f"{n_bad} bad, {gaps} bytes between packets, {tail} bytes of an unfinished packet at the end")
    if imu:
        p = imu[-1]
        print(f"   last IMU packet: flags {p['flags']:#04x}, WHO_AM_I {p['who1']:#04x} / {p['who2']:#04x}, "
              f"servo {fm.SERVO_MODES.get(p['servo_mode'], p['servo_mode'])} target {p['servo_target_us']} us "
              f"now {p['servo_now_us']} us, sample {p['samples'][-1]}")
    if n_bad or gaps or len(res) != a.expect_results or not imu or lost:
        errs.append("new parser: stream not clean")
    for i, (po, pn) in enumerate(zip(seen, res)):
        same = (po["cam_no"] == pn["cam_no"] and po["flags"] == pn["flags"] and len(po["objs"]) == len(pn["objs"])
                and (po["kmap"] == pn["kmap"]).all() and (po["tof_dist"] == pn["tof_dist"]).all())
        if not same:
            errs.append(f"result packet {i} differs between the parsers")

    if errs:
        for e in errs:
            print("  ", e)
        print("FAIL: " + "; ".join(errs))
        sys.exit(1)
    print(f"PASS: legacy reader {rd.n_ok} ok / 0 bad; new parser {len(res)} result + {len(imu)} IMU packets, 0 bad")


if __name__ == "__main__":
    main()
