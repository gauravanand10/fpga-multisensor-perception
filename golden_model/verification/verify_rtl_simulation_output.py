"""
verify_rtl_simulation_output.py - this Python model vs the SystemVerilog RTL (Vivado xsim)

The whole-FPGA testbench (tb/tb_nexys4_top.sv) drives SPI bytes into the Pmod pins
and records every byte the FPGA sends on its UART pin (build/xsim_top/uart_capture.bin). This script
feeds the SAME SPI bursts (hw/mem/tb_cam*.mem, tb_tof*.mem) through the reference model and
compares its UART packets with the captured ones byte by byte. The two cycle counters (offsets
10..17) are hardware measurements and only reported; the capture's checksum is checked.

    python verification\\verify_rtl_simulation_output.py
Output: results/verification/rtl_simulation_vs_model.txt / .json
"""
import os, struct
import numpy as np
from _common import save_report, read_hex_bytes
from reference_model import paths, fpga_model as fm, spi_link, uart_link, laptop_host
from reference_model.fpga_unet_accelerator import UNetAccelerator, CYCLES_PER_FRAME
from reference_model.fpga_postprocessing import PostProcessor


def main():
    mem = paths.need(paths.RTL_MEM, "hw/mem")
    cap = open(paths.need(paths.RTL_UART_CAPTURE, "golden_model/data/rtl_simulation"), "rb").read()
    acc, post = UNetAccelerator(), PostProcessor()
    cam_rx, tof_rx = spi_link.CameraReceiver(), spi_link.ToFReceiver()
    # same order as the testbench: ToF0, CAM0 (SW0 on), CAM1 (corrupted), ToF1, CAM2 (SW0 off)
    seq = [("tof", "tb_tof0.mem", None), ("cam", "tb_cam0.mem", True), ("cam", "tb_cam1.mem", True),
           ("tof", "tb_tof1.mem", None), ("cam", "tb_cam2.mem", False)]
    model_pkts, lines, rx_log = [], ["Python reference model vs RTL simulation (UART bytes from Vivado xsim)", ""], []
    for kind, fn, img in seq:
        burst = read_hex_bytes(os.path.join(mem, fn))
        r = (tof_rx if kind == "tof" else cam_rx).receive(burst)
        rx_log.append(dict(packet=fn, accepted=r.ok, reason=r.reason))
        lines.append(f"  SPI {fn:12s} {len(burst):6d} B  -> {'accepted' if r.ok else 'REJECTED (' + r.reason + ')'}")
        if kind == "cam" and r.ok:
            labels, _ = acc.run(cam_rx.frame)
            pr = post.run(labels, tof_rx.dist, tof_rx.status)
            model_pkts.append(uart_link.build_packet(cam_rx.frame_no, tof_rx.frame_no, tof_rx.valid, CYCLES_PER_FRAME,
                                                     0, pr, tof_rx.dist, tof_rx.status, cam_rx.frame if img else None))
    rtl = laptop_host.find_packets(cap)
    lines += ["", f"  UART capture: {len(cap)} bytes, {len(rtl)} checksum-valid packets; model: {len(model_pkts)} packets", ""]
    rows, ok = [], len(rtl) == len(model_pkts)
    for i, (m, (off, p, n)) in enumerate(zip(model_pkts, rtl)):
        c = cap[off:off + n]
        mask = [k for k in range(len(m)) if not (10 <= k < 18 or k >= len(m) - 2)]
        diff = sum(1 for k in mask if k >= len(c) or m[k] != c[k]) + abs(len(m) - len(c))
        icyc, pcyc = struct.unpack_from("<II", c, 10)
        rows.append(dict(packet=i, bytes_model=len(m), bytes_rtl=len(c), differing_bytes=diff,
                         rtl_unet_cycles=icyc, rtl_post_cycles=pcyc, n_obj=len(p["objs"])))
        ok &= diff == 0
        lines.append(f"  packet {i}: model {len(m)} B, RTL {len(c)} B, differing bytes {diff}, "
                     f"{len(p['objs'])} objects | RTL measured U-Net {icyc} cycles ({icyc / 1e5:.2f} ms), "
                     f"post-processing {pcyc} cycles ({pcyc / 1e5:.2f} ms)")
    ok &= [x["accepted"] for x in rx_log] == [True, True, False, True, True]
    lines += ["", ("PASS" if ok else "FAIL") + ": every UART byte of the RTL equals the Python model; "
              "corrupted camera packet rejected by both"]
    save_report("rtl_simulation_vs_model", lines, dict(spi=rx_log, packets=rows, passed=bool(ok)))


if __name__ == "__main__":
    main()
