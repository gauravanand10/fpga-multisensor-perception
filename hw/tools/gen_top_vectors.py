"""
gen_top_vectors.py - stimulus + expected output for tb/tb_nexys4_top.sv (whole FPGA)

Sequence the testbench plays on the Pmod pins:
  ToF packet 0, camera packet 0 (golden frame, SW0 = 1 -> image in the packet)  -> UART packet 0
  camera packet 1 with a wrong checksum                                         -> rejected
  ToF packet 1, camera packet 2 (golden frame mirrored, SW0 = 0)                -> UART packet 1

Files in mem/: tb_cam<i>.mem, tb_tof<i>.mem (SPI bytes), tb_uart<j>.mem (expected UART bytes;
the testbench ignores the two cycle counters, offsets 10..17, and checks the sum16 itself).

    python tools\\gen_top_vectors.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fpga_model as fm
import gen_pp_vectors as gv

REPO = os.path.dirname(ROOT)                                  # repository root
PC = os.path.join(HERE, "tof_geometry")                      # ToF geometry package
MEM = os.path.join(ROOT, "mem")


def wbytes(name, b):
    with open(os.path.join(MEM, name), "w") as f:
        f.writelines("%02x\n" % v for v in b)


def main():
    pc = sys.argv[1] if len(sys.argv) > 1 else PC
    gdir = os.path.join(REPO, "golden_model", "data", "golden_vectors")
    npz = np.load(os.path.join(REPO, "golden_model", "data", "model", "tiny_unet_int8_quantised.npz"))
    pk = fm.pack_network(os.path.join(gdir, "mem"), npz["gray__s"])
    zt = fm.load_zone_tables(os.path.join(MEM, "zone_tables.npz"))
    frame0 = np.array([int(t, 16) for t in open(os.path.join(MEM, "tb_frame.mem")).read().split()],
                      np.uint8).reshape(fm.IMG_H, fm.IMG_W)
    frame2 = frame0[:, ::-1].copy()
    rng = np.random.default_rng(5)

    _, lab0 = fm.run_unet(frame0, pk)
    d0, s0 = gv.synth_tof(lab0.ravel(), zt, rng)
    _, lab2 = fm.run_unet(frame2, pk)
    d1, s1 = gv.synth_tof(lab2.ravel(), zt, rng)

    wbytes("tb_tof0.mem", fm.tof_packet(d0, s0, 100))
    wbytes("tb_cam0.mem", fm.cam_packet(frame0, 7))
    bad = bytearray(fm.cam_packet(frame0[::-1].copy(), 8)); bad[-1] ^= 0x5A
    wbytes("tb_cam1.mem", bad)
    wbytes("tb_tof1.mem", fm.tof_packet(d1, s1, 101))
    wbytes("tb_cam2.mem", fm.cam_packet(frame2, 9))

    out = []
    for j, (fr, lab, d, s, cam_no, tof_no, img) in enumerate(
            [(frame0, lab0, d0, s0, 7, 100, True), (frame2, lab2, d1, s1, 9, 101, False)]):
        objs, kmap, flags, aux = fm.postprocess(lab, d, s, zt)
        pkt = fm.uart_packet(cam_no, tof_no, True, 0, 0, objs, kmap, d, s, fr if img else None,
                             flags_extra=4 * flags["overflow"])
        wbytes(f"tb_uart{j}.mem", pkt)
        out.append(pkt)
        print(f"UART packet {j}: {len(pkt)} bytes, {len(objs)} objects "
              f"{[(fm.CLASSES[o.cls], o.area, o.dist) for o in objs]}")
    with open(os.path.join(MEM, "tb_uart_len.mem"), "w") as f:
        f.write("%x\n%x\n" % (len(out[0]), len(out[1])))


if __name__ == "__main__":
    main()
