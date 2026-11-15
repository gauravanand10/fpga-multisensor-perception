"""
gen_imu_vectors.py - stimulus + expected bytes for tb/tb_imu_packer.sv (IMU FIFO + packer)

Four packets, each started after its samples were pushed into the FIFO:
  A  1 sample,  both IMUs ok,           servo manual 1500/1500 us
  B  5 samples, IMU1 ok only,           servo sweep 1600/1580 us
  C  34 samples pushed into the 32-deep FIFO: 2 dropped -> n = 32, overflow flag
  D  3 samples, overflow flag cleared again, no IMU ok, WHO_AM_I FF FF, servo external
sample_no runs on from 65530 (wraps through 0); the two dropped samples leave a gap of 2.

Files in mem/:
  tb_imu_samples.mem  one 208-bit sample per line {gz2 .. ax1, sample_no} (as imu_spi.sv)
  tb_imu_status.mem   per packet: {imu_ok u8, who1, who2, servo_mode, target_us u16, now_us u16}
  tb_imu_npush.mem    samples pushed per packet
  tb_imu_exp.mem      expected UART bytes of the 4 packets, concatenated
  tb_imu_len.mem      length of each expected packet

    python hw\\tools\\gen_imu_vectors.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fpga_model as fm

MEM = os.path.join(ROOT, "mem")


def sample_word(sno, v):
    w = sno & 0xFFFF
    for k, x in enumerate(v):
        w |= (int(x) & 0xFFFF) << (16 * (k + 1))
    return w


def main():
    rng = np.random.default_rng(2026)
    cases = [  # (npush, imu_ok, who1, who2, mode, target, now)
        (1, 3, 0x70, 0x71, 0, 1500, 1500),
        (5, 1, 0x70, 0xFF, 1, 1600, 1580),
        (34, 3, 0x71, 0x73, 0, 1250, 1270),
        (3, 0, 0xFF, 0xFF, 2, 1750, 1730),
    ]
    sno = 65530
    words, status, npush, exp, lens = [], [], [], bytearray(), []
    ovf = False
    for (n, ok, w1, w2, mode, tgt, now) in cases:
        smp = []
        for i in range(n):
            v = [int(x) for x in rng.integers(-32768, 32768, 12)]
            if i == 0:
                v[0], v[1] = -32768, 32767             # extremes
            words.append(sample_word(sno, v))
            smp.append((sno & 0xFFFF, v))
            sno += 1
        kept = smp[:fm.IMU_FIFO_DEPTH]
        ovf_now = ovf or n > fm.IMU_FIFO_DEPTH
        pkt = fm.imu_packet(ok, w1, w2, mode, tgt, now, kept, overflow=ovf_now)
        ovf = False                                    # reported -> cleared
        d, ln = fm.parse_imu_packet(pkt)
        assert ln == len(pkt) and d["samples"] == kept
        exp += pkt
        lens.append(len(pkt))
        npush.append(n)
        status.append((ok << 56) | (w1 << 48) | (w2 << 40) | (mode << 32) | (tgt << 16) | now)
        print(f"packet: {n} pushed, n = {len(kept)}, flags {pkt[5]:02x}, {len(pkt)} bytes")

    with open(os.path.join(MEM, "tb_imu_samples.mem"), "w") as f:
        f.writelines("%052x\n" % w for w in words)
    with open(os.path.join(MEM, "tb_imu_status.mem"), "w") as f:
        f.writelines("%016x\n" % s for s in status)
    with open(os.path.join(MEM, "tb_imu_npush.mem"), "w") as f:
        f.writelines("%x\n" % n for n in npush)
    with open(os.path.join(MEM, "tb_imu_exp.mem"), "w") as f:
        f.writelines("%02x\n" % b for b in exp)
    with open(os.path.join(MEM, "tb_imu_len.mem"), "w") as f:
        f.writelines("%x\n" % n for n in lens)
    print(f"{len(words)} samples, {len(exp)} expected bytes -> mem/tb_imu_*.mem")


if __name__ == "__main__":
    main()
