"""
gen_pp_vectors.py - test cases for tb/tb_postproc.sv (expected results from fpga_model.py)

Case 0 : label map of the golden frame (mem/tb_labels.mem) + a synthetic ToF frame
Case 1+: synthetic label maps: ellipses of all 5 classes, touching blobs, U shapes and a
         spiral (many CC passes), noise specks (small components), invalid ToF zones
Last   : > 63 components (overflow flag)

Per case i (in mem/): tb_pp<i>_lab.mem (4800 x class), tb_pp<i>_tof.mem (64 x dist, 64 x status),
tb_pp<i>_exp.mem (n_obj, n_comp, overflow, then 63 x {valid cls area x0 y0 x1 y1 dist nsel},
then 4800 x object id).

    python tools\\gen_pp_vectors.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fpga_model as fm

MEM = os.path.join(ROOT, "mem")


def synth_tof(labels, zt, rng, near=900, far=2600, p_invalid=0.08):
    """zones mostly over objects -> near (per class offset), else far; some invalid."""
    dist = np.zeros(64, np.int64); status = np.full(64, 5, np.int64)
    lab = labels.reshape(fm.LAB_H, fm.LAB_W)
    for i in range(64):
        x0, y0, x1, y1, v = zt.rects[i, 24]
        if v:
            sub = lab[y0:y1 + 1, x0:x1 + 1]
            fg = sub[sub != 0]
            if fg.size > 0.3 * sub.size:
                c = int(np.bincount(fg).argmax())
                dist[i] = near + 150 * c + rng.integers(-60, 60)
            else:
                dist[i] = far + rng.integers(-200, 200)
        else:
            dist[i] = far + rng.integers(-200, 200)
        r = rng.random()
        if r < p_invalid:
            status[i] = 255
        elif r < 2 * p_invalid:
            status[i] = int(rng.choice([6, 9, 10, 12]))
    status[rng.integers(0, 64)] = 5
    dist[rng.integers(0, 64)] = 4500                     # out of range -> invalid
    return dist, status


def ellipse(lab, cx, cy, ax, ay, cls):
    yy, xx = np.mgrid[0:fm.LAB_H, 0:fm.LAB_W]
    lab[((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1] = cls


def synth_labels(rng, kind):
    lab = np.zeros((fm.LAB_H, fm.LAB_W), np.int64)
    if kind == "blobs":
        for _ in range(7):
            ellipse(lab, rng.integers(5, 75), rng.integers(5, 55), rng.integers(3, 16), rng.integers(3, 14),
                    int(rng.integers(1, 6)))
        noise = rng.random(lab.shape) < 0.01
        lab[noise] = rng.integers(1, 6, noise.sum())
    elif kind == "shapes":
        # U shape (needs a backward pass), comb, spiral (many passes), diagonal chain (8-conn)
        lab[10:50, 5:9] = 1; lab[46:50, 5:30] = 1; lab[10:50, 26:30] = 1
        for k in range(6):
            lab[5:25, 35 + 4 * k:37 + 4 * k] = 2
        lab[22:25, 35:59] = 2
        sp = np.zeros_like(lab); x0, y0, x1, y1 = 40, 28, 78, 58
        while x0 < x1 and y0 < y1:
            sp[y0, x0:x1 + 1] = 1; sp[y0:y1 + 1, x1] = 1; sp[y1, x0:x1 + 1] = 1; sp[y0 + 2:y1 + 1, x0] = 1
            sp[y0 + 2, x0:x0 + 3] = 1
            x0 += 3; y0 += 3; x1 -= 3; y1 -= 3
        lab[sp == 1] = 4
        for k in range(12):
            lab[2 + k, 60 + k] = 5
            lab[2 + k, 61 + k] = 5
        lab[52:58, 10:20] = 3; lab[52:58, 21:30] = 3   # two tv blocks separated by one column
        lab[54, 20] = 1                                 # bridged by another class: stays separate
    elif kind == "touching":
        ellipse(lab, 25, 30, 18, 22, 1); ellipse(lab, 50, 30, 14, 20, 1)   # two people merge
        ellipse(lab, 65, 45, 10, 10, 2); ellipse(lab, 60, 12, 6, 8, 5)
        lab[0:3, :] = 3; lab[:, 77:80] = 4
    elif kind == "overflow":
        for yy in range(0, 60, 6):                     # 10 x 13 = 130 separate 5x5 blocks (25 px)
            for xx in range(0, 78, 6):
                lab[yy:yy + 5, xx:xx + 5] = 1 + (xx // 6 + yy // 6) % 5
    return lab.ravel()


def write_case(i, labels, dist, status, zt):
    objs, kmap, flags, aux = fm.postprocess(labels.reshape(fm.LAB_H, fm.LAB_W), dist, status, zt)
    n_comp = int(aux["kmap_all"].max())
    with open(os.path.join(MEM, f"tb_pp{i}_lab.mem"), "w") as f:
        f.writelines("%x\n" % v for v in labels)
    with open(os.path.join(MEM, f"tb_pp{i}_tof.mem"), "w") as f:
        f.writelines("%04x\n" % v for v in dist)
        f.writelines("%04x\n" % v for v in status)
    tab = {o.k: o for o in objs}
    with open(os.path.join(MEM, f"tb_pp{i}_exp.mem"), "w") as f:
        f.write("%04x\n%04x\n%04x\n" % (len(objs), n_comp, flags["overflow"]))
        for k in range(1, 64):
            o = tab.get(k)
            if o is None:
                f.write("0\n0\n0\n0\n0\n0\n0\n0\n0\n")
            else:
                f.write("1\n%x\n%x\n%x\n%x\n%x\n%x\n%x\n%x\n" % (o.cls, o.area, o.x0, o.y0, o.x1, o.y1, o.dist, o.nsel))
        f.writelines("%x\n" % v for v in kmap)
    return objs, flags, aux


def main():
    zt = fm.load_zone_tables(os.path.join(MEM, "zone_tables.npz"))
    rng = np.random.default_rng(7)
    cases = []
    lab0 = np.array([int(t, 16) for t in open(os.path.join(MEM, "tb_labels.mem")).read().split()])
    cases.append(("golden frame", lab0))
    for kind in ("blobs", "shapes", "touching", "blobs", "overflow"):
        cases.append((kind, synth_labels(rng, kind)))
    for i, (name, lab) in enumerate(cases):
        dist, status = synth_tof(lab, zt, rng)
        objs, flags, aux = write_case(i, lab, dist, status, zt)
        dd = [o.dist for o in objs]
        print(f"case {i} {name:13s}: {len(objs):2d} objects (>= 24 px), {aux['n_big']:3d} big components, "
              f"overflow={flags['overflow']}, distances {dd[:6]}{'...' if len(dd) > 6 else ''}")
    with open(os.path.join(MEM, "tb_pp_ncases.mem"), "w") as f:
        f.write("%x\n" % len(cases))


if __name__ == "__main__":
    main()
