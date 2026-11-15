"""
gen_zone_tables.py - ToF zone footprints on the 80x60 label grid, for the FPGA.

Uses the SAME geometry as the original PC pipeline (hw\\tools\\tof_geometry\\geo): ST zone directions, nominal
ESP32-CAM lens (66 deg diagonal), ToF 50 mm below the lens, rotation 180 deg about z,
orientation from hw\\tools\\tof_geometry\\orientation_auto.json, radial distances.

For every RAW zone index i and every distance bin b the zone quad is projected at the bin's
distance, rasterised on 80x60 exactly like DepthMapper.sparse (cv2.fillPoly, 4 sub-pixel bits)
and stored as a tight rectangle (median of the polygon edges). The FPGA picks the bin from the measured distance, so
parallax (ToF 50 mm below the camera) is handled per zone, like on the PC.

Outputs (in hw\\mem):
  zone_tables.npz          for tools/fpga_model.py
  zone_rects.mem           64*NB words: {valid, y1[5:0], x1[6:0], y0[5:0], x0[6:0]}  (27 bits, hex)
  zone_rz.mem              64 words: ray_z Q15 per raw zone
  zone_binlo.mem           NB words: lower distance edge of each bin (mm)

    python hw\\tools\\gen_zone_tables.py   (from the repository root)
"""
import argparse, json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # hw
PC = os.path.join(HERE, "tof_geometry")                      # ToF geometry package
LAB_W, LAB_H = 80, 60
NB = 32
D_NEAR, D_FAR = 150.0, 4000.0          # inverse-distance uniform bins between these


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", default=PC, help="ToF geometry package")
    ap.add_argument("--out", default=os.path.join(ROOT, "mem"))
    ap.add_argument("--offset-mm", default="0,50,0")
    ap.add_argument("--lens-diag-fov", type=float, default=66.0)
    a = ap.parse_args()
    for p in (os.path.join(a.pc, "geo"), a.pc):
        sys.path.insert(0, p)
    from calibration import CalibrationBundle, Extrinsic, rot_z
    from quick_calib import nominal_intrinsics
    from tof_zones import Orientation, make_zone_model
    from projection import Projector
    import config as gcfg

    orient = Orientation()
    of = os.path.join(a.pc, "orientation_auto.json")
    if os.path.exists(of):
        orient = Orientation.from_dict(json.load(open(of)))
    offset = np.array([float(v) for v in a.offset_mm.split(",")])
    intr = nominal_intrinsics(320, 240, a.lens_diag_fov)           # the PC viewer works at 320x240
    extr = Extrinsic(rot_z(180.0), offset, "nominal")
    zm = make_zone_model("st")
    bundle = CalibrationBundle(intr, extr, orient, zm, "radial", "auto")
    pr = Projector(bundle, (LAB_W, LAB_H))                          # = intrinsics scaled to 80x60
    perm = orient.perm                                              # perm[r,c] = raw index at geometry (r,c)

    inv = np.linspace(1.0 / D_FAR, 1.0 / D_NEAR, NB + 1)            # inverse-distance edges
    edges = 1.0 / inv[::-1]                                         # ascending distances
    bin_lo = np.round(edges[:-1]).astype(np.int64)
    bin_lo[0] = int(gcfg.TOF_MIN_MM)
    d_rep = 2.0 / (1.0 / edges[:-1] + 1.0 / edges[1:])              # harmonic centre of each bin

    rects = np.zeros((64, NB, 5), np.int64)
    for b in range(NB):
        d = np.full((8, 8), d_rep[b]); v = np.ones((8, 8), bool)
        zp = pr.project(d, v)
        for r in range(8):
            for c in range(8):
                i = int(perm[r, c])
                if not zp.valid[r, c]:
                    continue
                m = np.zeros((LAB_H, LAB_W), np.uint8)
                pts = np.round(zp.quads_px[r, c] * 16).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(m, [pts], 1, cv2.LINE_8, 4)
                ys, xs = np.nonzero(m)
                if xs.size:
                    # tight rectangle: median of the polygon's row / column edges (overlaps the
                    # neighbouring zones less than the bounding box; 95 % of objects within 5 %
                    # of the PC polygon method, see tools/compare_pc_distance.py)
                    rows = [np.nonzero(m[yy])[0] for yy in np.unique(ys)]
                    cols = [np.nonzero(m[:, xx])[0] for xx in np.unique(xs)]
                    x0 = int(round(np.median([r_.min() for r_ in rows])))
                    x1 = int(round(np.median([r_.max() for r_ in rows])))
                    y0 = int(round(np.median([c_.min() for c_ in cols])))
                    y1 = int(round(np.median([c_.max() for c_ in cols])))
                    rects[i, b] = (x0, y0, x1, y1, 1)
    rz = np.zeros(64, np.int64)
    cz = pr.center_vec[..., 2]
    for r in range(8):
        for c in range(8):
            rz[int(perm[r, c])] = int(round(cz[r, c] * (1 << 15)))

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "zone_tables.npz"), rz_q15=rz, bin_lo=bin_lo, rects=rects, d_rep=d_rep,
             orientation=json.dumps(orient.to_dict()), offset=offset, lens=a.lens_diag_fov)
    with open(os.path.join(a.out, "zone_rects.mem"), "w") as f:
        for i in range(64):
            for b in range(NB):
                x0, y0, x1, y1, v = rects[i, b]
                w = (v << 26) | (y1 << 20) | (x1 << 13) | (y0 << 7) | x0
                f.write("%07x\n" % w)
    with open(os.path.join(a.out, "zone_rz.mem"), "w") as f:
        f.writelines("%04x\n" % v for v in rz)
    with open(os.path.join(a.out, "zone_binlo.mem"), "w") as f:
        f.writelines("%04x\n" % v for v in bin_lo)
    cov = np.zeros((LAB_H, LAB_W), bool)
    for i in range(64):
        x0, y0, x1, y1, v = rects[i, 12]
        if v:
            cov[y0:y1 + 1, x0:x1 + 1] = True
    print(f"orientation {orient.describe()}  offset {offset} mm  lens {a.lens_diag_fov} deg")
    print(f"{NB} bins, bin_lo = {bin_lo.tolist()}")
    print(f"valid rects: {int(rects[..., 4].sum())}/{64 * NB};  ToF view covers {cov.mean() * 100:.0f} % "
          f"of the label grid at {d_rep[12]:.0f} mm")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
