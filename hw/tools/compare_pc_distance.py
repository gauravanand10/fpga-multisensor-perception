"""
compare_pc_distance.py - FPGA integer distance vs the PC float method of the original PC pipeline

For the post-processing test cases (mem/tb_pp*_lab.mem / _tof.mem) it computes each object's
distance twice:
  PC   : masks upsampled to 320x240, zones painted as exact polygons (DepthMapper.sparse logic,
         using the geo/ Projector at 320x240), float coverage >= 0.3 / best > 0.05, float
         coverage-weighted 35th percentile  (no temporal smoothing - that stays on the PC)
  FPGA : tools/fpga_model.py (80x60 rectangles, distance bins, integer weights)
and prints the difference.

    python tools\\compare_pc_distance.py
"""
import json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fpga_model as fm

PC = os.path.join(HERE, "tof_geometry")                      # ToF geometry package


def main():
    pc = sys.argv[1] if len(sys.argv) > 1 else PC
    for p in (os.path.join(pc, "geo"), pc):
        sys.path.insert(0, p)
    from calibration import CalibrationBundle, Extrinsic, rot_z
    from quick_calib import nominal_intrinsics
    from tof_zones import Orientation, make_zone_model
    from projection import Projector

    orient = Orientation()
    of = os.path.join(pc, "orientation_auto.json")
    if os.path.exists(of):
        orient = Orientation.from_dict(json.load(open(of)))
    W, H = 320, 240
    bundle = CalibrationBundle(nominal_intrinsics(W, H, 66.0), Extrinsic(rot_z(180.0), np.array([0, 50.0, 0]), "n"),
                               orient, make_zone_model("st"), "radial", "auto")
    pr = Projector(bundle, (W, H))
    zt = fm.load_zone_tables(os.path.join(ROOT, "mem", "zone_tables.npz"))
    nc = int(open(os.path.join(ROOT, "mem", "tb_pp_ncases.mem")).read().split()[0], 16)
    diffs = []
    for ci in range(nc):
        lab = np.array([int(t, 16) for t in open(os.path.join(ROOT, "mem", f"tb_pp{ci}_lab.mem")).read().split()])
        tv = np.array([int(t, 16) for t in open(os.path.join(ROOT, "mem", f"tb_pp{ci}_tof.mem")).read().split()])
        dist, status = tv[:64], tv[64:]
        objs, kmap, flags, aux = fm.postprocess(lab.reshape(60, 80), dist, status, zt)
        # ---- PC method ----
        ok_raw = fm.tof_valid(dist, status)
        d_geom = orient.apply(np.where(ok_raw, dist, 0).astype(float).reshape(8, 8))
        v_geom = orient.apply(ok_raw.reshape(8, 8))
        zp = pr.project(d_geom, v_geom)
        zid = np.full((H, W), -1, np.int16)
        idx = np.argwhere(zp.valid)
        for r, c in idx[np.argsort(-zp.depth_cam[zp.valid])]:
            pts = np.round(zp.quads_px[r, c] * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(zid, [pts], int(r * 8 + c), cv2.LINE_8, 4)
        zone_area = np.bincount(zid[zid >= 0].ravel(), minlength=64)
        km = kmap.reshape(60, 80)
        for o in objs:
            m = cv2.resize((km == o.k).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            ids = zid[m]; ids = ids[ids >= 0]
            d_pc = None
            if ids.size:
                cnt = np.bincount(ids, minlength=64)
                frac = cnt / np.maximum(zone_area, 1)
                sel = np.nonzero(frac >= 0.3)[0]
                if sel.size == 0:
                    sel = np.array([int(np.argmax(frac))]) if frac.max() > 0.05 else np.array([], int)
                if sel.size:
                    zr, zc = np.divmod(sel, 8)
                    vals, w = zp.depth_cam[zr, zc], frac[sel]
                    oo = np.argsort(vals); cw = np.cumsum(w[oo])
                    d_pc = float(vals[oo][np.searchsorted(cw, 0.35 * cw[-1])])
            fp = o.dist if o.dist else None
            tag = ""
            if d_pc is None and fp is None:
                tag = "both: no ToF"
            elif d_pc is None or fp is None:
                tag = "only one has a distance"
            else:
                diffs.append(abs(fp - d_pc) / d_pc)
                tag = f"diff {fp - d_pc:+6.0f} mm ({100 * (fp - d_pc) / d_pc:+5.1f} %)"
            if ci < nc - 1 or o.k <= 5:
                print(f"case {ci} obj {o.k:2d} {fm.CLASSES[o.cls]:8s} area {o.area:4d}  "
                      f"PC {('%6.0f' % d_pc) if d_pc else '   ---'}  FPGA {('%6d' % fp) if fp else '   ---'}  {tag}")
    d = np.array(diffs)
    print(f"\n{d.size} objects with a distance on both: median |diff| {100 * np.median(d):.1f} %, "
          f"within 5 %: {100 * (d <= 0.05).mean():.0f} %, max {100 * d.max():.1f} %")


if __name__ == "__main__":
    main()
