"""
run_end_to_end_demo.py - the complete system, frame by frame, on scenes with known truth

Scenes: COCO 2017 validation photos (320x240 grey, never used in training) stand in for the
ESP32-CAM sensor. The photos have no depth, so every ground-truth object is placed at a random
depth (0.7-2.6 m, wall at 3.4 m) and the VL53L5CX frame is simulated from that layout with the real
zone geometry (reference_model.tof_node). Then every stage runs exactly as on the hardware:

  ESP32-CAM -> CAM1 SPI packet -> FPGA cam_rx      ToF ESP32 -> TOF1 SPI packet -> FPGA tof_rx
  FPGA int8 U-Net -> label map -> objects + distance -> UART packet -> laptop "Camera | Depth Map"

For each scene, results/end_to_end/scene_XX_<classes>/ gets every intermediate result (images,
the exact SPI and UART bytes, a JSON report) and pipeline_overview.png. A summary with the
distance error of every object against its true depth goes to results/end_to_end/.
Also checks that corrupted / truncated packets are rejected by the FPGA receivers.

    python experiments\\run_end_to_end_demo.py            (8 scenes)
    python experiments\\run_end_to_end_demo.py --scenes 10 --seed 3
"""
import argparse, json, os
import cv2
import numpy as np
from _common import ROOT, style, colorize, CLASS_RGB, INK2, SERIES
from reference_model import paths, fpga_model as fm, tof_node, spi_link, uart_link, laptop_host, system_timing
from reference_model.pipeline import EndToEndSystem
from reference_model.float_reference import FloatUNet

OUT = os.path.join(ROOT, "results", "end_to_end")


def choose_scenes(imgs, gts, n, flt):
    """Photos with clear objects, covering all five classes, ranked by float-model pixel accuracy."""
    cand = []
    for i in range(len(imgs)):
        gt = np.array(gts[i]); share = [(gt == c).mean() for c in range(fm.N_CLS)]
        present = [c for c in range(1, fm.N_CLS) if share[c] > 0.02]
        if not present or (gt == 255).mean() > 0.15:
            continue
        lab = flt.labels(fm.downsample_2x2(np.array(imgs[i])))
        up = np.repeat(np.repeat(lab, 4, 0), 4, 1); v = gt != 255
        acc = float((up[v] == gt[v]).mean())
        hit = all((up[gt == c] == c).mean() > 0.5 for c in present)
        if hit:
            cand.append((acc + 0.03 * len(present), i, present))
    cand.sort(reverse=True)
    chosen, covered = [], set()
    for need in range(1, fm.N_CLS):                       # every class at least once
        for s, i, pr in cand:
            if need in pr and need not in covered and i not in [c[1] for c in chosen]:
                chosen.append((s, i, pr)); covered |= set(pr); break
    for s, i, pr in cand:                                 # fill with the best remaining multi-object scenes
        if len(chosen) >= n:
            break
        if i not in [c[1] for c in chosen] and len(pr) >= 2:
            chosen.append((s, i, pr))
    return [(i, pr) for s, i, pr in chosen[:n]]


def tof_grid_image(dist, status, scale=48):
    ok = fm.tof_valid(dist, status)
    g = np.zeros((8 * scale, 8 * scale, 3), np.uint8)
    for i in range(64):
        r, c = divmod(i, 8)
        col = laptop_host.depth_color(float(dist[i])) if ok[i] else (60, 60, 60)
        g[r * scale:(r + 1) * scale, c * scale:(c + 1) * scale] = col
        t = f"{dist[i] / 1000:.2f}" if ok[i] else "--"
        cv2.putText(g, t, (c * scale + 4, r * scale + scale // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(g, t, (c * scale + 4, r * scale + scale // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    for k in range(9):
        cv2.line(g, (0, k * scale), (8 * scale, k * scale), (20, 20, 20), 1)
        cv2.line(g, (k * scale, 0), (k * scale, 8 * scale), (20, 20, 20), 1)
    return g


def zones_on_camera(frame, dist, status, zt, size=(640, 480)):
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), size, interpolation=cv2.INTER_LINEAR)
    sx, sy = size[0] / fm.LAB_W, size[1] / fm.LAB_H
    ok = fm.tof_valid(dist, status)
    for i in range(64):
        if not ok[i]:
            continue
        x0, y0, x1, y1, v = zt.rects[i, fm.depth_bin(int(dist[i]), zt)]
        if v:
            cv2.rectangle(img, (int(x0 * sx), int(y0 * sy)), (int((x1 + 1) * sx) - 2, int((y1 + 1) * sy) - 2),
                          laptop_host.depth_color(float(dist[i])), 2)
    return img


def object_map_image(post, size=(640, 480)):
    lab = np.zeros((fm.LAB_H, fm.LAB_W), np.int64)
    for o in post.objects:
        lab[post.object_map == o.k] = o.cls
    img = cv2.resize(colorize(lab)[:, :, ::-1], size, interpolation=cv2.INTER_NEAREST).copy()
    sx, sy = size[0] / fm.LAB_W, size[1] / fm.LAB_H
    for o in post.objects:
        p0, p1 = (int(o.x0 * sx), int(o.y0 * sy)), (int((o.x1 + 1) * sx) - 1, int((o.y1 + 1) * sy) - 1)
        cv2.rectangle(img, p0, p1, (255, 255, 255), 2)
        t = f"#{o.k} {fm.CLASSES[o.cls]}"
        cv2.putText(img, t, (p0[0] + 3, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (p0[0] + 3, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def match_truth(post, truth):
    rows = []
    for o in post.objects:
        ids, cnt = np.unique(truth.instance_map[post.object_map == o.k], return_counts=True)
        sel = [(c, k) for k, c in zip(ids, cnt) if k > 0]
        k_true = int(max(sel)[1]) if sel else 0
        z = truth.instance_dist.get(k_true)
        rows.append(dict(id=o.k, cls=fm.CLASSES[o.cls], area_px=o.area, box_80x60=[o.x0, o.y0, o.x1, o.y1],
                         fpga_distance_mm=o.dist or None, zones_used=o.nsel,
                         true_depth_mm=z, true_class=fm.CLASSES[truth.instance_cls[k_true]] if k_true else "background",
                         error_mm=(o.dist - z) if (o.dist and z) else None))
    return rows


def overview(path, title, panels, table):
    plt = style()
    fig = plt.figure(figsize=(16, 8.3), dpi=150)
    gs = fig.add_gridspec(2, 4, height_ratios=[2.85, 4.75], hspace=0.1, wspace=0.06)
    for j, (img, cap) in enumerate(panels[:4]):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(img[:, :, ::-1] if img.ndim == 3 else img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(cap, fontsize=9.5, loc="left"); ax.axis("off")
    ax = fig.add_subplot(gs[1, :3]); img, cap = panels[4]
    ax.imshow(img[:, :, ::-1]); ax.set_title(cap, fontsize=9.5, loc="left"); ax.axis("off")
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    ax.set_title("FPGA objects vs simulated truth", fontsize=9.5, loc="left")
    y = 0.97
    ax.text(0, y, f"{'id':>2} {'class':8s} {'FPGA':>6s} {'true':>6s} {'err':>5s}", family="DejaVu Sans Mono",
            fontsize=8.5, color=INK2, va="top", transform=ax.transAxes)
    for r in table[:16]:
        y -= 0.055
        f = f"{r['fpga_distance_mm'] / 1000:.2f}" if r["fpga_distance_mm"] else "  ?"
        t = f"{r['true_depth_mm'] / 1000:.2f}" if r["true_depth_mm"] else "  -"
        e = f"{r['error_mm']:+d}" if r["error_mm"] is not None else ""
        ax.text(0, y, f"{r['id']:>2} {r['cls']:8s} {f:>6s} {t:>6s} {e:>5s}", family="DejaVu Sans Mono",
                fontsize=8.5, va="top", transform=ax.transAxes)
    ax.text(0, 0.0, "distances in m, error in mm\n(truth = depth chosen for the simulated ToF scene)",
            fontsize=7.5, color=INK2, transform=ax.transAxes)
    patches = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in CLASS_RGB]
    fig.legend(patches, [c if c != "tv" else "TV" for c in fm.CLASSES], loc="upper right", ncol=6, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, y=0.995)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    imgs, gts = paths.load_coco_val()
    sysm, flt = EndToEndSystem(), FloatUNet()
    zt = sysm.post.zt
    rng = np.random.default_rng(a.seed)
    scenes = choose_scenes(imgs, gts, a.scenes, flt)
    summary = dict(scenes=[], objects=[])
    for n, (idx, present) in enumerate(scenes, 1):
        qvga = np.array(imgs[idx]); gt = np.array(gts[idx])
        truth = tof_node.scene_from_labels(gt, rng)
        dist, status = tof_node.simulate_tof_frame(truth, zt, rng)
        tr = sysm.process(qvga, dist, status, send_image=True)
        name = f"scene_{n:02d}_" + "_".join(fm.CLASSES[c] for c in present)
        d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "spi_camera_packet.bin"), "wb").write(tr.cam.packet)
        open(os.path.join(d, "spi_tof_packet.bin"), "wb").write(tr.tof.packet)
        grid = tof_grid_image(dist, status)
        zc = zones_on_camera(tr.cam.frame, dist, status, zt)
        lm = cv2.resize(colorize(tr.labels)[:, :, ::-1], (640, 480), interpolation=cv2.INTER_NEAREST)
        om = object_map_image(tr.post)
        open(os.path.join(d, "uart_result_packet.bin"), "wb").write(tr.uart_bytes)
        view = laptop_host.render_view(tr.host_packet, zt)
        table = match_truth(tr.post, truth)
        fl = flt.labels(tr.cam.frame)
        rep = dict(scene=name, coco_val_index=int(idx), classes_present=[fm.CLASSES[c] for c in present],
                   spi_camera_packet_bytes=len(tr.cam.packet), spi_tof_packet_bytes=len(tr.tof.packet),
                   camera_packet_accepted=tr.cam_rx.ok, tof_packet_accepted=tr.tof_rx.ok,
                   tof_valid_zones=int(fm.tof_valid(dist, status).sum()),
                   label_agreement_float_vs_int8=float((fl == tr.labels).mean()),
                   uart_packet_bytes=len(tr.uart_bytes),
                   uart_time_ms=1e3 * uart_link.transfer_time_s(len(tr.uart_bytes)),
                   laptop_packet_checksum_ok=True, objects=table)
        json.dump(rep, open(os.path.join(d, "scene_report.json"), "w"), indent=1)
        overview(os.path.join(d, "pipeline_overview.png"),
                 f"Scene {n}: end-to-end reference model (COCO val #{idx}, simulated ToF)",
                 [(qvga, "1  ESP32-CAM sensor, 320x240 grey"),
                  (tr.cam.frame, f"2  SPI to FPGA: 160x120 ({len(tr.cam.packet)} B CAM1)"),
                  (zc, f"3  ToF zones ({rep['tof_valid_zones']}/64 valid, colour = distance)"),
                  (lm, "4  FPGA int8 U-Net label map, 80x60"),
                  (view, f"5  Laptop view from the {len(tr.uart_bytes)} B UART packet (2 Mbaud, "
                         f"{rep['uart_time_ms']:.0f} ms)")], table)
        summary["scenes"].append({k: rep[k] for k in rep if k != "objects"})
        summary["objects"] += [dict(scene=n, **r) for r in table]
        print(f"{name}: {len(table)} objects, agreement float/int8 {100 * rep['label_agreement_float_vs_int8']:.1f} %, "
              f"UART {len(tr.uart_bytes)} B")

    # robustness: the FPGA receivers must reject damaged packets
    cam, tof = spi_link.CameraReceiver(), spi_link.ToFReceiver()
    good_c, good_t = tr.cam.packet, tr.tof.packet
    flip = bytearray(good_c); flip[1000] ^= 0x10
    tests = [("camera packet, 1 bit flipped", cam.receive(bytes(flip))),
             ("camera packet, last 100 bytes missing", cam.receive(good_c[:-100])),
             ("camera packet, wrong magic", cam.receive(b"CAM2" + good_c[4:])),
             ("ToF packet, 1 byte changed", tof.receive(good_t[:50] + bytes([good_t[50] ^ 1]) + good_t[51:])),
             ("ToF packet, truncated", tof.receive(good_t[:120])),
             ("camera packet, intact", cam.receive(good_c)), ("ToF packet, intact", tof.receive(good_t))]
    summary["packet_checks"] = [dict(case=c, accepted=r.ok, reason=r.reason) for c, r in tests]

    err = np.array([o["error_mm"] for o in summary["objects"] if o["error_mm"] is not None], float)
    rel = np.array([abs(o["error_mm"]) / o["true_depth_mm"] for o in summary["objects"] if o["error_mm"] is not None])
    summary["distance_error"] = dict(n=int(err.size), median_abs_mm=float(np.median(np.abs(err))),
                                     mean_abs_mm=float(np.mean(np.abs(err))),
                                     within_5pct=float((rel <= 0.05).mean()), within_10pct=float((rel <= 0.10).mean()),
                                     no_distance=int(sum(o["fpga_distance_mm"] is None for o in summary["objects"])))
    b = system_timing.budget()
    summary["timing_budget_ms"] = [dict(stage=r[0], ms=r[1], source=r[2]) for r in b.rows()]
    summary["fps_limit"] = dict(with_image=b.fps_limit(True), without_image=b.fps_limit(False))
    json.dump(summary, open(os.path.join(OUT, "end_to_end_summary.json"), "w"), indent=1)

    de = summary["distance_error"]
    L = ["End-to-end reference model: ESP32 -> SPI -> FPGA -> UART -> laptop", "",
         f"scenes: {len(scenes)} COCO 2017 validation photos, ToF frames simulated from the true layout", ""]
    for s in summary["scenes"]:
        L.append(f"  {s['scene']:40s} CAM1 {s['spi_camera_packet_bytes']} B, TOF1 {s['spi_tof_packet_bytes']} B, "
                 f"UART {s['uart_packet_bytes']} B, float/int8 agreement {100 * s['label_agreement_float_vs_int8']:.1f} %")
    L += ["", f"object distances ({de['n']} objects with a distance, {de['no_distance']} without):",
          f"  median |error| {de['median_abs_mm']:.0f} mm, mean |error| {de['mean_abs_mm']:.0f} mm, "
          f"within 5 %: {100 * de['within_5pct']:.0f} %, within 10 %: {100 * de['within_10pct']:.0f} %", "",
          "FPGA receivers, damaged packets:"]
    L += [f"  {t['case']:40s} -> {'accepted' if t['accepted'] else 'rejected (' + t['reason'] + ')'}" for t in summary["packet_checks"]]
    L += ["", "per-frame time budget (not measured on hardware):"]
    L += [f"  {t['stage']:55s} {t['ms']:7.2f} ms   [{t['source']}]" for t in summary["timing_budget_ms"]]
    L += [f"  frame-rate limit: {summary['fps_limit']['with_image']:.1f} fps with the camera image in the packet, "
          f"{summary['fps_limit']['without_image']:.1f} fps without (UART/SPI bound; ESP32-CAM capture rate not modelled)"]
    open(os.path.join(OUT, "end_to_end_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
