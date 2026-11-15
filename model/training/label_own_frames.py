"""
label_own_frames.py - label your own ESP32-CAM frames automatically with YOLO (the "teacher").

    python label_own_frames.py
    python label_own_frames.py --images "D:\\somewhere\\*.png"

Frames are taken from
    data\\own_frames\\*.png                       (record_frames.py)
    ..\\model\\quantization\\fpga_ref\\calib_frames\\*.png   (the int8 calibration frames)
    model\\quantization\\recordings\\*\\*.png            (objects_viewer.py, key r)
    --images <glob>                              (anything else, camera-sized grey or colour)
YOLOv8n-seg (seg_teacher.py, the same model the live viewer uses) finds the objects:
  * sure detections (score >= 0.45) of a class in config.CLASSES -> that class,
  * unsure detections (0.25 .. 0.45)                             -> 255 = ignored by training,
  * everything else                                              -> background.
Output: data\\own_images.npy / own_labels.npy, and data\\own_preview.png to check the labels.
Training uses these frames several times per epoch (they are your real camera and room).
"""
import argparse, glob, json, os
import numpy as np
import cv2

from config import CLASSES, DATA, FUSED, IGNORE, STORE_HW, ensure_fused_on_path

SURE = 0.45


def frame_list(extra):
    pats = [os.path.join(DATA, "own_frames", "*.png"),
            os.path.join(FUSED, "fpga_ref", "calib_frames", "*.png"),
            os.path.join(FUSED, "recordings", "*", "*.png")]
    if extra:
        pats.append(extra)
    fs = []
    for p in pats:
        fs += sorted(glob.glob(p))
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None)
    a = ap.parse_args()
    ensure_fused_on_path()
    from seg_teacher import SegTeacher
    fs = frame_list(a.images)
    if not fs:
        print("no own frames found - record some with:  python record_frames.py   (training will use COCO only)")
        return
    teacher = SegTeacher(conf=0.25)
    H, W = STORE_HW
    X = np.zeros((len(fs), H, W), np.uint8); Y = np.zeros((len(fs), H, W), np.uint8)
    n_obj = np.zeros(len(CLASSES), int); k = 0
    for i, f in enumerate(fs):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if im.shape != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        lab = np.zeros((H, W), np.uint8)
        for o in sorted(teacher(im), key=lambda o: -o["mask"].sum()):      # big first, small on top
            if o["name"] in CLASSES and o["score"] >= SURE:
                c = CLASSES.index(o["name"]); lab[o["mask"]] = c; n_obj[c] += 1
            elif o["score"] < SURE:
                lab[o["mask"] & (lab == 0)] = IGNORE
        X[k], Y[k] = im, lab; k += 1
        print(f"\r[label] {i + 1}/{len(fs)}", end="")
    print()
    X, Y = X[:k], Y[:k]
    np.save(os.path.join(DATA, "own_images.npy"), X); np.save(os.path.join(DATA, "own_labels.npy"), Y)
    with open(os.path.join(DATA, "own_info.json"), "w") as f:
        cnt = np.bincount(Y[Y != IGNORE].ravel(), minlength=len(CLASSES))[:len(CLASSES)]
        json.dump(dict(classes=CLASSES, n=k, objects={c: int(n) for c, n in zip(CLASSES, n_obj) if n},
                       pixel_share=(cnt / max(cnt.sum(), 1) * 100).tolist()), f, indent=1)
    print(f"[label] {k} frames labelled; objects found: " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, n_obj) if n))
    # preview: up to 12 frames, objects tinted by class
    pal = np.array([[0, 0, 0], [60, 60, 255], [60, 200, 60], [255, 160, 0], [0, 200, 255], [200, 60, 200],
                    [255, 255, 60], [120, 120, 255], [60, 255, 160]], np.uint8)
    tiles = []
    for j in np.linspace(0, k - 1, min(12, k)).astype(int):
        v = cv2.cvtColor(X[j], cv2.COLOR_GRAY2BGR)
        m = (Y[j] > 0) & (Y[j] != IGNORE)
        v[m] = (0.5 * v[m] + 0.5 * pal[Y[j][m] % len(pal)]).astype(np.uint8)
        v[Y[j] == IGNORE] //= 3
        tiles.append(v)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(DATA, "own_preview.png"), grid)
    print("   check the labels:", os.path.join(DATA, "own_preview.png"))


if __name__ == "__main__":
    main()
