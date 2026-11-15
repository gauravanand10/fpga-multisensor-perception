"""
record_frames.py - save camera frames of YOUR room for training (needs the ESP32-CAM on COM7).

    python record_frames.py                 300 frames, one every 0.7 s (about 3.5 minutes)
    python record_frames.py --n 600 --every 0.5

Walk around, sit, stand, move chairs, hold a bottle / cup / phone, change the lighting.
Varied frames help far more than many similar ones. Frames go to data\\own_frames\\.
Then run:  python label_own_frames.py
(Close any other program on COM7 and the Serial Monitor first - COM7 must be free.)
"""
import argparse, os, time
import cv2

from config import DATA, ensure_fused_on_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--every", type=float, default=0.7, help="seconds between saved frames")
    ap.add_argument("--port", default="COM7"); ap.add_argument("--baud", type=int, default=921_600)
    a = ap.parse_args()
    ensure_fused_on_path()
    from calibrate import grab_camera            # model\quantization\fpga_ref\calibrate.py
    out = os.path.join(DATA, "own_frames"); os.makedirs(out, exist_ok=True)
    frames = grab_camera(a.n, a.port, a.baud, every=a.every)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(out, f"{stamp}_{i:04d}.png"), f)
    print(f"saved {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
