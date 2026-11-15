"""
dataset.py - training / validation data with ESP32-CAM-like augmentation.

Every sample: a 320x240 grey frame + label image from data\\*.npy, randomly cropped / scaled /
flipped / rotated, given random exposure, gamma, blur, noise and JPEG artefacts (like the
ESP32-CAM in a bright room), then resized to the network input (160x120) and the label to
the network output (80x60).
"""
import json, os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset

from config import CLASSES, DATA, IGNORE, INPUT_HW, OUTPUT_HW


def available():
    """which prepared sets exist: dict name -> n."""
    out = {}
    for name in ("coco_train", "coco_val", "own"):
        p = os.path.join(DATA, f"{name}_info.json")
        if os.path.exists(p) and os.path.exists(os.path.join(DATA, f"{name}_images.npy")):
            info = json.load(open(p))
            if info.get("classes") != CLASSES:
                raise SystemExit(f"data/{name} was prepared for classes {info.get('classes')}, config.py now says "
                                 f"{CLASSES}. Re-run prepare_coco.py / label_own_frames.py.")
            out[name] = int(info["n"])
    return out


class Frames(Dataset):
    def __init__(self, name, idx=None, train=True, repeat=1):
        self.name, self.train, self.repeat = name, train, repeat
        n = int(json.load(open(os.path.join(DATA, f"{name}_info.json")))["n"])
        self.idx = np.arange(n) if idx is None else np.asarray(idx)
        self.X = self.Y = None                     # opened lazily (once per worker process)

    def __len__(self):
        return len(self.idx) * self.repeat

    def __getstate__(self):                       # Windows workers get a fresh copy: never send open memmaps
        d = dict(self.__dict__); d["X"] = d["Y"] = None
        return d

    def _open(self):
        self.X = np.load(os.path.join(DATA, f"{self.name}_images.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(DATA, f"{self.name}_labels.npy"), mmap_mode="r")

    def __getitem__(self, i):
        if self.X is None:
            self._open()
        j = self.idx[i % len(self.idx)]
        img, lab = np.array(self.X[j]), np.array(self.Y[j])
        if self.train:
            img, lab = augment(img, lab, np.random)
        x = cv2.resize(img, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
        y = cv2.resize(lab, (OUTPUT_HW[1], OUTPUT_HW[0]), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(x.astype(np.float32) / 255.0)[None], torch.from_numpy(y.astype(np.int64))


def augment(img, lab, rng):
    H, W = img.shape
    # random crop (scale 55-100 % of the frame, aspect 4:3 +-10 %) -> back to 320x240
    s = rng.uniform(0.55, 1.0); ar = rng.uniform(0.9, 1.1)
    ch, cw = int(H * s), min(W, int(H * s * W / H * ar))
    y0, x0 = rng.randint(0, H - ch + 1), rng.randint(0, W - cw + 1)
    img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(lab[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_NEAREST)
    if rng.rand() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.rand() < 0.3:                                          # camera tilt
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-8, 8), 1.0)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        lab = cv2.warpAffine(lab, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE)
    f = img.astype(np.float32) / 255.0
    f = f ** rng.uniform(0.6, 1.6)                                # gamma
    f = (f - 0.5) * rng.uniform(0.6, 1.3) + 0.5 + rng.uniform(-0.12, 0.12)
    if rng.rand() < 0.2:                                          # over-exposed (bright window / lamp)
        f = f * rng.uniform(1.2, 1.8)
    img = np.clip(f * 255, 0, 255).astype(np.uint8)
    if rng.rand() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3) if rng.rand() < 0.7 else (5, 5), 0)
    if rng.rand() < 0.5:
        n = rng.normal(0, rng.uniform(1, 8), img.shape)
        img = np.clip(img + n, 0, 255).astype(np.uint8)
    if rng.rand() < 0.5:                                          # ESP32-CAM JPEG artefacts
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(25, 85))])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(img), np.ascontiguousarray(lab)


def build(own_share=0.2):
    """train set (COCO + own frames, own repeated to about own_share of an epoch) and val sets."""
    have = available()
    if "coco_train" not in have and "own" not in have:
        raise SystemExit("no training data - run prepare_coco.py (and label_own_frames.py) first")
    parts, val = [], {}
    if "coco_train" in have:
        parts.append(Frames("coco_train"))
    if "coco_val" in have:
        val["coco"] = Frames("coco_val", train=False)
    if "own" in have:
        n = have["own"]; idx = np.arange(n)
        v, t = idx[idx % 10 == 0], idx[idx % 10 != 0]          # every 10th own frame = validation
        if len(t):
            base = len(parts[0]) if parts else len(t)
            rep = int(np.clip(round(own_share * base / max(1, (1 - own_share) * len(t))), 1, 50)) if parts else 1
            parts.append(Frames("own", t, repeat=rep))
        if len(v):
            val["own"] = Frames("own", v, train=False)
    return ConcatDataset(parts), val


def class_weights():
    """loss weights ~ 1/sqrt(pixel share): rare classes (cup, phone) count more, background less."""
    shares = []
    for name in ("coco_train", "own"):
        p = os.path.join(DATA, f"{name}_info.json")
        if os.path.exists(p) and "pixel_share" in json.load(open(p)):
            shares.append(np.array(json.load(open(p))["pixel_share"]))
    if not shares:
        return torch.ones(len(CLASSES))
    s = np.maximum(np.mean(shares, 0), 0.05)
    w = 1 / np.sqrt(s)
    w = np.clip(w / w[0], 1.0, 10.0)
    return torch.tensor(w, dtype=torch.float32)
