"""
train.py - train the small U-Net on the GPU (RTX 4050), then fine-tune it for int8.

    python train.py                          60 epochs float + 6 epochs int8-aware (QAT)
    python train.py --epochs 20              quicker
    python train.py --resume                 continue an interrupted run
    python train.py --qat-only               only the int8 fine-tuning, from runs\\<name>\\best.pt

Outputs in runs\\<name>\\ (default name "tiny_unet"):
    best.pt        best float model (highest mean IoU on the validation frames)
    best_qat.pt    after int8-aware fine-tuning  <- export_onnx.py uses this one
    log.csv        loss and IoU per epoch
    samples_XX.png validation frames: camera | label | prediction, every 5 epochs
Mean IoU = for each object class, overlap / union of predicted and true pixels, averaged.
"""
import argparse, csv, math, os, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CLASSES, IGNORE, RUNS, WIDTHS, INPUT_HW
from dataset import build, class_weights
from model import TinyUNet, cost
from qat import make_qat, build_qat, enable

PALETTE = np.array([[0, 0, 0], [60, 60, 255], [60, 200, 60], [255, 160, 0], [0, 200, 255], [200, 60, 200],
                    [255, 255, 60], [120, 120, 255], [60, 255, 160], [160, 160, 160]], np.uint8)


def seed_worker(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); import random; random.seed(s)


def loss_fn(logits, y, w):
    ce = F.cross_entropy(logits, y, weight=w, ignore_index=IGNORE)
    # soft Dice over the object classes: keeps small objects (cups, phones) from vanishing
    valid = (y != IGNORE)
    p = logits.softmax(1)[:, 1:] * valid[:, None]
    t = F.one_hot(torch.where(valid, y, 0), logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].float()
    inter = (p * t).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    dice = 1 - (2 * inter + 1) / (den + 1)
    return ce + (dice[present].mean() if present.any() else 0.0)


@torch.no_grad()
def evaluate(model, ds, dev, bs=128):
    model.eval()
    C = len(CLASSES)
    inter = torch.zeros(C, dtype=torch.float64); union = torch.zeros(C, dtype=torch.float64)
    for x, y in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = model(x).argmax(1)
        v = y != IGNORE
        for c in range(C):
            a, b = (p == c) & v, (y == c) & v
            inter[c] += (a & b).sum().item(); union[c] += (a | b).sum().item()
    iou = (inter / union.clamp_min(1)).numpy()
    seen = union.numpy() > 0
    miou = float(iou[1:][seen[1:]].mean()) if seen[1:].any() else float("nan")    # nan: no objects in this set
    return miou, iou


@torch.no_grad()
def samples(model, ds, dev, fn, n=6):
    model.eval()
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    rows = []
    for i in idx:
        x, y = ds[i]
        p = model(x[None].to(dev)).argmax(1)[0].cpu().numpy()
        H, W = INPUT_HW
        cam = cv2.cvtColor((x[0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        def col(l):
            v = PALETTE[np.where(l == IGNORE, 9, l) % len(PALETTE)]
            return cv2.resize(v, (W, H), interpolation=cv2.INTER_NEAREST)
        rows.append(np.hstack([cam, col(y.numpy()), col(p)]))
    cv2.imwrite(fn, np.vstack(rows))


def run_epochs(model, train_ds, val, dev, a, epochs, lr, tag, out, log, best_name):
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                        pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0, worker_init_fn=seed_worker)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(loader); warm = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and tag == "float")
    w = class_weights().to(dev)
    best, start = -1.0, 0
    ck = os.path.join(out, f"last_{tag}.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, best = s["epoch"] + 1, s["best"]
        print(f"[{tag}] resumed at epoch {start}")
    for ep in range(start, epochs):
        model.train(); t0 = time.time(); tot = n = 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda" and tag == "float"):
                loss = loss_fn(model(x), y, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); n += len(x)
        res = {k: evaluate(model, d, dev) for k, d in val.items()}
        vals = [r[0] for r in res.values() if not np.isnan(r[0])]
        score = float(np.mean(vals)) if vals else -tot / n
        dt = time.time() - t0
        msg = " | ".join(f"{k} mIoU {r[0]:.3f}" for k, r in res.items())
        print(f"[{tag}] epoch {ep + 1}/{epochs}  loss {tot / n:.3f}  {msg}  ({dt:.0f} s, "
              f"about {dt * (epochs - ep - 1) / 60:.0f} min left)")
        row = dict(phase=tag, epoch=ep + 1, loss=round(tot / n, 4), seconds=round(dt, 1))
        for k, r in res.items():
            row[f"{k}_miou"] = round(r[0], 4)
            row.update({f"{k}_{c}": round(float(v), 4) for c, v in zip(CLASSES, r[1])})
        log(row)
        if score > best:
            best = score
            torch.save(dict(model=model.state_dict(), classes=CLASSES, widths=WIDTHS, kind=tag, epoch=ep + 1,
                            score=score), os.path.join(out, best_name))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best),
                   ck)
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            for k, d in val.items():
                samples(model, d, dev, os.path.join(out, f"samples_{tag}_{k}_{ep + 1:03d}.png"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tiny_unet")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--qat-epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=6, help="CPU processes preparing images (lower if the PC struggles)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--qat-only", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU (very slow, for testing only)")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if dev.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("[gpu] NO CUDA GPU FOUND - training on the CPU will be very slow. See README (install the CUDA PyTorch).")
    out = os.path.join(RUNS, a.name); os.makedirs(out, exist_ok=True)
    train_ds, val = build()
    print(f"[data] {len(train_ds)} training samples per epoch; validation: "
          + ", ".join(f"{k} {len(d)}" for k, d in val.items()))
    model = TinyUNet(len(CLASSES), WIDTHS).to(dev)
    p, mac = cost(TinyUNet(len(CLASSES), WIDTHS), INPUT_HW)
    print(f"[model] {p/1e3:.0f} K parameters, {mac/1e6:.0f} M MACs per frame, classes {CLASSES}")

    fcsv = open(os.path.join(out, "log.csv"), "a", newline="")
    writer = [None]

    def log(row):
        if writer[0] is None:
            writer[0] = csv.DictWriter(fcsv, fieldnames=list(row)); writer[0].writeheader()
        writer[0].writerow({k: row.get(k, "") for k in writer[0].fieldnames}); fcsv.flush()

    if not a.qat_only:
        best = run_epochs(model, train_ds, val, dev, a, a.epochs, a.lr, "float", out, log, "best.pt")
        print(f"[float] best mean IoU {best:.3f} -> {os.path.join(out, 'best.pt')}")
    s = torch.load(os.path.join(out, "best.pt"), map_location="cpu")
    model = TinyUNet(len(CLASSES), WIDTHS); model.load_state_dict(s["model"])
    if a.qat_epochs > 0:
        q = make_qat(model).to(dev)
        best = run_epochs(q, train_ds, val, dev, a, a.qat_epochs, a.lr * 0.05, "qat", out, log, "best_qat.pt")
        print(f"[qat] best mean IoU with int8 rounding {best:.3f} -> {os.path.join(out, 'best_qat.pt')}")
    print("next:  python export_onnx.py")


if __name__ == "__main__":
    main()
