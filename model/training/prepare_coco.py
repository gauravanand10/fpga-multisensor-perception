"""
prepare_coco.py - download a COCO 2017 subset and turn it into ESP32-CAM-like training data.

    python prepare_coco.py                       (15000 train + 1000 val images, see config.py)
    python prepare_coco.py --train 3000 --val 300      quick try-out

What it does
  1. downloads the COCO 2017 annotation file (about 240 MB, once) into data\\coco\\;
  2. picks images that contain the classes in config.CLASSES, BALANCED: the rare classes
     (tv, laptop, bottle ...) get their share first, person fills the rest; plus 5 % images
     without any of them (so the network learns "nothing here" too);
  3. downloads only those images (about 160 KB each), 12 at a time, backing off when the server is busy; already downloaded
     images are skipped, so the script can be stopped and started again;
  4. makes each image look like the ESP32-CAM: centre crop to 4:3, greyscale, 320x240;
     the object masks become a label image (pixel = class index, 255 = ignore);
  5. writes data\\coco_train_images.npy / _labels.npy (and _val_) - fast to read while training.
"""
import argparse, json, os, random, sys, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2

from config import CLASSES, DATA, IGNORE, STORE_HW, COCO_TRAIN_IMAGES, COCO_VAL_IMAGES, MIN_OBJECT_AREA

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
THREADS = 12                      # parallel image downloads (lower with --threads if you see many 503 errors)
COCO_DIR = os.path.join(DATA, "coco")


UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) coco-subset-downloader"}


def download(url, path, show=True, tries=8):
    """download with resume-safe temp file; backs off when the COCO server says 503 'Slow Down'."""
    tmp = path + ".part"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0)); got = 0; t0 = time.time()
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); got += len(b)
                    if show and total:
                        print(f"\r  {got/1e6:.0f}/{total/1e6:.0f} MB  ({got/1e6/max(time.time()-t0, 1e-3):.1f} MB/s)", end="")
            os.replace(tmp, path)
            if show:
                print()
            return True
        except Exception as e:
            wait = min(60, 3 * 2 ** attempt) + random.uniform(0, 2)       # 3, 6, 12, 24, 48, 60 ... s
            ra = getattr(e, "headers", None) and e.headers.get("Retry-After")
            if ra and str(ra).isdigit():
                wait = max(wait, int(ra))
            if show:
                print(f"\n  download failed ({e}), waiting {wait:.0f} s, retry {attempt + 1}/{tries - 1}")
            if attempt < tries - 1:
                time.sleep(wait)
    return False


def get_annotations():
    os.makedirs(COCO_DIR, exist_ok=True)
    need = [os.path.join(COCO_DIR, f"instances_{s}2017.json") for s in ("train", "val")]
    if all(os.path.exists(p) for p in need):
        return need
    z = os.path.join(COCO_DIR, "annotations_trainval2017.zip")
    if not os.path.exists(z):
        print("[coco] downloading annotations (about 240 MB, once) ...")
        if not download(ANN_URL, z):
            raise SystemExit("could not download the COCO annotations. Download this file with a browser:\n  "
                             f"{ANN_URL}\nand save it as {z}  - then run this script again.")
    print("[coco] extracting ...")
    with zipfile.ZipFile(z) as zf:
        for s in ("train", "val"):
            name = f"annotations/instances_{s}2017.json"
            with zf.open(name) as src, open(os.path.join(COCO_DIR, f"instances_{s}2017.json"), "wb") as dst:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    dst.write(b)
    return need


def crop43(h, w):
    """centre crop box (y0, y1, x0, x1) with a 4:3 (landscape) aspect."""
    th, tw = STORE_HW
    if w * th > h * tw:                       # too wide
        cw = int(round(h * tw / th)); x0 = (w - cw) // 2
        return 0, h, x0, x0 + cw
    ch = int(round(w * th / tw)); y0 = (h - ch) // 2
    return y0, y0 + ch, 0, w


def make_label(coco, img, anns, cat2cls):
    h, w = img["height"], img["width"]
    lab = np.zeros((h, w), np.uint8)
    # big objects first, small ones drawn on top (a cup on a table stays a cup)
    for a in sorted(anns, key=lambda a: -a["area"]):
        if a["category_id"] not in cat2cls:
            continue
        m = coco.annToMask(a).astype(bool)
        if a.get("iscrowd", 0):
            lab[m & (lab == 0)] = IGNORE
        else:
            lab[m] = cat2cls[a["category_id"]]
    return lab


def select(coco, cat2cls, n, rng):
    """balanced pick: rare classes (tv, laptop, bottle ...) get their share first, person fills the rest.
    A plain random pick would be ~90 % person photos and leave too few examples of the rest."""
    per = {c: [] for c in sorted(set(cat2cls.values()))}
    neg = []
    for img_id, img in coco.imgs.items():
        anns = coco.imgToAnns.get(img_id, [])
        area = img["height"] * img["width"]
        good = {cat2cls[a["category_id"]] for a in anns if a["category_id"] in cat2cls and not a.get("iscrowd", 0)
                and a["area"] / area >= MIN_OBJECT_AREA}
        for c in good:
            per[c].append(img_id)
        if not good and not any(a["category_id"] in cat2cls for a in anns):
            neg.append(img_id)
    for v in per.values():
        rng.shuffle(v)
    rng.shuffle(neg)
    budget = n - int(round(0.05 * n))
    chosen, order = [], sorted(per, key=lambda c: len(per[c]))          # rarest class first
    taken = set()
    for k, c in enumerate(order):
        quota = (budget - len(chosen)) // (len(order) - k)
        new = [i for i in per[c] if i not in taken][:quota]
        chosen += new; taken.update(new)
    chosen += neg[:n - len(chosen)]
    rng.shuffle(chosen)
    sets = {c: set(v) for c, v in per.items()}
    counts = {CLASSES[c]: sum(1 for i in chosen if i in sets[c]) for c in per}
    print("   images per class: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + "  (an image can contain several classes)")
    return chosen, len(set().union(*sets.values()))


def build_split(split, n, seed=0):
    from pycocotools.coco import COCO
    js = os.path.join(COCO_DIR, f"instances_{split}2017.json")
    print(f"[{split}] loading annotations ...")
    coco = COCO(js)
    name2cat = {c["name"].replace(" ", "_"): c["id"] for c in coco.loadCats(coco.getCatIds())}
    missing = [c for c in CLASSES[1:] if c not in name2cat]
    if missing:
        raise SystemExit(f"not COCO class names: {missing}. Valid: {sorted(name2cat)}")
    cat2cls = {name2cat[c]: i for i, c in enumerate(CLASSES) if i}
    ids, n_pos = select(coco, cat2cls, n, random.Random(seed))
    print(f"[{split}] {n_pos} images contain the classes; using {len(ids)}")

    img_dir = os.path.join(COCO_DIR, "images", split); os.makedirs(img_dir, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(img_dir, f"{i}.jpg"))]
    if todo:
        print(f"[{split}] downloading {len(todo)} images ({THREADS} at a time) ...")
        t0, done, failed = time.time(), 0, 0
        with ThreadPoolExecutor(THREADS) as ex:
            futs = {ex.submit(download, coco.imgs[i]["coco_url"], os.path.join(img_dir, f"{i}.jpg"), False): i for i in todo}
            for f in as_completed(futs):
                done += 1; failed += not f.result()
                if done % 50 == 0 or done == len(todo):
                    rate = done / max(time.time() - t0, 1e-3)
                    print(f"\r  {done}/{len(todo)}  {rate:.0f} img/s  about {(len(todo) - done) / max(rate, 1e-3) / 60:.0f} min left"
                          f"{f'  ({failed} failed)' if failed else ''}   ", end="")
        print()

    H, W = STORE_HW
    ok = [i for i in ids if os.path.exists(os.path.join(img_dir, f"{i}.jpg"))]
    X = np.lib.format.open_memmap(os.path.join(DATA, f"coco_{split}_images.npy"), "w+", np.uint8, (len(ok), H, W))
    Y = np.lib.format.open_memmap(os.path.join(DATA, f"coco_{split}_labels.npy"), "w+", np.uint8, (len(ok), H, W))
    counts = np.zeros(len(CLASSES), np.int64)
    k = 0
    for j, i in enumerate(ok):
        im = cv2.imread(os.path.join(img_dir, f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        img = coco.imgs[i]
        lab = make_label(coco, img, coco.imgToAnns.get(i, []), cat2cls)
        if lab.shape != im.shape:                                    # rare: annotation size differs
            lab = cv2.resize(lab, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        y0, y1, x0, x1 = crop43(*im.shape)
        X[k] = cv2.resize(im[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_AREA)
        Y[k] = cv2.resize(lab[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_NEAREST)
        counts += np.bincount(Y[k][Y[k] != IGNORE].ravel(), minlength=len(CLASSES))[:len(CLASSES)]
        k += 1
        if j % 500 == 0:
            print(f"\r[{split}] converting {j + 1}/{len(ok)}", end="")
    print(f"\r[{split}] converted {k} images -> data\\coco_{split}_*.npy          ")
    X.flush(); Y.flush()
    del X, Y                       # unreadable files leave unused rows at the end; info.json "n" says how many are valid
    share = counts / max(counts.sum(), 1) * 100
    print("   pixels per class: " + ", ".join(f"{c} {s:.1f}%" for c, s in zip(CLASSES, share)))
    with open(os.path.join(DATA, f"coco_{split}_info.json"), "w") as f:
        json.dump(dict(classes=CLASSES, n=k, pixel_share=share.tolist()), f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=COCO_TRAIN_IMAGES)
    ap.add_argument("--val", type=int, default=COCO_VAL_IMAGES)
    ap.add_argument("--threads", type=int, default=THREADS, help="parallel downloads (lower if you see many 503 errors)")
    a = ap.parse_args()
    globals()["THREADS"] = max(1, a.threads)
    os.makedirs(DATA, exist_ok=True)
    get_annotations()
    build_split("val", a.val)
    build_split("train", a.train)
    print("done. Next: python label_own_frames.py   then   python train.py")


if __name__ == "__main__":
    main()
