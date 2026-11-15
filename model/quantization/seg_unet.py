"""
seg_unet.py - the small U-Net (trained in model/training) as a drop-in for seg_teacher.py.

    seg = SegUNet()                       ONNX Runtime (fast)
    seg = SegUNet(backend="int8")         numpy int8 = exactly what the FPGA computes (slower)
    objs = seg(gray)                      same output as SegTeacher: list of dict(cls, name, score, box, mask)

The network gives one class per pixel (a label map at 80x60). Objects = connected regions of
one class, so two people touching become one object (YOLO separates them, the U-Net cannot).
Used by the original PC pipeline (--seg unet).
"""
import json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "tiny_unet.onnx")
INFO = os.path.join(HERE, "models", "tiny_unet.json")


def preprocess(img, input_hw=(120, 160)):
    """camera frame (grey or BGR, any size) -> float32 1x1xHxW in 0..1. Used by training tools too."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(img, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_AREA)
    return (x.astype(np.float32) / 255.0)[None, None]


class SegUNet:
    def __init__(self, backend="ort", threads=None, min_area=0.005):
        if not os.path.exists(INFO):
            raise FileNotFoundError("models/tiny_unet.json missing - train and export the U-Net first "
                                    "(model/training: train.py, export_onnx.py)")
        info = json.load(open(INFO))
        self.classes, self.input_hw = info["classes"], tuple(info["input_hw"])
        self.min_area = min_area
        if backend == "ort":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 4) // 2)
            self.sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
        else:
            sys.path.insert(0, os.path.join(HERE, "fpga_ref"))
            from engine import Session
            self.sess = Session(os.path.join(HERE, "models", "fpga", "tiny_unet"), backend)
        self.inp = self.sess.get_inputs()[0].name

    def logits(self, img):
        return self.sess.run(None, {self.inp: preprocess(img, self.input_hw)})[0][0]

    def __call__(self, img):
        H, W = img.shape[:2]
        z = self.logits(img)                                   # C x h x w
        e = np.exp(z - z.max(0, keepdims=True)); p = e / e.sum(0, keepdims=True)
        lab = p.argmax(0).astype(np.uint8)
        out = []
        for c in range(1, len(self.classes)):
            m = (lab == c).astype(np.uint8)
            if not m.any():
                continue
            n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < self.min_area * m.size:
                    continue
                small = comp == k
                mask = cv2.resize(small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                x, y, w, h = stats[k, :4]
                sx, sy = W / lab.shape[1], H / lab.shape[0]
                out.append(dict(cls=c, name=self.classes[c], score=float(p[c][small].mean()),
                                box=(x * sx, y * sy, (x + w) * sx, (y + h) * sy), mask=mask))
        out.sort(key=lambda d: -d["mask"].sum())
        return out
