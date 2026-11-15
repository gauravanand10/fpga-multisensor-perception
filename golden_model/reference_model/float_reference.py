"""
float_reference.py - the original PC system's network: float32 Tiny U-Net in ONNX Runtime
(model/quantization/seg_unet.py). Same 160x120 input (INTER_AREA resize of the 320x240 frame, /255).
Used to show how closely the int8 FPGA arithmetic follows the float model.
"""
import numpy as np

from . import paths


class FloatUNet:
    def __init__(self, onnx_path=paths.FLOAT_ONNX):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(paths.need(onnx_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def logits(self, frame_u8):
        x = (np.asarray(frame_u8, np.float32) / np.float32(255.0))[None, None]
        return self.sess.run(None, {self.inp: x})[0][0]

    def labels(self, frame_u8):
        return self.logits(frame_u8).argmax(0).astype(np.int64)
