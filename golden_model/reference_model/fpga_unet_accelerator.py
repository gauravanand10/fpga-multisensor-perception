"""
fpga_unet_accelerator.py - stage 4: the int8 Tiny U-Net as the FPGA accelerator computes it
(rtl/unet_core.sv + rtl/unet_engine.sv)

Network (206 K parameters, 164 M multiply-adds per frame), input 1 x 120 x 160, output 6 x 60 x 80:
    stem   3x3 s2  1->16, 3x3 16->16                      60x80   (skip 1)
    down1  3x3 s2 16->32, 3x3 32->32                      30x40   (skip 2)
    down2  3x3 s2 32->80, 3x3 80->80, 3x3 80->80          15x20
    up2    nearest x2, concat skip 2 (112), 3x3 ->32, 3x3 32->32   30x40
    up1    nearest x2, concat skip 1 (48),  3x3 ->16, 3x3 16->16   60x80
    head   1x1 16->6, argmax -> 80 x 60 label map (background, person, chair, tv, laptop, bottle)

Arithmetic (model/quantization/fpga_ref/INT8_SPEC.md): input pixel -> int8 via a 256-entry LUT; per-channel
int8 weights, int32 bias and accumulator; requantisation q = clamp((acc*M + 2^(sh-1)) >> sh, -127, 127)
then ReLU; concat inputs are rescaled and upsampled while they are read, never stored.

Hardware: 16 output x 8 input channels = 128 MAC per clock, 20-stage pipeline, 16 activation banks
of 12 000 bytes, weights in a 2048 x 1024-bit ROM. 1 327 877 clock cycles per frame (measured in the
RTL simulation; data independent) = 13.28 ms at 100 MHz.
"""
import numpy as np

from . import fpga_model as fm
from . import paths

CYCLES_PER_FRAME = 1_327_877         # RTL simulation, tb_unet_core / tb_nexys4_top
CLOCK_HZ = 100_000_000
MAC_LANES = fm.PO * fm.PI


class UNetAccelerator:
    def __init__(self, param_dir=paths.INT8_PARAMS, int8_npz=paths.INT8_NPZ):
        z = np.load(paths.need(int8_npz))
        self.input_scale = float(z["gray__s"])
        self.packed = fm.pack_network(paths.need(param_dir), z["gray__s"])
        self.layers = self.packed.layers

    def run(self, frame_u8):
        """frame_u8 (120,160) uint8 -> (label map (60,80) int, per-layer int8 outputs dict)."""
        outs, labels = fm.run_unet(np.asarray(frame_u8, np.uint8), self.packed)
        return labels, outs

    def macs_per_frame(self):
        return int(sum(l.cout * l.cin * (9 if l.k3 else 1) * l.hout * l.wout for l in self.layers))

    def n_parameters(self):
        return int(sum(W.size + b.size for (W, b, M, sh) in self.packed.raw.values()))

    @staticmethod
    def latency_s():
        return CYCLES_PER_FRAME / CLOCK_HZ
