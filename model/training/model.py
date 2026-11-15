"""
model.py - the small U-Net, built ONLY from layers the FPGA conv engine supports
(INT8_SPEC.md): 3x3 / 1x1 convolutions (stride 1 or 2, padding 1), ReLU, nearest x2
upsampling, channel concatenation. BatchNorm is used for training and folded into the
convolutions before export.

    input : 1 x 1 x 120 x 160  grey camera frame / 255        (config.INPUT_HW)
    output: 1 x C x 60 x 80    class scores, argmax = label map (config.OUTPUT_HW)

    stem  3x3 s2  1 ->16, 3x3 16->16               60x80   (skip 1)
    down1 3x3 s2 16->32, 3x3 32->32                30x40   (skip 2)
    down2 3x3 s2 32->64, 3x3 64->64, 3x3 64->64    15x20
    up2   x2, concat skip 2 (96), 3x3 ->32, 3x3 32->32       30x40
    up1   x2, concat skip 1 (48), 3x3 ->16, 3x3 16->16       60x80
    head  1x1 16->C
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Sequential):
    """Conv 3x3 + BatchNorm + ReLU."""
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self, n_classes, widths=(16, 32, 64)):
        super().__init__()
        a, b, c = widths
        self.stem = nn.Sequential(CBR(1, a, 2), CBR(a, a))
        self.down1 = nn.Sequential(CBR(a, b, 2), CBR(b, b))
        self.down2 = nn.Sequential(CBR(b, c, 2), CBR(c, c), CBR(c, c))
        self.up2 = nn.Sequential(CBR(c + b, b), CBR(b, b))
        self.up1 = nn.Sequential(CBR(b + a, a), CBR(a, a))
        self.head = nn.Conv2d(a, n_classes, 1)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        y = self.down2(s2)
        y = self.up2(torch.cat([F.interpolate(y, scale_factor=2.0, mode="nearest"), s2], 1))
        y = self.up1(torch.cat([F.interpolate(y, scale_factor=2.0, mode="nearest"), s1], 1))
        return self.head(y)


def fuse_bn(model):
    """eval-mode copy with every Conv+BN folded into one Conv (what the FPGA runs)."""
    import copy
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, CBR):
            mod[0] = fuse_conv_bn_eval(mod[0], mod[1])
            mod[1] = nn.Identity()
    return m


def cost(model, hw=(120, 160)):
    """parameters and multiply-accumulates per frame (convolutions only)."""
    macs = []

    def hook(mod, inp, out):
        macs.append(out.numel() * mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1])
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, *hw))
    for h in hs:
        h.remove()
    params = sum(p.numel() for n, p in model.named_parameters() if "bn" not in n)
    return params, sum(macs)


if __name__ == "__main__":
    from config import CLASSES, WIDTHS, INPUT_HW
    m = TinyUNet(len(CLASSES), WIDTHS)
    p, mac = cost(m, INPUT_HW)
    print(f"TinyUNet {WIDTHS}: {p/1e3:.0f} K parameters, {mac/1e6:.0f} M MACs per frame, "
          f"output {tuple(m(torch.zeros(1, 1, *INPUT_HW)).shape)}")
