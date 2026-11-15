"""
qat.py - quantisation-aware fine-tuning: the network trains while seeing the same int8
rounding the FPGA will do (INT8_SPEC.md), so the int8 version loses almost nothing.

  * BatchNorm is folded into the convolutions first (as on the FPGA);
  * conv weights: symmetric int8, one scale per output channel (rounded in the forward pass,
    gradients pass straight through);
  * activations: symmetric int8 per tensor at the network input, after every ReLU and at the
    output, with the range tracked as a running maximum.
With enable(model, False) the fake rounding is switched off and the model is a plain float
network again (exported to ONNX and quantised for real by quantize_fpga.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TinyUNet, fuse_bn, CBR

QMAX = 127.0


def _round_ste(x):
    return x + (torch.round(x) - x).detach()


class QConv2d(nn.Conv2d):
    enabled = True

    def forward(self, x):
        if not self.enabled:
            return super().forward(x)
        w = self.weight
        s = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8) / QMAX
        wq = torch.clamp(_round_ste(w / s), -QMAX, QMAX) * s
        return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)


class FakeQuant(nn.Module):
    def __init__(self, momentum=0.01):
        super().__init__()
        self.register_buffer("amax", torch.tensor(0.0))
        self.momentum, self.enabled = momentum, True

    def forward(self, x):
        if not self.enabled:
            return x
        if self.training:
            m = x.detach().abs().amax()
            self.amax.copy_(m if self.amax == 0 else (1 - self.momentum) * self.amax + self.momentum * m)
        s = self.amax.clamp_min(1e-8) / QMAX
        return torch.clamp(_round_ste(x / s), -QMAX, QMAX) * s


def _qconv(c):
    q = QConv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, c.dilation, c.groups, bias=True)
    with torch.no_grad():
        q.weight.copy_(c.weight)
        q.bias.copy_(c.bias if c.bias is not None else torch.zeros(c.out_channels))
    return q


class QATUNet(nn.Module):
    """fused TinyUNet with fake int8 rounding; same forward graph as TinyUNet."""
    def __init__(self, fused):
        super().__init__()
        self.net = fused
        self.qin = FakeQuant()
        for mod in fused.modules():
            if isinstance(mod, CBR):
                mod[0] = _qconv(mod[0])
                mod[1] = FakeQuant()          # BN slot (Identity after folding) -> the ReLU output is quantised below
                mod[1], mod[2] = mod[2], mod[1]  # order: conv, ReLU, fake-quant
        fused.head = nn.Sequential(_qconv(fused.head), FakeQuant())

    def forward(self, x):
        return self.net(self.qin(x))


def make_qat(float_model):
    return QATUNet(fuse_bn(float_model))


def build_qat(n_classes, widths):
    """empty QAT model with the right structure (for loading a QAT checkpoint)."""
    return make_qat(TinyUNet(n_classes, widths))


def enable(model, on=True):
    for m in model.modules():
        if isinstance(m, (QConv2d, FakeQuant)):
            m.enabled = on
    return model
