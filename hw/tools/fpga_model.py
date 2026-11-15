"""
fpga_model.py - bit-exact Python model of everything the FPGA does.

The RTL testbenches compare against THIS model, and this model is itself checked
against the golden vectors of fpga_ref (golden_model\\data\\golden_vectors).

  * U-Net layer table + BRAM region map          (LAYERS, build_layers)
  * weight / param / input-LUT packing             (pack_network)
  * int8 network exactly as the hardware runs it  (run_unet)
  * connected components, ToF zone painting, per-object distance  (postprocess)
  * SPI packets (ESP32 -> FPGA) and the UART packet (FPGA -> PC)  (*_packet, parse_uart_packet)

Only numpy is needed (cv2 only for reading the golden frame / the zone-table generator).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# hardware geometry  (must match rtl/unet_pkg.sv)
# ---------------------------------------------------------------------------
PO = 16                 # output-channel lanes
PI = 8                  # input-channel lanes
BANK_DEPTH = 12000      # bytes per activation bank (16 banks)
IMG_W, IMG_H = 160, 120  # camera frame the FPGA receives
LAB_W, LAB_H = 80, 60   # label map
NPIX = LAB_W * LAB_H
QMAX = 127
CLASSES = ["background", "person", "chair", "tv", "laptop", "bottle"]
N_CLS = len(CLASSES)
MIN_AREA = 24           # 0.005 * 4800, as seg_unet.SegUNet(min_area=0.005)
K_MAX = 63              # objects (components >= MIN_AREA) the FPGA can number (ids 1..63)
CC_MAX_PASSES = 32      # forward+backward passes before giving up (flag)

ID_M, ID_SH = 1 << 14, 14     # identity requant: (q*2^14 + 2^13) >> 14 == q


def _rd(path, bits):
    v = np.array([int(t, 16) for t in open(path).read().split()], np.int64)
    return np.where(v >= 1 << (bits - 1), v - (1 << bits), v)


# ---------------------------------------------------------------------------
# layer table
# ---------------------------------------------------------------------------
@dataclass
class Layer:
    name: str               # golden file tag, e.g. "L000_Conv"
    check: str              # golden tensor the hardware output must equal
    cin: int
    cout: int
    hin: int                # logical input size (after upsample)
    win: int
    k3: int
    s2: int
    relu: int
    argmax: int = 0
    src_frame: int = 0      # 1: input = camera frame (through the input LUT)
    # source A (channels 0..ca-1) and B (channels ca..cin-1)
    a_base: int = 0
    a_w: int = 0            # stored width / plane of A
    a_hw: int = 0
    a_up: int = 0           # nearest x2 upsample on read
    a_m: int = ID_M
    a_sh: int = ID_SH
    ca: int = 0
    b_base: int = 0
    b_w: int = 0
    b_hw: int = 0
    b_m: int = ID_M
    b_sh: int = ID_SH
    out_base: int = 0
    concat: str = ""        # golden Concat tag (for the M/sh of A and B)
    w_base: int = 0
    p_base: int = 0

    @property
    def hout(self):
        return self.hin // 2 if self.s2 else self.hin

    @property
    def wout(self):
        return self.win // 2 if self.s2 else self.win

    @property
    def kk(self):
        return 9 if self.k3 else 1

    @property
    def n_ocg(self):
        return -(-self.cout // PO)

    @property
    def n_icg(self):
        return -(-self.cin // PI)


# region map in every bank (bytes):  B=[0,4800)  R1=[4800,9600)  R2=[9600,12000)
def build_layers():
    L = []
    def add(**kw):
        L.append(Layer(**kw))
    # name, golden-check, cin, cout, hin, win, k3, s2, relu, sources, out
    add(name="L000_Conv", check="L001_Relu_out0", cin=1,  cout=16, hin=120, win=160, k3=1, s2=1, relu=1,
        src_frame=1, ca=1, out_base=4800)
    add(name="L002_Conv", check="L003_Relu_out0", cin=16, cout=16, hin=60, win=80, k3=1, s2=0, relu=1,
        a_base=4800, a_w=80, a_hw=4800, ca=16, out_base=0)                      # skip1 @0
    add(name="L004_Conv", check="L005_Relu_out0", cin=16, cout=32, hin=60, win=80, k3=1, s2=1, relu=1,
        a_base=0, a_w=80, a_hw=4800, ca=16, out_base=9600)
    add(name="L006_Conv", check="L007_Relu_out0", cin=32, cout=32, hin=30, win=40, k3=1, s2=0, relu=1,
        a_base=9600, a_w=40, a_hw=1200, ca=32, out_base=4800)                   # skip2 @4800
    add(name="L008_Conv", check="L009_Relu_out0", cin=32, cout=80, hin=30, win=40, k3=1, s2=1, relu=1,
        a_base=4800, a_w=40, a_hw=1200, ca=32, out_base=9600)
    add(name="L010_Conv", check="L011_Relu_out0", cin=80, cout=80, hin=15, win=20, k3=1, s2=0, relu=1,
        a_base=9600, a_w=20, a_hw=300, ca=80, out_base=7200)
    add(name="L012_Conv", check="L013_Relu_out0", cin=80, cout=80, hin=15, win=20, k3=1, s2=0, relu=1,
        a_base=7200, a_w=20, a_hw=300, ca=80, out_base=9600)
    add(name="L016_Conv", check="L017_Relu_out0", cin=112, cout=32, hin=30, win=40, k3=1, s2=0, relu=1,
        a_base=9600, a_w=20, a_hw=300, a_up=1, ca=80, b_base=4800, b_w=40, b_hw=1200,
        out_base=7200, concat="L015_Concat")
    add(name="L018_Conv", check="L019_Relu_out0", cin=32, cout=32, hin=30, win=40, k3=1, s2=0, relu=1,
        a_base=7200, a_w=40, a_hw=1200, ca=32, out_base=9600)
    add(name="L022_Conv", check="L023_Relu_out0", cin=48, cout=16, hin=60, win=80, k3=1, s2=0, relu=1,
        a_base=9600, a_w=40, a_hw=1200, a_up=1, ca=32, b_base=0, b_w=80, b_hw=4800,
        out_base=4800, concat="L021_Concat")
    add(name="L024_Conv", check="L025_Relu_out0", cin=16, cout=16, hin=60, win=80, k3=1, s2=0, relu=1,
        a_base=4800, a_w=80, a_hw=4800, ca=16, out_base=0)
    add(name="L026_Conv", check="L026_Conv_out0", cin=16, cout=6, hin=60, win=80, k3=0, s2=0, relu=0, argmax=1,
        a_base=0, a_w=80, a_hw=4800, ca=16, out_base=4800)
    w = p = 0
    for l in L:
        l.w_base, l.p_base = w, p
        w += l.n_ocg * l.n_icg * l.kk
        p += l.n_ocg
    return L


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------
@dataclass
class Packed:
    layers: list
    weights: np.ndarray       # (n_words, PO, PI) int8
    params: list              # per word: list of PO (bias, M, sh)
    in_lut: np.ndarray        # (256,) int8
    raw: dict = field(default_factory=dict)   # per layer: W, b, M, sh (flat golden arrays)


def input_lut(scale):
    """q = clamp(floor((p/255)/s + 0.5)) in float32, exactly as fpga_ref.engine.quantize."""
    x = np.arange(256, dtype=np.float32) / np.float32(255.0)
    return np.clip(np.floor(x / np.float32(scale) + np.float32(0.5)), -QMAX, QMAX).astype(np.int8)


def pack_network(memdir, gray_scale):
    layers = build_layers()
    n_words = sum(l.n_ocg * l.n_icg * l.kk for l in layers)
    Wp = np.zeros((n_words, PO, PI), np.int8)
    params = []
    raw = {}
    for l in layers:
        t = os.path.join(memdir, l.name)
        W = _rd(t + "_w.mem", 8).reshape(l.cout, l.cin, l.kk)
        b = _rd(t + "_b.mem", 32); M = _rd(t + "_M.mem", 32); sh = _rd(t + "_sh.mem", 32)
        assert b.size == l.cout, l.name
        assert np.all((M >= 0) & (M < 1 << 15)) and np.all((sh >= 1) & (sh < 64)), l.name
        raw[l.name] = (W, b, M, sh)
        if l.concat:
            c = os.path.join(memdir, l.concat)
            l.a_m, l.a_sh = int(_rd(c + "_M0.mem", 32)[0]), int(_rd(c + "_sh0.mem", 32)[0])
            l.b_m, l.b_sh = int(_rd(c + "_M1.mem", 32)[0]), int(_rd(c + "_sh1.mem", 32)[0])
            assert 0 <= l.a_m < 1 << 15 and 0 <= l.b_m < 1 << 15
        a = l.w_base
        for og in range(l.n_ocg):
            for ig in range(l.n_icg):
                for tp in range(l.kk):
                    for p in range(PO):
                        oc = og * PO + p
                        if oc >= l.cout:
                            continue
                        for i in range(PI):
                            ic = ig * PI + i
                            if ic < l.cin:
                                Wp[a, p, i] = W[oc, ic, tp]
                    a += 1
            lanes = []
            for p in range(PO):
                oc = og * PO + p
                lanes.append((int(b[oc]), int(M[oc]), int(sh[oc])) if oc < l.cout else (0, 0, 1))
            params.append(lanes)
    return Packed(layers, Wp, params, input_lut(gray_scale), raw)


def requant(x, M, sh):
    x = np.asarray(x, np.int64)
    return np.clip((x * M + (np.int64(1) << (sh - 1))) >> sh, -QMAX, QMAX)


# ---------------------------------------------------------------------------
# the network, as the hardware computes it
# ---------------------------------------------------------------------------
def run_unet(frame_u8, pk: Packed):
    """frame_u8: (120,160) uint8 camera frame. Returns (per-layer outputs dict, label map (60,80))."""
    assert frame_u8.shape == (IMG_H, IMG_W)
    bank = np.zeros((PO, BANK_DEPTH), np.int64)        # the 16 activation banks
    outs = {}
    labels = None
    q_in = pk.in_lut[frame_u8.astype(np.int64)].astype(np.int64)
    for l in pk.layers:
        W, b, M, sh = pk.raw[l.name]
        # gather the logical input tensor (cin, hin, win) the way the hardware reads it
        x = np.zeros((l.cin, l.hin, l.win), np.int64)
        if l.src_frame:
            x[0] = q_in
        else:
            yy, xx = np.mgrid[0:l.hin, 0:l.win]
            for c in range(l.cin):
                if c < l.ca:
                    loc, base, wd, hw, up, m_, s_ = c, l.a_base, l.a_w, l.a_hw, l.a_up, l.a_m, l.a_sh
                else:
                    loc, base, wd, hw, up, m_, s_ = c - l.ca, l.b_base, l.b_w, l.b_hw, 0, l.b_m, l.b_sh
                addr = base + (loc // PO) * hw + (yy >> up) * wd + (xx >> up)
                x[c] = requant(bank[loc % PO][addr], m_, s_)
        # conv
        pad = 1 if l.k3 else 0
        k = 3 if l.k3 else 1
        s = 2 if l.s2 else 1
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
        acc = np.zeros((l.cout, l.hout, l.wout), np.int64) + b[:, None, None]
        Wk = W.reshape(l.cout, l.cin, k, k)
        for u in range(k):
            for v in range(k):
                patch = xp[:, u:u + s * (l.hout - 1) + 1:s, v:v + s * (l.wout - 1) + 1:s]
                acc += np.einsum("oc,chw->ohw", Wk[:, :, u, v], patch)
        y = requant(acc, M[:, None, None], sh[:, None, None])
        if l.relu:
            y = np.maximum(y, 0)
        outs[l.name] = y
        for oc in range(l.cout):
            bank[oc % PO][l.out_base + (oc // PO) * l.hout * l.wout + np.arange(l.hout * l.wout)] = y[oc].ravel()
        if l.argmax:
            labels = np.argmax(y, axis=0).astype(np.int64)       # first max, like numpy / the RTL
    return outs, labels


def check_against_golden(memdir, pk, frame_u8):
    outs, labels = run_unet(frame_u8, pk)
    res = []
    for l in pk.layers:
        g = _rd(os.path.join(memdir, l.check + ".mem"), 8).reshape(l.cout, l.hout, l.wout)
        res.append((l.name, l.check, int((outs[l.name] != g).sum())))
    return res, outs, labels


# ---------------------------------------------------------------------------
# writing .mem files
# ---------------------------------------------------------------------------
def _hex(v, bits):
    return format(int(v) & ((1 << bits) - 1), "0%dx" % (bits // 4))


def weight_words(pk):
    """one PO*PI*8 = 1024-bit word per line, lane (p,i) at bits [(p*PI+i)*8 +: 8]."""
    lines = []
    for wd in pk.weights:
        v = 0
        for p in range(PO):
            for i in range(PI):
                v |= (int(wd[p, i]) & 0xFF) << ((p * PI + i) * 8)
        lines.append(_hex(v, PO * PI * 8))
    return lines


def param_words(pk):
    """one PO*56-bit word per line, lane p = {sh[7:0], M[15:0], bias[31:0]} at bits [p*56 +: 56]."""
    lines = []
    for lanes in pk.params:
        v = 0
        for p, (b, m, s) in enumerate(lanes):
            lane = ((s & 0xFF) << 48) | ((m & 0xFFFF) << 32) | (b & 0xFFFFFFFF)
            v |= lane << (p * 56)
        lines.append(_hex(v, PO * 56))
    return lines


# ---------------------------------------------------------------------------
# ToF zones on the label grid  (tables come from tools/gen_zone_tables.py)
# ---------------------------------------------------------------------------
TOF_VALID_STATUS = (5, 6, 9, 10, 12)      # esp32_tof_v2 valid()
TOF_MIN_MM, TOF_MAX_MM = 20, 4000         # geo/config.py
NO_ZONE = 127
ZBUF_INF = 0xFFFF


@dataclass
class ZoneTables:
    rz_q15: np.ndarray        # (64,) ray_z of each RAW zone centre, Q15 (depth_cam = d*rz)
    bin_lo: np.ndarray        # (NB,) lower distance edge of each bin, mm (bin b: lo[b] <= d < lo[b+1])
    rects: np.ndarray         # (64, NB, 5) x0, y0, x1, y1 (inclusive), valid


def load_zone_tables(path):
    z = np.load(path)
    return ZoneTables(z["rz_q15"].astype(np.int64), z["bin_lo"].astype(np.int64), z["rects"].astype(np.int64))


def tof_valid(dist, status):
    dist = np.asarray(dist, np.int64); status = np.asarray(status, np.int64)
    ok = np.isin(status, TOF_VALID_STATUS) & (dist >= TOF_MIN_MM) & (dist <= TOF_MAX_MM)
    return ok


def depth_bin(d, zt):
    """largest b with bin_lo[b] <= d (bin_lo[0] = TOF_MIN_MM)."""
    b = 0
    for i in range(1, len(zt.bin_lo)):
        if d >= zt.bin_lo[i]:
            b = i
    return b


def paint_zones(dist, status, zt):
    """z-buffer paint of the 64 zone rectangles on the 80x60 grid (zone index order, nearer or equal wins)."""
    zid = np.full(NPIX, NO_ZONE, np.int64)
    zbuf = np.full(NPIX, ZBUF_INF, np.int64)
    ok = tof_valid(dist, status)
    dcam = np.zeros(64, np.int64)
    for i in range(64):
        if not ok[i]:
            continue
        d = int(dist[i])
        dc = (d * int(zt.rz_q15[i]) + (1 << 14)) >> 15
        dcam[i] = dc
        x0, y0, x1, y1, v = zt.rects[i, depth_bin(d, zt)]
        if not v:
            continue
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                p = y * LAB_W + x
                if dc <= zbuf[p]:
                    zbuf[p] = dc
                    zid[p] = i
    area = np.bincount(zid[zid != NO_ZONE], minlength=64)[:64]
    return zid, dcam, area, ok


def connected_components(labels):
    """8-connected components of equal non-zero class (union-find; the FPGA gets the same sets with
    min-label passes). Only components with >= MIN_AREA pixels get an id, numbered 1..K_MAX in raster
    order of their first pixel, exactly as the FPGA numbers them.
    Returns kmap (0 = background / small / beyond K_MAX), n_numbered, n_big (components >= MIN_AREA)."""
    lab = labels.ravel()
    parent = np.arange(NPIX)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for y in range(LAB_H):
        for x in range(LAB_W):
            p = y * LAB_W + x
            if lab[p] == 0:
                continue
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < LAB_H and 0 <= xx < LAB_W:
                    q = yy * LAB_W + xx
                    if lab[q] == lab[p]:
                        ra, rb = find(p), find(q)
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(p) if lab[p] else -1 for p in range(NPIX)])
    area = np.bincount(root[root >= 0], minlength=NPIX)
    kmap = np.zeros(NPIX, np.int64)
    root2k = {}
    nk = n_big = 0
    for p in range(NPIX):
        if lab[p] == 0:
            continue
        r = root[p]
        if r == p:
            k = 0
            if area[p] >= MIN_AREA:
                n_big += 1
                if nk < K_MAX:
                    nk += 1; k = nk
            root2k[p] = k
        kmap[p] = root2k[r]
    return kmap, nk, n_big


def object_distance(cnt, area, dcam, ok):
    """FPGA integer version of the original PC pipeline: zones with coverage >= 0.3 (else the best zone if > 0.05),
    then the coverage-weighted 35th percentile of their depths. cnt/area: per-zone ints. Returns (mm, n_sel)."""
    sel = [z for z in range(64) if area[z] > 0 and 10 * cnt[z] >= 3 * area[z]]
    if not sel:
        best = -1
        for z in range(64):
            if area[z] == 0 or cnt[z] == 0:
                continue
            if best < 0 or cnt[z] * area[best] > cnt[best] * area[z]:
                best = z
        if best >= 0 and 20 * cnt[best] > area[best]:
            sel = [best]
    if not sel:
        return 0, 0
    w = {z: (int(cnt[z]) << 12) // int(area[z]) for z in sel}
    W = sum(w.values())
    d = None
    for j in sel:
        S = sum(w[i] for i in sel if dcam[i] <= dcam[j])
        if 20 * S >= 7 * W and (d is None or dcam[j] < d):
            d = int(dcam[j])
    return d, len(sel)


@dataclass
class Obj:
    k: int
    cls: int
    area: int
    x0: int
    y0: int
    x1: int
    y1: int
    dist: int
    nsel: int


def postprocess(labels, dist, status, zt):
    """labels (60,80) -> (objects, kmap_out (4800,), flags dict)."""
    kmap, nk, n_big = connected_components(labels)
    zid, dcam, zarea, ok = paint_zones(dist, status, zt)
    lab = labels.ravel()
    objs = []
    kmap_out = np.zeros(NPIX, np.int64)
    for k in range(1, nk + 1):
        m = kmap == k
        a = int(m.sum())
        ys, xs = np.divmod(np.nonzero(m)[0], LAB_W)
        cls = int(lab[np.nonzero(m)[0][0]])
        z = zid[m]
        cnt = np.bincount(z[z != NO_ZONE], minlength=64)[:64]
        d, ns = object_distance(cnt, zarea, dcam, ok)
        objs.append(Obj(k, cls, a, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), d, ns))
        kmap_out[m] = k
    flags = {"overflow": int(n_big > K_MAX)}
    return objs, kmap_out, flags, dict(zid=zid, dcam=dcam, zarea=zarea, kmap_all=kmap, n_big=n_big)


# ---------------------------------------------------------------------------
# packets
# ---------------------------------------------------------------------------
CAM_MAGIC = b"CAM1"
TOF_MAGIC = b"TOF1"
UART_SYNC = bytes([0xA5, 0x5A, 0x46, 0x50])      # A5 5A 'F' 'P'
UART_VERSION = 1


def sum16(b):
    return sum(b) & 0xFFFF


def cam_packet(frame_u8, frame_no):
    """ESP32-CAM -> FPGA over SPI (one CS-low burst):
       'CAM1', frame_no u16, 19200 pixels (row-major 160x120), sum16(pixels) u16   (little-endian)"""
    px = np.asarray(frame_u8, np.uint8).ravel().tobytes()
    assert len(px) == IMG_W * IMG_H
    return CAM_MAGIC + struct.pack("<H", frame_no & 0xFFFF) + px + struct.pack("<H", sum16(px))


def tof_packet(dist, status, frame_no):
    """ToF ESP32 -> FPGA over SPI: 'TOF1', frame_no u16, 64 x u16 distance_mm (ST raw index order),
       64 x u8 target_status, sum16(distance+status bytes) u16"""
    body = struct.pack("<64H", *[int(d) & 0xFFFF for d in dist]) + bytes(int(s) & 0xFF for s in status)
    return TOF_MAGIC + struct.pack("<H", frame_no & 0xFFFF) + body + struct.pack("<H", sum16(body))


OBJ_BYTES = 12
UART_HDR = 19


def uart_packet(cam_no, tof_no, tof_ok, infer_cycles, post_cycles, objs, kmap_out, dist, status,
                frame_u8=None, flags_extra=0):
    """FPGA -> PC (USB-UART). All little-endian.
      0  A5 5A 46 50     sync
      4  u8  version (1)
      5  u8  flags  bit0 image included, bit1 ToF frame valid, bit2 >63 components, bit3 CC not converged
      6  u16 camera frame_no      8  u16 ToF frame_no
     10  u32 U-Net cycles        14  u32 post-processing cycles
     18  u8  n_obj
     19  n_obj x 12: id, class, area u16, x0, y0, x1, y1, dist_mm u16 (0 = none), n_zones, 0
         64 x u16 ToF distance, 64 x u8 ToF status
         4800 x u8 object map (80x60, value = object id, 0 = none)
         [19200 x u8 camera frame 160x120]   if flags bit0
         u16 sum16 of all bytes after the 4-byte sync
    """
    flags = (1 if frame_u8 is not None else 0) | (2 if tof_ok else 0) | flags_extra
    b = bytearray()
    b += bytes([UART_VERSION, flags])
    b += struct.pack("<HHIIB", cam_no & 0xFFFF, tof_no & 0xFFFF, infer_cycles & 0xFFFFFFFF,
                     post_cycles & 0xFFFFFFFF, len(objs))
    for o in objs:
        b += struct.pack("<BBHBBBBHBB", o.k, o.cls, o.area, o.x0, o.y0, o.x1, o.y1, o.dist, o.nsel, 0)
    b += struct.pack("<64H", *[int(d) & 0xFFFF for d in dist]) + bytes(int(s) & 0xFF for s in status)
    b += bytes(int(v) for v in kmap_out)
    if frame_u8 is not None:
        b += np.asarray(frame_u8, np.uint8).tobytes()
    return UART_SYNC + bytes(b) + struct.pack("<H", sum16(b))


def parse_uart_packet(buf):
    """buf starts at the sync. Returns (dict, total_length) or (None, needed_length) if incomplete,
    raises ValueError on a bad checksum."""
    if len(buf) < UART_HDR:
        return None, UART_HDR
    ver, flags, cam_no, tof_no, cyc, pcyc, n = struct.unpack_from("<BBHHIIB", buf, 4)
    n_img = IMG_W * IMG_H if flags & 1 else 0
    total = UART_HDR + n * OBJ_BYTES + 192 + NPIX + n_img + 2
    if len(buf) < total:
        return None, total
    body = buf[4:total - 2]
    if sum16(body) != struct.unpack_from("<H", buf, total - 2)[0]:
        raise ValueError("UART packet checksum mismatch")
    o = UART_HDR
    objs = []
    for _ in range(n):
        k, c, a, x0, y0, x1, y1, d, ns, _r = struct.unpack_from("<BBHBBBBHBB", buf, o)
        objs.append(Obj(k, c, a, x0, y0, x1, y1, d, ns)); o += OBJ_BYTES
    dist = np.array(struct.unpack_from("<64H", buf, o)); o += 128
    status = np.frombuffer(buf, np.uint8, 64, o).copy(); o += 64
    kmap = np.frombuffer(buf, np.uint8, NPIX, o).reshape(LAB_H, LAB_W).copy(); o += NPIX
    img = np.frombuffer(buf, np.uint8, n_img, o).reshape(IMG_H, IMG_W).copy() if n_img else None
    return dict(version=ver, flags=flags, cam_no=cam_no, tof_no=tof_no, infer_cycles=cyc, post_cycles=pcyc,
                objs=objs, tof_dist=dist, tof_status=status, kmap=kmap, image=img), total


def downsample_2x2(img320):
    """ESP32-CAM side: 320x240 -> 160x120, (a+b+c+d+2)>>2 == cv2.INTER_AREA for exactly 2x."""
    a = np.asarray(img320, np.int32)
    s = a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]
    return ((s + 2) >> 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# IMU / servo packet: sent between result packets
# ---------------------------------------------------------------------------
IMU_SYNC = bytes([0xA5, 0x5A, 0x49, 0x4D])       # A5 5A 'I' 'M'
IMU_VERSION = 1
IMU_HDR = 14                                     # sync 4 + header 10
IMU_SAMPLE_BYTES = 26                            # sample_no u16 + 12 x int16
IMU_FIFO_DEPTH = 32
IMU_FIELDS = ["ax1", "ay1", "az1", "gx1", "gy1", "gz1", "ax2", "ay2", "az2", "gx2", "gy2", "gz2"]
ACC_LSB_PER_G = 16384.0                          # +-2 g
GYRO_LSB_PER_DPS = 131.0                         # +-250 deg/s
SERVO_MODES = {0: "manual", 1: "sweep", 2: "external"}


def imu_packet(imu_ok, who1, who2, servo_mode, target_us, now_us, samples, overflow=False):
    """FPGA -> PC, between result packets. All little-endian.
      0  A5 5A 49 4D   sync
      4  u8  version (1)
      5  u8  flags  bit0 IMU1 ok, bit1 IMU2 ok, bit2 FIFO overflow (samples were dropped)
      6  u8  WHO_AM_I IMU1     7  u8  WHO_AM_I IMU2   (0xFF = nothing answers)
      8  u8  servo mode (0 manual, 1 sweep, 2 external)
      9  u16 servo target pulse us    11  u16 servo pulse us now
     13  u8  n (1..32)
     14  n x { u16 sample_no, 12 x int16: ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2 }
         u16 sum16 of all bytes after the 4-byte sync
    imu_ok: 2-bit value (bit0 IMU1, bit1 IMU2). samples: list of (sample_no, 12 ints)."""
    assert 1 <= len(samples) <= 255
    flags = (imu_ok & 3) | (4 if overflow else 0)
    b = bytearray([IMU_VERSION, flags, who1 & 0xFF, who2 & 0xFF, servo_mode & 0xFF])
    b += struct.pack("<HHB", target_us & 0xFFFF, now_us & 0xFFFF, len(samples))
    for sno, v in samples:
        assert len(v) == 12
        b += struct.pack("<H12h", sno & 0xFFFF, *[((int(x) + 0x8000) & 0xFFFF) - 0x8000 for x in v])
    return IMU_SYNC + bytes(b) + struct.pack("<H", sum16(b))


def parse_imu_packet(buf):
    """buf starts at the IMU sync. Returns (dict, total_length) or (None, needed_length) if incomplete,
    raises ValueError on a bad checksum / version / sample count."""
    if len(buf) < IMU_HDR:
        return None, IMU_HDR
    ver, flags, who1, who2, mode, tgt, now, n = struct.unpack_from("<BBBBBHHB", buf, 4)
    if ver != IMU_VERSION or n == 0 or n > IMU_FIFO_DEPTH:
        raise ValueError(f"IMU packet header invalid (version {ver}, n {n})")
    total = IMU_HDR + n * IMU_SAMPLE_BYTES + 2
    if len(buf) < total:
        return None, total
    if sum16(buf[4:total - 2]) != struct.unpack_from("<H", buf, total - 2)[0]:
        raise ValueError("IMU packet checksum mismatch")
    samples = []
    o = IMU_HDR
    for _ in range(n):
        v = struct.unpack_from("<H12h", buf, o)
        samples.append((v[0], list(v[1:])))
        o += IMU_SAMPLE_BYTES
    return dict(version=ver, flags=flags, imu_ok=flags & 3, overflow=bool(flags & 4), who1=who1, who2=who2,
                servo_mode=mode, servo_target_us=tgt, servo_now_us=now, samples=samples), total


def servo_us_to_deg(us):
    """1000 us = 0 deg, 2000 us = 180 deg (servo_pwm.sv)."""
    return (us - 1000) * 180.0 / 1000.0


def parse_stream(buf):
    """Walk a UART byte stream that holds result packets and IMU packets.
    Returns (items, consumed, n_bad): items = [("FP", dict) | ("IM", dict)], consumed = bytes
    that can be dropped. Bytes before a sync are skipped (n_bad is not counted for them)."""
    items, n_bad, pos = [], 0, 0
    while True:
        i = buf.find(b"\xA5\x5A", pos)
        if i < 0 or len(buf) - i < 4:
            return items, (max(pos, len(buf) - 3) if i < 0 else i), n_bad
        sync = bytes(buf[i:i + 4])
        if sync not in (UART_SYNC, IMU_SYNC):
            pos = i + 1
            continue
        try:
            if sync == UART_SYNC:
                pkt, n = parse_uart_packet(bytes(buf[i:]))
            else:
                pkt, n = parse_imu_packet(bytes(buf[i:]))
        except ValueError:
            n_bad += 1
            pos = i + 4
            continue
        if pkt is None:
            return items, i, n_bad
        items.append(("FP" if sync == UART_SYNC else "IM", pkt))
        pos = i + n
