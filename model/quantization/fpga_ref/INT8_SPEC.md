# Integer arithmetic of the FPGA reference model

This file defines the maths that `engine.py` (int8 mode) does. The hardware has to do exactly the same arithmetic to give bit-identical results. `verify_mem.py` re-implements these rules separately from the `.mem` files and checks every layer.

## Numbers

| What | Format |
|---|---|
| Activations | int8, symmetric, zero point 0, range −127…127. There is one scale `s` per tensor: `real = q · s`. |
| Weights | int8, symmetric, range −127…127. There is one scale per **output** channel. |
| Bias | int32, at scale `s_in · s_w[c]`. |
| Accumulator | int32. Measured peaks: YOLO 2^18; MiDaS 2^30, from a few channels with a large folded-BatchNorm bias. Both fit int32, and `acc · M` stays below 2^46. |
| Requantisation | `M` (15-bit, 2^14 ≤ M < 2^15) and shift `sh`, per output channel. |

The 15-bit `M` fits the 18-bit input of a DSP48, and `acc · M` fits its 48-bit accumulator.

With `--abits 16` the activations are int16 (range ±32767) and everything else stays the same.

Rounding is always *round half up*: `floor(x + 0.5)`. A right shift `>>` is arithmetic (floor).

## Layers

**Conv** (any kernel, stride, padding, dilation or group; padding value 0) and **ConvTranspose** (kernel = stride only):

```
acc[c,y,x] = b[c] + Σ  w[c,i,u,v] · q_in[i, y·s+u·d, x·s+v·d]      (int8 × int8, int32 sum)
q_out      = clamp( (acc · M[c] + 2^(sh[c]−1)) >> sh[c] , −127, 127 )
```

Here `M[c] · 2^−sh[c] ≈ s_in · s_w[c] / s_out`.

**SiLU** (YOLOv8 `x·sigmoid(x)`) is a 256-entry table per layer: `q_out = lut[q_in + 128]`. With 16-bit activations the index is `q_in + 32768`.

**Add**:

```
q_out = clamp( ((q_a·M0 + 2^(sh0−1)) >> sh0) + ((q_b·M1 + 2^(sh1−1)) >> sh1), −127, 127 )
```

**Concat**: each input is rescaled with its own `M_k / sh_k` (the same requant formula, clamped), then the channels are placed one after another.

**Pass-through**: MaxPool, nearest Resize, Split, Slice, Reshape, Transpose, ReLU, Clip and Pad with value 0 work directly on the int8 values and keep the input's scale. Clip uses `round(min/s) … round(max/s)`, and MaxPool padding uses −127.

The following run in **float** on the dequantised values (`q · s`). They are host-side or post-processing work, and the hardware would do them with a small fixed-point block:

- the input normalisation (MiDaS);
- bilinear Resize (the MiDaS decoder);
- the YOLO box decoding (DFL softmax, anchors, sigmoid of the class scores).

If the input of a Conv is a float tensor, it is quantised first: `q = clamp(floor(x/s + 0.5), −127, 127)`.

## Files

`models/fpga/<model>_graph.json` lists the layers in execution order. Each layer has `op`, `inputs`, `outputs` and `attrs`. Every layer is identified by the name of its first output tensor.

`models/fpga/<model>_int8.npz` holds the values per layer `<o>` (keys):

| Key | Contents |
|---|---|
| `<o>__w` | int8 weights. Conv: O,I/g,kh,kw. ConvTranspose: I,O,kh,kw. |
| `<o>__b` | int32 bias |
| `<o>__M`, `<o>__sh` | requantisation (int32 arrays, one entry per output channel) |
| `<o>__wscale` | float32 weight scale per output channel (information only) |
| `<o>__lut` | SiLU table |
| `<o>__M0 __sh0 __M1 __sh1` | Add |
| `<o>__M<k> __sh<k>` | Concat input k |
| `<t>__s` | float32 activation scale. A pass-through tensor uses its source's scale. |
| `__qmax` | 127 (int8) or 32767 (int16) |

Other keys are float constants for the float layers (shapes, resize factors, YOLO decoding tables).

`fpga_ref/out/golden_<model>/` holds one camera frame through the whole network:

- `tensors.npz`: every tensor (int8 where integer, float32 otherwise);
- `index.json`: each tensor's layer, shape, scale and inputs;
- `mem/`: `$readmemh` files per integer layer (`_w _b _M _sh _lut … _out0`) plus `input_q.mem`, in two's-complement hex, C order.
