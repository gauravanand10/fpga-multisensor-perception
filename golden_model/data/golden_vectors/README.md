# Golden vectors

The integer reference engine (`model/quantization/fpga_ref`) run on one test frame, dumped layer by layer in
`$readmemh` hex format. The RTL testbenches and `verification/verify_int8_vs_golden_vectors.py` compare against
these files.

| File | Content |
|---|---|
| `layer_index.json` | Layer list: name, type, shapes, scales, file names |
| `mem/input_q.mem` | Quantised 160×120 input frame |
| `mem/Lnnn_Conv_w.mem`, `_b.mem`, `_M.mem`, `_sh.mem` | Weights, biases, requantisation multiplier and shift of conv layer *nnn* |
| `mem/Lnnn_Concat_M0.mem`, `_sh0.mem`, `_M1.mem`, `_sh1.mem` | Rescale constants of the two concatenations (L015, L021) |
| `mem/Lnnn_<op>_out0.mem` | Output tensor of layer *nnn* (Conv, Relu, Resize, Concat) |

Number format and rounding rules: [`model/quantization/fpga_ref/INT8_SPEC.md`](../../../model/quantization/fpga_ref/INT8_SPEC.md).
