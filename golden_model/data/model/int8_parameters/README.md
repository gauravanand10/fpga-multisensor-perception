# int8 parameters per layer

For each of the 12 convolution layers *nnn* (L000 … L026): `Lnnn_Conv_w.mem` (int8 weights), `Lnnn_Conv_b.mem`
(int32 bias), `Lnnn_Conv_M.mem` (15-bit requantisation multiplier) and `Lnnn_Conv_sh.mem` (shift). For the two
concatenations (L015, L021): `Lnnn_Concat_M0.mem`, `_sh0.mem` (rescale of the upsampled input) and `_M1.mem`,
`_sh1.mem` (rescale of the skip input). All in `$readmemh` format.
They are packed into the FPGA weight and parameter ROMs by [`hw/tools/unet_pack.py`](../../../../hw/tools/unet_pack.py).
