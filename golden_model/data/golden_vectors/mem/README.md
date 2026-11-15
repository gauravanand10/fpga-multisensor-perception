# Golden vector files

One `$readmemh` file per tensor (see [`../README.md`](../README.md)): `input_q.mem` is the quantised input
frame; `Lnnn_Conv_{w,b,M,sh}.mem` are the parameters of conv layer *nnn* and `Lnnn_Concat_{M0,sh0,M1,sh1}.mem`
the rescale constants of the two concatenations; `Lnnn_<op>_out0.mem` is the output
of layer *nnn*. Values are two's-complement hex, one value per line.
