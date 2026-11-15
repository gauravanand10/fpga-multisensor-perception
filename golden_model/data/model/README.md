# Trained network

| File | Content |
|---|---|
| `tiny_unet_float32.onnx` | float32 network (ONNX), used for the float32 vs int8 comparison |
| `tiny_unet_int8_quantised.npz` | int8 weights, biases, scales and requantisation constants |
| `tiny_unet_model_info.json` | Classes, channel widths (16/32/80), input 160×120, output 80×60 |
| [`int8_parameters/`](int8_parameters) | Per-layer int8 parameters as `$readmemh` files |

Same network as [`model/weights/`](../../../model/weights); copied here so the golden model is self-contained.
