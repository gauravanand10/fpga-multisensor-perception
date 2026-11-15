# FPGA packages

| File | Content |
|---|---|
| `tiny_unet_graph.json` | Layer graph: operators, shapes, connections |
| `tiny_unet_fp32.npz` | float32 weights in the graph's layout |
| `tiny_unet_int8.npz` | int8 weights, int32 biases, scales, multipliers and shifts (the deployed network) |

`tiny_unet_int8.npz` is the source of `model/weights/unet_int8.npz` and of the FPGA ROM images in `hw/mem/`.
