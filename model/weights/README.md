# Weights

| File | Content |
|---|---|
| `unet_float32.pt` | Best float32 checkpoint (PyTorch) |
| `unet_int8_qat.pt` | Best quantisation-aware checkpoint (the deployed network) |
| `unet_float32.onnx` | ONNX export of the network |
| `unet_int8.npz` | int8 weights, biases, scales and requantisation constants used by the FPGA |

6 classes, input 160×120 grey, output 80×60 label map, 206 K parameters.

```python
import numpy as np
q = np.load("model/weights/unet_int8.npz")
print(list(q.keys())[:8])
```
