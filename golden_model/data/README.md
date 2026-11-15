# Golden model data

Read-only inputs of the golden model. Nothing here is written by the scripts.

| Folder | Content |
|---|---|
| [`model/`](model) | Trained network: float32 ONNX, int8 package, per-layer int8 parameters |
| [`golden_vectors/`](golden_vectors) | Per-layer golden tensors of the test frame (`$readmemh` format) |
| [`rtl_simulation/`](rtl_simulation) | UART bytes produced by the whole-FPGA RTL simulation |
| [`tof_geometry/`](tof_geometry) | Footprint of each ToF zone on the 80×60 label grid |
