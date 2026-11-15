# Quantisation

Turns the trained float network into the int8 package the FPGA runs, and provides the integer reference
engine that defines the FPGA's arithmetic.

| Path | Content |
|---|---|
| [`fpga_ref/`](fpga_ref) | int8 toolkit: graph export, calibration, reference engine, golden vectors; [`INT8_SPEC.md`](fpga_ref/INT8_SPEC.md) |
| [`models/`](models) | Exported ONNX model and the FPGA packages |
| `seg_unet.py` | Runs the exported network (ONNX Runtime) as a segmentation module for the PC viewers |
