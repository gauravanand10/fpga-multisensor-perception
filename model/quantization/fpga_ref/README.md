# int8 toolkit

Pure-NumPy tools that define and check the FPGA's integer arithmetic. The rules are in
[`INT8_SPEC.md`](INT8_SPEC.md): int8 weights per output channel, int8 activations per tensor, int32
accumulation, 15-bit multiplier and shift requantisation.

| Script | Purpose |
|---|---|
| `export_graph.py` | ONNX → FPGA-friendly float package (`models/fpga/*_fp32.npz`, `*_graph.json`) |
| `calibrate.py` | float package + calibration frames → int8 package (`models/fpga/*_int8.npz`) |
| `engine.py` | Integer reference inference: the golden model of the accelerator |
| `run_ref.py` | Runs the engine, writes per-layer golden vectors (`--dump --mem`) |
| `verify_mem.py` | Recomputes every layer from the `.mem` files alone, as an independent check |

```powershell
cd model\quantization
python fpga_ref\export_graph.py
python fpga_ref\run_ref.py --dump --mem
python fpga_ref\verify_mem.py
```
