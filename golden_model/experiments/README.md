# Experiments

| Script | What it does | Output |
|---|---|---|
| `run_end_to_end_demo.py` | 8 COCO scenes through every stage (camera node, SPI, U-Net, post-processing, UART, host), ToF simulated from the true layout; damaged-packet checks | [`results/end_to_end/`](../results/end_to_end) |
| `evaluate_float_vs_int8.py` | float32 (ONNX) vs int8 (FPGA arithmetic) on 1000 COCO validation images | [`results/accuracy/`](../results/accuracy) |
| `_common.py` | Shared paths, colours and plot style | |

```powershell
cd golden_model
python experiments/run_end_to_end_demo.py          # about 1 min
python experiments/evaluate_float_vs_int8.py       # about 4 min (--n 200 for a quick run)
```
The graphs are drawn afterwards from the stored results by `python ../docs/make_plots.py`.
