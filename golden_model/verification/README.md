# Verification

| Script | Compares | Report |
|---|---|---|
| `verify_int8_vs_golden_vectors.py` | Accelerator model vs the per-layer golden tensors in `data/golden_vectors/` | `results/verification/int8_vs_golden_vectors.*` |
| `verify_rtl_simulation_output.py` | Golden model vs the UART bytes of the RTL simulation (`data/rtl_simulation/uart_capture.bin`) | `results/verification/rtl_simulation_vs_model.*` |
| `_common.py` | Shared path setup | |

```powershell
cd golden_model
python verification/verify_int8_vs_golden_vectors.py
python verification/verify_rtl_simulation_output.py
```
Both exit with code 0 only if every value matches.
