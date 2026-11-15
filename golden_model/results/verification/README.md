# Verification results

| File | Check | Result |
|---|---|---|
| `int8_vs_golden_vectors.json`, `.txt` | Accelerator model vs the golden tensors of every layer | 12/12 layers bit-exact, 0 label mismatches |
| `rtl_simulation_vs_model.json`, `.txt` | Golden model vs the UART bytes of the RTL simulation (`data/rtl_simulation/uart_capture.bin`) | 0 differing bytes, corrupted packet rejected by both |

Written by the scripts in [`../../verification`](../../verification).
