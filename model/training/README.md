# Training

PyTorch scripts that build the dataset, train the Tiny U-Net and produce the int8 package. Settings are in
`config.py`.

| Script | Purpose |
|---|---|
| `config.py` | Classes, paths, image size, training settings |
| `prepare_coco.py` | Downloads the COCO 2017 subset and converts it to 320×240 grey + labels |
| `dataset.py` | Data loading with ESP32-CAM-like augmentation (exposure, blur, noise, JPEG) |
| `model.py` | Tiny U-Net built only from layers the FPGA engine supports |
| `train.py` | Float training, then quantisation-aware fine-tuning |
| `qat.py` | Fake-quantisation layers that mimic the FPGA's int8 rounding |
| `export_onnx.py` | Best checkpoint → ONNX |
| `evaluate.py` | IoU per class of the exported model |
| `quantize_fpga.py` | int8 package, golden vectors, bit-exact check |
| `record_frames.py`, `label_own_frames.py` | Optional: record and auto-label frames from your own camera |
| `run_all.bat` | All steps in order |
| `requirements.txt` | Python packages (PyTorch with CUDA installed separately) |

```powershell
cd model\training
pip install -r requirements.txt
python prepare_coco.py
python train.py --epochs 120
python export_onnx.py
python evaluate.py
python quantize_fpga.py
```
Checkpoints go to `model/runs/` (not in the repository); the final ones are in [`../weights`](../weights).
