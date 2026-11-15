@echo off
rem Full U-Net pipeline: data -> training -> ONNX -> evaluation -> FPGA int8 package
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -c "import torch,sys; ok=torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND - see README')"
python prepare_coco.py || goto :err
python label_own_frames.py || goto :err
python train.py || goto :err
python export_onnx.py || goto :err
python evaluate.py || goto :err
python quantize_fpga.py || goto :err
echo.
echo DONE - see model\runs\ and model\README.md
pause
exit /b 0
:err
echo.
echo FAILED - see the message above
pause
exit /b 1
