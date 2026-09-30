@echo off
REM One-time setup on Windows. Needs Python 3.10-3.12 from python.org (tick "Add to PATH").
cd /d "%~dp0\.."
python --version || (echo Python not found - install from https://www.python.org/downloads/ & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo NVIDIA GPU found - installing CUDA PyTorch
  nvidia-smi --query-gpu=name,memory.total --format=csv
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
  echo No NVIDIA GPU found - installing CPU PyTorch ^(training will be slow^)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)
pip install -r requirements.txt
call windows\set_location.bat
python -m dentassist.paths
python -c "import torch;print('GPU ready:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo Setup done.
echo Next: put the DENTEX files in the data\raw folder shown above, then run windows\train_local.bat
pause
