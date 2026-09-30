@echo off
setlocal EnableDelayedExpansion
REM Full local pipeline: prepare data -> train Model 1 -> train Model 5 -> evaluate -> export.
REM Safe to re-run: finished steps are skipped, interrupted training resumes.
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
for /f "delims=" %%h in ('python -c "from dentassist import paths as P;print(P.HOME)"') do set "HOME_DIR=%%h"
echo Project location: %HOME_DIR%

if not exist "%HOME_DIR%\data\yolo\findings\data.yaml" (
  echo === 1/5 Preparing dataset from %HOME_DIR%\data\raw ===
  python scripts\prepare_data.py
  if errorlevel 1 goto :fail
) else ( echo === 1/5 dataset already prepared - skip )

if not exist "%HOME_DIR%\weights\teeth_best.pt" (
  echo === 2/5 Training Model 1 - teeth / FDI ===
  set "EXTRA="
  if exist "%HOME_DIR%\runs\teeth\weights\last.pt" set "EXTRA=--resume"
  python scripts\train.py --task teeth !EXTRA!
  if errorlevel 1 goto :fail
) else ( echo === 2/5 teeth model done - skip )

if not exist "%HOME_DIR%\weights\findings_best.pt" (
  echo === 3/5 Training Model 5 - findings ===
  set "EXTRA="
  if exist "%HOME_DIR%\runs\findings\weights\last.pt" set "EXTRA=--resume"
  python scripts\train.py --task findings !EXTRA!
  if errorlevel 1 goto :fail
) else ( echo === 3/5 findings model done - skip )

echo === 4/5 Evaluating on test set ===
python scripts\evaluate.py || goto :fail
echo === 5/5 Exporting ONNX ===
python scripts\export.py || goto :fail
echo.
echo ALL DONE. Report: %HOME_DIR%\results\REPORT.md   Models: %HOME_DIR%\weights
pause
exit /b 0
:fail
echo.
echo FAILED - screenshot this window and send it to Claude. Re-run this file to continue where it stopped.
pause
exit /b 1
