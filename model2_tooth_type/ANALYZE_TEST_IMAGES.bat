@echo off
REM Analyse every photo in test_images\  ->  results in test_images\results\
call "%~dp0..\model1_occlusal\code\.venv\Scripts\activate.bat" 2>nul || (echo First run setup once:  model1_occlusal\code\windows\setup.bat & pause & exit /b 1)
cd /d "%~dp0code"
set "IMGS="
for %%x in (jpg jpeg png JPG JPEG PNG) do (
  for %%f in ("%~dp0test_images\*.%%x") do (
    if exist "%%~f" call set IMGS=%%IMGS%% "%%~f"
  )
)
if not defined IMGS ( echo Put photos in:  %~dp0test_images & pause & exit /b 1 )
python scripts\analyze.py %IMGS% --out "%~dp0test_images\results"
echo.
echo Done. Opening results...
start "" "%~dp0test_images\results"
pause
