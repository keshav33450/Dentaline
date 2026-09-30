@echo off
REM Analyse every X-ray in test_images\  ->  results in test_images\results\
call "%~dp0code\.venv\Scripts\activate.bat" 2>nul || (echo First run setup:  code\windows\setup.bat & pause & exit /b 1)
cd /d "%~dp0code"
set "IMGS="
for %%f in ("%~dp0test_images\*.jpg" "%~dp0test_images\*.jpeg" "%~dp0test_images\*.png" "%~dp0test_images\*.dcm") do call set IMGS=%%IMGS%% "%%~f"
if not defined IMGS ( echo Put X-rays in:  %~dp0test_images & pause & exit /b 1 )
python scripts\analyze.py %IMGS% --out "%~dp0test_images\results"
echo.
echo Done. Opening results...
start "" "%~dp0test_images\results"
pause
