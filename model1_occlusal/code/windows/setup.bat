@echo off
REM Shared Python env for all three models (lives here).
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Creating shared venv...
  python -m venv .venv || (echo Need Python 3.10-3.12 on PATH & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r "%~dp0..\..\..\requirements.txt"
echo.
echo Shared env ready: %CD%\.venv
echo Then use ANALYZE_TEST_IMAGES.bat or windows\run_api.bat
pause
