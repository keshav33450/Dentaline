@echo off
REM Start the DentAssist API on http://localhost:8000  (docs: http://localhost:8000/docs)
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m dentassist.paths
uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
