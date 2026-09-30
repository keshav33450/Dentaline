@echo off
REM Drag an X-ray (jpg/png/dcm) onto this file -> saves <name>_result.jpg and <name>_result.json next to it
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python scripts\analyze.py %*
pause
