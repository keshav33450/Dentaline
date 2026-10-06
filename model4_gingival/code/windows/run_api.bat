@echo off
REM Local API for the web app.  http://localhost:8004/health   (needs: pip install fastapi uvicorn python-multipart)
call "%~dp0..\..\..\model1_opg\code\.venv\Scripts\activate.bat" 2>nul || (echo First run setup once:  ..\model1_opg\code\windows\setup.bat & pause & exit /b 1)
cd /d "%~dp0.."
python -m uvicorn api.main:app --host 127.0.0.1 --port 8004
pause
