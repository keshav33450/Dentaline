@echo off
REM Local API for the web app.  http://localhost:8003/health   (needs: pip install fastapi uvicorn python-multipart)
call "%~dp0..\..\..\model1_opg\code\.venv\Scripts\activate.bat" 2>nul || (echo First run setup once:  ..\model1_opg\code\windows\setup.bat & pause & exit /b 1)
cd /d "%~dp0.."
if not defined ROBOFLOW_API_KEY set ROBOFLOW_API_KEY=rf_iwIiyh43vKbF03xfqxNwN0x3CO42
python -m uvicorn api.main:app --host 127.0.0.1 --port 8003
pause
