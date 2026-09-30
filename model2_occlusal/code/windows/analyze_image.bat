@echo off
REM Drag a photo onto this file to analyse it.
call "%~dp0..\..\..\model1_opg\code\.venv\Scripts\activate.bat" 2>nul || (echo First run setup once:  ..\model1_opg\code\windows\setup.bat & pause & exit /b 1)
cd /d "%~dp0.."
if not defined ROBOFLOW_API_KEY set ROBOFLOW_API_KEY=rf_iwIiyh43vKbF03xfqxNwN0x3CO42
set IMG=%~1
if "%IMG%"=="" set /p IMG=Path to photo: 
python scripts\analyze.py "%IMG%" --out "%~dp0..\..\test_images\results"
echo.
echo Result in test_images\results
pause
