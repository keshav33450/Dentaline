@echo off
setlocal
set CAMP=%~dp0
set ROOT=%CAMP%..\
set PY=%ROOT%model1_occlusal\code\.venv\Scripts\python.exe

if not exist "%PY%" (
  echo Shared Python environment is missing.
  echo Create it from model1_occlusal\code and install model requirements plus camp_web\backend\requirements.txt.
  pause
  exit /b 1
)

if not exist "%CAMP%client\node_modules" (
  pushd "%CAMP%client"
  call npm install
  if errorlevel 1 (popd & pause & exit /b 1)
  popd
)

echo Building the DentalX Camp web client...
pushd "%CAMP%client"
call npm run build
if errorlevel 1 (popd & pause & exit /b 1)
popd

echo Starting the three independent model APIs...
start "DentalX Caries API :8001" cmd /k "cd /d %ROOT%model1_occlusal\code && %PY% -m uvicorn api.main:app --host 127.0.0.1 --port 8001"
start "DentalX Tooth API :8002" cmd /k "cd /d %ROOT%model2_tooth_type\code && %PY% -m uvicorn api.main:app --host 127.0.0.1 --port 8002"
start "DentalX Gingivitis API :8003" cmd /k "cd /d %ROOT%model3_gingival\code && %PY% -m uvicorn api.main:app --host 127.0.0.1 --port 8003"

echo Starting the DentalX Camp backend on port 8080...
start "DentalX Camp Backend :8080" cmd /k "cd /d %CAMP% && %PY% -m uvicorn backend.server:app --host 0.0.0.0 --port 8080"
timeout /t 5 /nobreak >nul
start "" http://localhost:8080
echo.
echo Keep the four service windows open while using DentalX Camp.
endlocal
