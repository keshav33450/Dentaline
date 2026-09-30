@echo off
REM Choose where DentAssist keeps data, weights, runs and results.
cd /d "%~dp0\.."
set "CUR=%CD%"
if exist project_location.txt set /p CUR=<project_location.txt
echo Current project location: %CUR%
set "LOC="
set /p LOC=New project folder (Enter = keep current): 
if "%LOC%"=="" set "LOC=%CUR%"
set "LOC=%LOC:"=%"
if not exist "%LOC%" mkdir "%LOC%"
> project_location.txt echo %LOC%
for %%d in (data\raw data\yolo runs weights results) do if not exist "%LOC%\%%d" mkdir "%LOC%\%%d"
echo.
echo Project location set to: %LOC%
echo   weights -^> %LOC%\weights   (put teeth.onnx / findings.onnx here)
echo   results -^> %LOC%\results
