@echo off
setlocal
chcp 65001 >nul
title Pipeline

rem Usage: run.bat --job job.json [--dry-run] [--undo] [--threads N]
rem        run.bat --example          prints an example job file

rem Keep the window open at the end when started by double-click
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run ..\install.bat first.
    pause
    exit /b 1
)

rem Paths are not changed to the script folder, so relative paths in the
rem arguments resolve against the current directory.
"%VPY%" "%~dp0pipeline.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
