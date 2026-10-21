@echo off
setlocal
chcp 65001 >nul
title Recaption

rem Usage: run.bat <dataset> [options]
rem        run.bat <dataset> --batch 1 --limit 50
rem        run.bat <dataset> --dry-run
rem Writes <stem>.short next to every <stem>.txt that has an image: one short
rem user-style prompt made from the long caption by the local LLM (KoboldCpp at
rem http://localhost:5001/v1/ unless --endpoint). A rerun continues where the
rem last one stopped. All arguments go to recaption.py unchanged; run.bat --help
rem lists the options.

rem Keep the window open at the end when started by double-click or drag-and-drop
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

rem The shared venv (install.bat puts httpx in it), else Python on PATH.
set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"
"%VPY%" --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and tick "Add Python to PATH".
    pause
    exit /b 1
)

rem Paths are not changed to the script folder, so relative folder arguments
rem resolve against the current directory.
"%VPY%" "%~dp0recaption.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
