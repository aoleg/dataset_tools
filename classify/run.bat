@echo off
setlocal
chcp 65001 >nul
title Classify

rem Usage: run.bat --dataset <folder> [--dataset <folder> ...] --samples <folder> -o <folder> [options]
rem        run.bat -o <folder> --undo
rem All arguments go to classify.py unchanged; run.bat --help lists the options.

rem Keep the window open at the end when started by double-click
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

rem The encoder was downloaded by install.bat; never touch the network at run time.
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

rem Paths are not changed to the script folder, so relative folder arguments
rem resolve against the current directory.
"%VPY%" "%~dp0classify.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
