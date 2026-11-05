@echo off
setlocal
chcp 65001 >nul
title Judge

rem Usage: run.bat <folder> --condition "<text>" [--images] [...]   (run.bat --help lists everything)
rem        run.bat <folder> --undo
rem All arguments go to judge.py unchanged.

set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat in the repository root first.
    pause
    exit /b 1
)

rem The SigLIP 2 snapshot was downloaded by classify's installer; never touch the network for it at run time.
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "YOLO_OFFLINE=1"

"%VPY%" "%~dp0judge.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
