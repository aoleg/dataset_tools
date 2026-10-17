@echo off
setlocal
chcp 65001 >nul
title duck

rem Usage: run.bat --dataset <name> [options]     search one dataset, write a CSV
rem        run.bat --dataset <name> --help        list that dataset's options
rem        run.bat --list-datasets                list the datasets
rem All arguments go to duck.py unchanged. The dataset folders come from
rem duck.conf next to this file.

rem Keep the window open at the end when started by double-click
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "PYTHONUTF8=1"
set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

rem Paths are not changed to the script folder, so a relative --out or --dir
rem resolves against the current directory.
"%VPY%" "%~dp0duck.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
