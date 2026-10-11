@echo off
setlocal
chcp 65001 >nul
title Rebalance medium

rem Usage: run.bat <dataset folder> [options]          dry run: the plan only
rem        run.bat <dataset folder> --apply [options]  moves the images
rem        run.bat <dataset folder> --undo             moves them back
rem A share of the colour (--color, default 40) and black-and-white (--bw,
rem default 20) images moves to <folder>_media-rebalanced next to the dataset,
rem with the medium word removed from the caption. All arguments go to
rem rebalance_medium.py unchanged; run.bat --help lists the options.

rem Keep the window open at the end when started by double-click or drag-and-drop
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

rem The tool needs Python only: the shared venv when it exists, else Python on PATH.
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
"%VPY%" "%~dp0rebalance_medium.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
