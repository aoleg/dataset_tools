@echo off
setlocal
chcp 65001 >nul
title Flatten dataset

rem Usage: run.bat <folder> [<folder> ...] [options]
rem A folder dropped onto this file is flattened: the images and captions of all
rem its subfolders move into it under new names, and the plan is written to
rem <folder>\_flatten_dataset\manifest.json for --undo. All arguments go to
rem flatten_dataset.py unchanged; run.bat --help lists the options.

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
"%VPY%" "%~dp0flatten_dataset.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
