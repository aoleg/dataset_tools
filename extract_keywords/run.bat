@echo off
setlocal
chcp 65001 >nul
title Extract keywords

rem Usage: run.bat <keyword> [<keyword> ...] --dataset <folder> [options]
rem        run.bat <keyword> AND <keyword> --dataset <folder>
rem        run.bat <keyword> ... --dataset <folder> --undo
rem The images whose captions contain any of the keywords (all of them when
rem joined by AND) move with their captions to <folder>_<first keyword> next to
rem the dataset, in the same subfolders. Put a phrase in quotes. All arguments go
rem to extract_keywords.py unchanged; run.bat --help lists the options.

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
"%VPY%" "%~dp0extract_keywords.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
