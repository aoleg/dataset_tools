@echo off
setlocal
chcp 65001 >nul
title Watermark

rem Usage: run.bat <folder> [<folder> ...] [options]
rem A folder dropped onto this file gets its watermarks painted out into
rem <folder>_watermark_removed next to it.

rem Keep the window open at the end when started by double-click or drag-and-drop
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

rem Paths are not changed to the script folder, so relative paths in the
rem arguments resolve against the current directory.
"%VPY%" "%~dp0watermark.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
