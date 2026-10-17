@echo off
setlocal
chcp 65001 >nul
title duck - download

rem Usage: download.bat <results.csv> [options]
rem Downloads the images of a CSV written by run.bat into the downloads folder
rem from duck.conf, one image plus one .txt caption each. All arguments go to
rem download.py unchanged; download.bat --help lists the options.

rem Keep the window open at the end when started by double-click or drag-and-drop
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "PYTHONUTF8=1"
set "NO_ALBUMENTATIONS_UPDATE=1"
set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

rem Paths are not changed to the script folder, so a relative CSV path or
rem --out resolves against the current directory.
"%VPY%" "%~dp0download.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
