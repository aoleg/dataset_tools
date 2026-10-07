@echo off
setlocal
chcp 65001 >nul
title JPEG cleanup

rem Usage: run.bat <folder> [<folder> ...] --dry-run [options]
rem The images of the folder with a QF under --threshold are restored with
rem FBCNN in memory; the report and the contact sheets go to
rem <folder>\_backup\_jpeg_cleanup. No image is changed. All arguments go to
rem jpeg_cleanup.py unchanged; run.bat --help lists the options.

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

rem Paths are not changed to the script folder, so relative folder arguments
rem resolve against the current directory.
"%VPY%" "%~dp0jpeg_cleanup.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
