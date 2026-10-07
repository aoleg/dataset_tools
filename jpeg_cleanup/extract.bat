@echo off
setlocal
chcp 65001 >nul
title JPEG cleanup - extract

rem Usage: extract.bat <folder> [<folder> ...] [options]
rem A folder dropped onto this file is measured; the images under each band
rem limit are copied into <folder>_jpeg_extract\<band> next to it. The dataset
rem is not changed. All arguments go to jpeg_cleanup.py --extract unchanged;
rem extract.bat --help lists the options.

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
"%VPY%" "%~dp0jpeg_cleanup.py" --extract %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
