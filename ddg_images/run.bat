@echo off
setlocal
chcp 65001 >nul
title Image Search Downloader

rem Usage: run.bat "query one" "query two" -o <folder> [options]
rem        run.bat -f queries.txt -o <folder> --backend searxng
rem All arguments go to ddg_images.py unchanged; run.bat --help lists them.

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

rem Paths are not changed to the script folder, so relative paths in the
rem arguments (-f, -o) resolve against the current directory.
"%VPY%" "%~dp0ddg_images.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
