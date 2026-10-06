@echo off
setlocal
chcp 65001 >nul
title Deduplicate: review

rem Usage: review.bat <folder>
rem Opens the groups of the last dedup.py run full screen, one group per
rem screen, to check them by eye and change the kept copy. A folder dropped
rem onto this file works too. review.bat --help lists the keys.

set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

"%VPY%" "%~dp0review.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
