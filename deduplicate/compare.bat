@echo off
setlocal
chcp 65001 >nul
title Deduplicate: compare

rem Usage: compare.bat <folder> [options]
rem Shows which files dedup.py judged copies of which, from the plan of the
rem last run in <folder>\_duplicates: writes pairs.txt and one side-by-side
rem sheet per group to <folder>\_duplicates\compare. Nothing moves.
rem A folder dropped onto this file works too. All arguments go to compare.py
rem unchanged; compare.bat --help lists the options.

rem Keep the window open at the end when started by double-click or drag-and-drop
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "VPY=%~dp0venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

rem Paths are not changed to the script folder, so relative folder arguments
rem resolve against the current directory.
"%VPY%" "%~dp0compare.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
