@echo off
setlocal
chcp 65001 >nul
title Normalize

rem Usage: run.bat <folder> <pixels> [-o <folder>] [--format original [Q]^|jpg [Q]^|png] [--dry-run] [--yes]
rem Every image under <folder>, in subfolders of any depth, that is larger than
rem <pixels> x <pixels> in area is downscaled to that area, in place, and saved
rem as .jpg quality 97 by default; an image in another format is re-encoded
rem and its original deleted. The tool asks before it overwrites anything.
rem With -o the images and their captions go to the output folder instead, and
rem the source folder is not changed.
rem All arguments go to normalize.py unchanged; run.bat --help lists the options.

rem Keep the window open at the end when started by double-click or drag-and-drop
rem (set NOPAUSE=1 to skip).
set "PAUSE_AT_END="
if not defined NOPAUSE echo %CMDCMDLINE% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

set "VPY=%~dp0..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The shared venv was not found. Run install.bat in this folder first.
    pause
    exit /b 1
)

rem Paths are not changed to the script folder, so a relative folder argument
rem resolves against the current directory. UTF-8 mode, so that Cyrillic and
rem other non-ASCII names print correctly in a redirected log too.
"%VPY%" -X utf8 "%~dp0normalize.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The program exited with an error.
    if not defined NOPAUSE set "PAUSE_AT_END=1"
)

if defined PAUSE_AT_END pause
exit /b %RC%
