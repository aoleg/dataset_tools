@echo off
setlocal enabledelayedexpansion
rem ---------------------------------------------------------------------------
rem  cleanup.bat - run cleanup.py inside the local venv.
rem
rem      cleanup.bat <folder> <size> [--dim] [--dry-run]
rem
rem  Moves every image below <size>, and its .txt sidecar, out of <folder> and
rem  its first-level subfolders into _foldername next to it, keeping the folder
rem  structure. <size> is N (meaning NxN) or WxH, and the default test is area:
rem  1024 moves anything with fewer than 1048576 pixels.
rem
rem      cleanup.bat T:\somefolder 1024
rem
rem  moves T:\somefolder\small.jpg to T:\_somefolder\small.jpg and
rem  T:\somefolder\1\small.jpg to T:\_somefolder\1\small.jpg.
rem
rem  Subfolders whose name starts with an underscore are skipped, so _prep and
rem  the sidecar folder itself are never touched.
rem
rem  As in run.bat, the folder is whichever argument names an existing
rem  directory, and it is resolved before this script switches to its own
rem  directory so that a relative path means what you typed.
rem ---------------------------------------------------------------------------

rem  shift moves %0 too, so %~dp0 stops meaning this script once the
rem  parsing loop below has run. Take the directory first.
set "HERE=%~dp0"

set "FOLDER="
set "ARGS="

:parse
if "%~1"=="" goto parsed
if defined FOLDER goto addarg
if not exist "%~1\" goto addarg
set "FOLDER=%~f1"
shift
goto parse
:addarg
set "ARGS=!ARGS! %1"
shift
goto parse
:parsed

cd /d "%HERE%"
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

rem  A line with no existing folder on it goes straight through, so that
rem  cleanup.bat --help and a mistyped path both get argparse's own answer.
if defined FOLDER (
    python cleanup.py "%FOLDER%" %ARGS%
) else (
    python cleanup.py %ARGS%
)
exit /b %errorlevel%
