@echo off
setlocal enabledelayedexpansion
rem ---------------------------------------------------------------------------
rem  run.bat - run k2prep.py inside the local venv.
rem
rem      run.bat <folder> [options]
rem      run.bat -R <folder> [options]
rem
rem  -R runs k2prep once for <folder>, then once more for each first-level
rem  subfolder, so every folder gets its OWN _prep, reports and dataset.toml -
rem  independent datasets, trained separately. Nothing below the first level is
rem  visited. This is not k2prep's --recursive, which is one run over the whole
rem  tree producing ONE shared _prep and ONE dataset.toml with a [[datasets]]
rem  block per subfolder - several concepts trained together. Pick one; passing
rem  --recursive together with -R runs the tree mode once per subfolder, which
rem  is rarely what anyone means.
rem
rem  Subfolders whose name starts with an underscore are skipped: that is
rem  k2prep's own _prep output and cleanup.bat's _foldername sidecars. The
rem  score* and quality* folders that --sort produces are NOT skipped, since
rem  building a dataset out of one triaged tier is a real thing to want.
rem
rem  Every other option is passed through to k2prep unchanged, once per folder.
rem  <folder> is whichever argument names an existing directory, so it can sit
rem  anywhere on the line and is never confused with an option's value
rem  (--filter lanczos, --vl "sharpness, composition"). The scan runs before
rem  this script switches to its own directory, so a relative path resolves
rem  against the directory you called it from; k2prep is handed the absolute
rem  path.
rem
rem  A folder that fails does not stop the sweep. The failures are listed at the
rem  end and the exit code is non-zero.
rem
rem  One limit: an argument containing an exclamation mark is mangled by the
rem  delayed expansion this parsing needs. Only --vl text can realistically
rem  contain one.
rem ---------------------------------------------------------------------------

rem  shift moves %0 too, so %~dp0 stops meaning this script once the
rem  parsing loop below has run. Take the directory first.
set "HERE=%~dp0"

set "RECURSE="
set "FOLDER="
set "ARGS="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="-R" (
    set "RECURSE=1"
    shift
    goto parse
)
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

if defined RECURSE if not defined FOLDER (
    echo -R needs a folder, and none of the arguments named one that exists.
    exit /b 2
)

cd /d "%HERE%"
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

rem  No -R: one call, exactly as before. A line with no folder on it at all
rem  (run.bat --version, run.bat --help) still goes straight to k2prep so that
rem  argparse gets to answer it.
if not defined RECURSE (
    if defined FOLDER (
        python k2prep.py "%FOLDER%" %ARGS%
    ) else (
        python k2prep.py %ARGS%
    )
    exit /b %errorlevel%
)

set /a NFAIL=0
call :run "%FOLDER%"
for /d %%d in ("%FOLDER%\*") do (
    set "NAME=%%~nxd"
    if "!NAME:~0,1!"=="_" (
        echo.
        echo === skipping %%~fd
    ) else (
        call :run "%%~fd"
    )
)

echo.
if %NFAIL%==0 (
    echo All folders finished.
    exit /b 0
)
echo %NFAIL% folder^(s^) failed:
for /l %%i in (1,1,%NFAIL%) do echo     !FAIL_%%i!
exit /b 1

:run
echo.
echo === %~1
python k2prep.py "%~1" %ARGS%
if errorlevel 1 (
    set /a NFAIL+=1
    set "FAIL_!NFAIL!=%~1"
)
goto :eof
