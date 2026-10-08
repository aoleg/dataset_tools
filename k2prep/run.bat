@echo off
setlocal disabledelayedexpansion
rem ---------------------------------------------------------------------------
rem  run.bat - run k2prep.py inside the local venv.
rem
rem      run.bat <folder> [options]
rem      run.bat <folder> -R [options]
rem
rem  Every argument goes to k2prep.py exactly as typed. This script does not
rem  parse the command line: cmd mangles paths while parsing them, and the
rem  delayed expansion an argument loop needs deletes every "!" - a folder
rem  named "!!!_photos" arrived as "_photos". -R, the per-subfolder sweep that
rem  was the reason for parsing here, is a k2prep option now.
rem
rem  Python is called by its full path instead of after a cd, so a relative
rem  <folder> means what it meant where you typed it.
rem ---------------------------------------------------------------------------

if not exist "%~dp0venv\Scripts\python.exe" (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
"%~dp0venv\Scripts\python.exe" "%~dp0k2prep.py" %*
exit /b %errorlevel%
