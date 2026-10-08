@echo off
setlocal disabledelayedexpansion
rem ---------------------------------------------------------------------------
rem  cleanup.bat - run cleanup.py inside the shared venv.
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
rem  Every argument goes to cleanup.py exactly as typed, for the reason given
rem  in run.bat: parsing it here deleted every "!" in a path.
rem ---------------------------------------------------------------------------

if not exist "%~dp0..\venv\Scripts\python.exe" (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
"%~dp0..\venv\Scripts\python.exe" "%~dp0cleanup.py" %*
exit /b %errorlevel%
