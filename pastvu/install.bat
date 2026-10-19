@echo off
setlocal
chcp 65001 >nul
title PastVu - install

rem ============================================================
rem  install.bat - installs the dependencies of this tool into the shared
rem  venv (..\venv) and fetches turbojpeg.dll into bin\. The models and
rem  their dependencies come from the installers of remove_borders,
rem  watermark and classify, which the root install.bat runs before this
rem  one; run that first on a new machine. Running install.bat again
rem  installs what is missing.
rem ============================================================

cd /d "%~dp0"
set "VENV=..\venv"
set "VPY=%VENV%\Scripts\python.exe"

echo ============================================================
echo  PastVu - installation
echo ============================================================
echo.

if not exist "%VPY%" (
    echo [ERROR] The shared venv is missing. Run install.bat in the repository root first:
    echo         it installs remove_borders, watermark and classify, which this tool uses.
    pause
    exit /b 1
)

set "CONSTRAINT="
if exist "%VENV%\torch-constraints.txt" set "CONSTRAINT=--constraint %VENV%\torch-constraints.txt"

echo Installing the dependencies...
"%VPY%" -m pip install -r requirements.txt %CONSTRAINT%
if errorlevel 1 goto :fail

echo.
echo Fetching turbojpeg.dll ...
"%VPY%" fetch_turbojpeg.py
if errorlevel 1 goto :fail

echo.
echo Checking the tools this one uses...
"%VPY%" -c "import sys; sys.path[:0] = ['../remove_borders', '../watermark', '../classify']; import remove_borders, watermark, classify, torch, httpx, rich, turbojpeg; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available()); sys.exit(0 if classify.models_present() else 2)"
if errorlevel 2 (
    echo [ERROR] The classify encoder is missing. Run classify\install.bat.
    pause
    exit /b 1
)
if errorlevel 1 (
    echo [ERROR] remove_borders, watermark or classify is not installed. Run install.bat in the repository root.
    pause
    exit /b 1
)

echo.
echo Installation finished. Count the photos of a region with
echo   run.bat count --region Belarus
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
