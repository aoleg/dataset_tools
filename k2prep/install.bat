@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title k2prep - install

rem ============================================================
rem  install.bat - creates the shared venv when missing, installs the
rem  dependencies of this tool into it.
rem  Pillow, numpy and tqdm; no GPU and no torch needed.
rem  The venv is shared by all tools of this repository and lives in
rem  ..\venv, next to the tool folders. Each tool's install.bat installs
rem  only its own dependencies, so a tool that needs no GPU never pulls
rem  torch in. Running install.bat again installs what is missing.
rem ============================================================

cd /d "%~dp0"
set "VENV=..\venv"
set "VPY=%VENV%\Scripts\python.exe"

set "PYTHON="
where python >nul 2>nul && set "PYTHON=python"
if not defined PYTHON (
    where py >nul 2>nul && set "PYTHON=py -3.12"
)
if not defined PYTHON (
    echo [ERROR] Python was not found. Install Python 3.12 from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH".
    pause
    exit /b 1
)

echo ============================================================
echo  k2prep - installation
echo ============================================================
echo.

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    pause
    exit /b 1
)

if not exist "%VPY%" (
    echo Creating the shared virtual environment in %VENV% ...
    %PYTHON% -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)

echo Upgrading pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Installing the dependencies...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Self check...
"%VPY%" -c "import numpy, PIL, tqdm; print('numpy', numpy.__version__, '| Pillow', PIL.__version__, '| tqdm', tqdm.__version__)"
if errorlevel 1 goto :fail
"%VPY%" k2prep.py --help >nul
if errorlevel 1 goto :fail
"%VPY%" cleanup.py --help >nul
if errorlevel 1 goto :fail

echo.
echo Installation finished. Prepare a dataset with run.bat ^<folder^> [options]
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
