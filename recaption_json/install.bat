@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Recaption - install

rem ============================================================
rem  install.bat - creates the shared venv when missing, installs the
rem  dependencies of this tool into it.
rem  The tool needs one package, httpx, for the LLM endpoint.
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
    where py >nul 2>nul && set "PYTHON=py -3"
)
if not defined PYTHON (
    echo [ERROR] Python was not found. Install Python 3.13 from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH".
    pause
    exit /b 1
)

echo ============================================================
echo  Recaption - installation
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
"%VPY%" -c "import importlib.metadata as m, httpx; print('httpx', m.version('httpx'))"
if errorlevel 1 goto :fail
"%VPY%" recaption.py --help >nul
if errorlevel 1 goto :fail

echo.
echo Installation finished. Write the short prompts with run.bat ^<dataset^> [options]
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
