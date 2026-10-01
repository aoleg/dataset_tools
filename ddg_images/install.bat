@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Image Search Downloader - install

rem ============================================================
rem  install.bat - creates the venv and installs dependencies.
rem  Only the Bing and DuckDuckGo backends need a package (ddgs);
rem  the SearXNG backend uses the standard library only.
rem ============================================================

cd /d "%~dp0"

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
echo  Image Search Downloader - installation
echo ============================================================
echo.

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating the virtual environment...
    %PYTHON% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)

set "VPY=venv\Scripts\python.exe"

echo Upgrading pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Installing the dependencies...
"%VPY%" -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Self check...
"%VPY%" -c "import importlib.metadata as m, ddgs; print('ddgs', m.version('ddgs'))"
if errorlevel 1 goto :fail
"%VPY%" ddg_images.py --help >nul
if errorlevel 1 goto :fail

echo.
echo Installation finished. Download images with run.bat "query" -o ^<folder^> [options]
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
