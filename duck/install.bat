@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title duck - install

rem ============================================================
rem  install.bat - creates the shared venv when missing, installs the
rem  dependencies of this tool into it.
rem  DuckDB, img2dataset (from GitHub, which needs git on PATH; see
rem  requirements.txt for why not PyPI) and huggingface_hub for the
rem  dataset downloads. No GPU and no torch needed.
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
    where py >nul 2>nul && set "PYTHON=py -3.13"
)
if not defined PYTHON (
    echo [ERROR] Python was not found. Install Python 3.13 from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH".
    pause
    exit /b 1
)

echo ============================================================
echo  duck - installation
echo ============================================================
echo.

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    pause
    exit /b 1
)

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] git was not found on PATH. It is needed to install img2dataset from GitHub.
    echo         Install Git for Windows from https://git-scm.com/ and run install.bat again.
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
rem The GPU tools pin their CUDA torch in torch-constraints.txt; honour it when
rem it exists so that nothing here can replace that build.
if exist "%VENV%\torch-constraints.txt" (
    "%VPY%" -m pip install -r requirements.txt --constraint "%VENV%\torch-constraints.txt"
) else (
    "%VPY%" -m pip install -r requirements.txt
)
if errorlevel 1 goto :fail

echo.
echo Self check...
set "NO_ALBUMENTATIONS_UPDATE=1"
"%VPY%" -c "import importlib.metadata as m, duckdb, img2dataset, huggingface_hub; print('duckdb', duckdb.__version__, '| img2dataset', m.version('img2dataset'), '| huggingface_hub', m.version('huggingface_hub'))"
if errorlevel 1 goto :fail
"%VPY%" duck.py --list-datasets >nul
if errorlevel 1 goto :fail

if not exist "duck.conf" (
    copy duck.conf.example duck.conf >nul
    echo.
    echo Created duck.conf from duck.conf.example. Edit it and set the dataset folders.
)

echo.
echo Installation finished. Search a dataset with run.bat --dataset ^<name^> [options]
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
