@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Reframe - install

rem ============================================================
rem  install.bat - creates the shared venv when missing, installs the
rem  dependencies of this tool into it.
rem  torch comes from the PyTorch CUDA index, never from PyPI:
rem  a PyPI torch on Windows is CPU only.
rem  The venv is shared by all tools of this repository and lives in
rem  ..\venv, next to the tool folders. Each tool's install.bat installs
rem  only its own dependencies, so a tool that needs no GPU never pulls
rem  torch in. Running install.bat again installs what is missing.
rem ============================================================

cd /d "%~dp0"
set "VENV=..\venv"
set "VPY=%VENV%\Scripts\python.exe"

set "TORCH_SPEC=torch==2.13.0 torchvision"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu132"

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
echo  Reframe - installation
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
echo Installing torch from %TORCH_INDEX% ...
"%VPY%" -m pip install %TORCH_SPEC% --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail

rem Pin the CUDA builds just installed so that pip cannot replace them with
rem the CPU builds from PyPI while installing the rest.
"%VPY%" -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('torch', 'torchvision')))" > "%VENV%\torch-constraints.txt"
if errorlevel 1 goto :fail

echo.
echo Installing the remaining dependencies...
"%VPY%" -m pip install -r requirements.txt --constraint "%VENV%\torch-constraints.txt"
if errorlevel 1 goto :fail

echo.
echo Self check...
"%VPY%" -c "import sys, torch, ultralytics, huggingface_hub; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU available:', torch.cuda.is_available()); sys.exit(0 if torch.version.cuda else 1)"
if errorlevel 1 (
    echo.
    echo [WARNING] torch is not a CUDA build. See the messages above.
    pause
    exit /b 1
)

echo.
echo Downloading the person, face and pose models...
"%VPY%" reframe.py --fetch-models
if errorlevel 1 goto :fail

echo.
echo Installation finished. Reframe photos with run.bat ^<folder^> [^<folder^> ...] [options]
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
