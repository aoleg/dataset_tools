@echo off
setlocal
chcp 65001 >nul
title PastVu - install

rem ============================================================
rem  install.bat - installs the dependencies of this tool into the shared
rem  venv (..\venv), fetches turbojpeg.dll into bin\ and the models of the
rem  detect command into models\ (with the prompt embeddings of its
rem  vocabulary). The other models and their dependencies come from the
rem  installers of remove_borders, watermark and classify, which the root
rem  install.bat runs before this one; run that first on a new machine. It
rem  also makes pastvu.conf from pastvu.conf.example when there is none.
rem  Running install.bat again installs what is missing and leaves
rem  pastvu.conf alone.
rem
rem  install.bat replica installs only what the rebuild of a shared pick needs
rem  (run on a base made by share): its own venv when there is none, torch
rem  for LaMa, the packages of requirements-replica.txt, turbojpeg.dll and
rem  the LaMa weights. No classify, no detector, no conf.
rem ============================================================

cd /d "%~dp0"
set "VENV=..\venv"
set "VPY=%VENV%\Scripts\python.exe"

if /i "%~1"=="replica" goto :replica

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
echo Fetching the models of detect (YOLOE-26l, the face model) and making the prompt embeddings...
"%VPY%" detect.py --fetch-models
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

if not exist "pastvu.conf" (
    copy pastvu.conf.example pastvu.conf >nul
    echo.
    echo Created pastvu.conf from pastvu.conf.example. Edit it: set the base folder of the
    echo downloads and the classify samples folder under [paths].
)

echo.
echo Installation finished. Count the photos of a region with
echo   run.bat count --region belarus
pause
exit /b 0

:replica
echo ============================================================
echo  PastVu - installation for a shared pick
echo ============================================================
echo.
set "TORCH_SPEC=torch==2.13.0 torchvision"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu132"
set "PYTHON="
where python >nul 2>nul && set "PYTHON=python"
if not defined PYTHON (
    where py >nul 2>nul && set "PYTHON=py -3"
)
if not exist "%VPY%" (
    if not defined PYTHON (
        echo [ERROR] Python was not found. Install Python 3.13 from https://www.python.org/downloads/
        echo         and tick "Add python.exe to PATH".
        pause
        exit /b 1
    )
    %PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python 3.13 or newer is required. Install it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo Creating the virtual environment in %VENV% ...
    %PYTHON% -m venv "%VENV%"
    if errorlevel 1 goto :fail
)

"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Installing PyTorch (about 3 GB) ...
"%VPY%" -m pip install %TORCH_SPEC% --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail
"%VPY%" -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('torch', 'torchvision')))" > "%VENV%\torch-constraints.txt"
if errorlevel 1 goto :fail

echo.
echo Installing the other packages ...
"%VPY%" -m pip install -r requirements-replica.txt --constraint "%VENV%\torch-constraints.txt"
if errorlevel 1 goto :fail

echo.
echo Fetching turbojpeg.dll ...
"%VPY%" fetch_turbojpeg.py
if errorlevel 1 goto :fail

echo.
echo Fetching the LaMa weights ...
"%VPY%" -c "import sys; sys.path[:0] = ['../watermark']; import watermark; print(watermark.lama_path())"
if errorlevel 1 goto :fail

echo.
echo Checking ...
"%VPY%" -c "import sys; sys.path[:0] = ['.', '../remove_borders', '../watermark']; import torch, cv2, httpx, rich, turbojpeg, replica, share, encode; encode.turbo(); print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
if errorlevel 1 goto :fail

echo.
echo Installation finished. Put the pastvu.conf of the shared pick next to run.bat and start
echo   run.bat run
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
