@echo off
setlocal
chcp 65001 >nul
title Judge - install

rem ============================================================
rem  install.bat - judge needs what classify installs: torch, transformers,
rem  numpy, Pillow and classify's SigLIP 2 snapshot in classify\models.
rem  It installs nothing of its own (requirements.txt is empty) and never
rem  downloads a second copy of the model: it checks that classify's
rem  snapshot holds both the vision and the text tower and, when the
rem  snapshot is missing, downloads it through classify into classify\models.
rem  The shared venv (..\venv) comes from the root install.bat.
rem ============================================================

cd /d "%~dp0"
set "VENV=..\venv"
set "VPY=%VENV%\Scripts\python.exe"

echo ============================================================
echo  Judge - installation
echo ============================================================
echo.

if not exist "%VPY%" (
    echo [ERROR] The shared venv is missing. Run install.bat in the repository root first.
    pause
    exit /b 1
)

set "CONSTRAINT="
if exist "%VENV%\torch-constraints.txt" set "CONSTRAINT=--constraint %VENV%\torch-constraints.txt"

echo Installing the dependencies...
"%VPY%" -m pip install -r requirements.txt %CONSTRAINT%
if errorlevel 1 goto :fail

echo.
echo Checking classify's dependencies...
"%VPY%" -c "import torch, transformers, numpy, PIL, safetensors; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| transformers', transformers.__version__)"
if errorlevel 1 (
    echo.
    echo [ERROR] torch or transformers is missing. Run classify\install.bat first, then this one again.
    pause
    exit /b 1
)

echo.
echo Checking the SigLIP 2 model in classify\models ...
"%VPY%" judge.py --check-models
if errorlevel 1 goto :fail

echo.
echo Installation finished. Check it with
echo   run.bat --help
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
