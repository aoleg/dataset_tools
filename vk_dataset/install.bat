@echo off
setlocal
chcp 65001 >nul
title VK dataset - install

rem ============================================================
rem  install.bat - installs the dependencies of this tool (requests) into
rem  the shared venv (..\venv), which the root install.bat creates; run
rem  that first on a new machine. It also makes vkds.conf from
rem  vkds.conf.example when there is none. Running install.bat again
rem  installs what is missing and leaves vkds.conf alone.
rem ============================================================

cd /d "%~dp0"
set "VENV=..\venv"
set "VPY=%VENV%\Scripts\python.exe"

echo ============================================================
echo  VK dataset - installation
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
echo Checking the dependencies...
"%VPY%" -c "import requests; print('requests', requests.__version__)"
if errorlevel 1 goto :fail

if not exist "vkds.conf" (
    copy vkds.conf.example vkds.conf >nul
    echo.
    echo Created vkds.conf from vkds.conf.example. Edit it: set the datasets root base
    echo under [paths], and put the token file vk_token.txt next to vkds.conf.
)

echo.
echo Installation finished. Check it with
echo   run.bat --version
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. Fix the error above and run install.bat again.
pause
exit /b 1
