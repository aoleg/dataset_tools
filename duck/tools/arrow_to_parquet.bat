@echo off
setlocal
chcp 65001 >nul
title duck - Arrow to Parquet

rem Usage: tools\arrow_to_parquet.bat --src <folder with .arrow shards> [--dst <folder>]
rem Converts the Arrow shards that Hugging Face ships for coyo-hd-11m-llavanext
rem into Parquet, by default into a "parquet" folder next to --src. Run once,
rem then point duck.conf at the output folder. All arguments go to
rem arrow_to_parquet.py unchanged; --help lists the options.

set "PYTHONUTF8=1"
set "VPY=%~dp0..\..\venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

"%VPY%" "%~dp0arrow_to_parquet.py" %*
exit /b %ERRORLEVEL%
