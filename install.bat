@echo off
setlocal
chcp 65001 >nul
title Dataset tools - install the pipeline tools

rem ============================================================
rem  install.bat - runs the install.bat of every tool the pipeline
rem  uses, one after the other: remove_borders, reframe, jpeg_cleanup,
rem  face_masks and watermark. Each one creates the shared venv when
rem  it is missing and installs only its own dependencies into it, so
rem  running this file again installs what is missing and changes
rem  nothing that is there. The tools' install.bat pause at their end;
rem  they get an empty input here, so they run through without a key press.
rem ============================================================

cd /d "%~dp0"

for %%T in (remove_borders reframe jpeg_cleanup face_masks watermark) do (
    echo.
    echo ============================================================
    echo  %%T
    echo ============================================================
    call "%%T\install.bat" <nul
    if errorlevel 1 (
        echo.
        echo [ERROR] The installation of %%T failed. Fix the error above and run install.bat again.
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo  All pipeline tools are installed. Run the pipeline with
echo  pipeline\run.bat --job job.json
echo ============================================================
pause
exit /b 0
