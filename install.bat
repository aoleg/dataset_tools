@echo off
setlocal
chcp 65001 >nul
title Dataset tools - install all tools

rem ============================================================
rem  install.bat - runs the install.bat of every tool that needs one, one
rem  after the other. First the five tools the pipeline uses:
rem  remove_borders, reframe, jpeg_cleanup, face_masks and watermark; the
rem  GPU ones install torch from the PyTorch CUDA index. Then deduplicate,
rem  k2prep, ddg_images, classify and duck. Each one creates the shared
rem  venv when it is missing and installs only its own dependencies into
rem  it, so running this file again installs what is missing and changes
rem  nothing that is there. duck needs git on PATH, and classify downloads
rem  its encoder. The tools' install.bat pause at their end; they get an
rem  empty input here, so they run through without a key press.
rem ============================================================

cd /d "%~dp0"

for %%T in (remove_borders reframe jpeg_cleanup face_masks watermark deduplicate k2prep ddg_images classify duck) do (
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
echo  All tools are installed. Run the pipeline with
echo  pipeline\run.bat --job job.json
echo ============================================================
pause
exit /b 0
