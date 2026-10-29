@echo off
setlocal
chcp 65001 >nul
title Dataset tools - install all tools

rem ============================================================
rem  install.bat - runs the install.bat of every tool that needs one, one
rem  after the other. First the five tools the pipeline uses:
rem  remove_borders, reframe, jpeg_cleanup, face_masks and watermark; the
rem  GPU ones install torch from the PyTorch CUDA index. Then deduplicate,
rem  k2prep, ddg_images, classify, duck, pastvu and normalize. Each one creates the shared
rem  venv when it is missing and installs only its own dependencies into
rem  it, so running this file again installs what is missing and changes
rem  nothing that is there. duck needs git on PATH, and classify downloads
rem  its encoder. The tools' install.bat pause at their end; they get an
rem  empty input here, so they run through without a key press.
rem ============================================================

cd /d "%~dp0"

for %%T in (remove_borders reframe jpeg_cleanup face_masks watermark deduplicate k2prep ddg_images classify duck pastvu normalize) do (
    echo.
    echo ============================================================
    echo  %%T
    echo ============================================================
    call "%%T\install.bat" <nul
    rem The Xet transfer of huggingface_hub uses its own CDN, which some
    rem networks block; the plain HTTPS download often still works. Once
    rem set, it stays off for the remaining tools.
    if errorlevel 1 (
        echo.
        echo Retrying %%T without the Hugging Face Xet transfer...
        set "HF_HUB_DISABLE_XET=1"
        call "%%T\install.bat" <nul
    )
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
rem reframe and face_masks finish without the face model when it cannot be
rem downloaded; both find it in reframe\models.
if exist "reframe\models\face_yolov8m.pt" goto :done
echo.
echo [WARNING] face_yolov8m.pt could not be downloaded. reframe and face_masks do not run without it.
echo           Download it with a browser from
echo           https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt
echo           and put it into %~dp0reframe\models
:done
pause
exit /b 0
