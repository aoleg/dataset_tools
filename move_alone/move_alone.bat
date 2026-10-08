@echo off
rem Move every image that has no .txt sidecar (same base name) into "single_files".
cd /d "%~dp0"
if not exist "single_files" mkdir "single_files"

set count=0
for %%F in (*.jpg *.jpeg *.png *.webp *.gif *.bmp) do (
    if not exist "%%~nF.txt" (
        move /y "%%F" "single_files\" >nul
        set /a count+=1
    )
)

call echo Moved %%count%% image(s) to single_files.
pause
