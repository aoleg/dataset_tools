@echo off
rem Move every image that has no caption (no .txt, .short, .caption or .json of the
rem same base name) into "single_files", with every other file of its name.
cd /d "%~dp0"
if not exist "single_files" mkdir "single_files"

set count=0
for %%F in (*.jpg *.jpeg *.png *.webp *.gif *.bmp) do (
    if exist "%%F" if not exist "%%~nF.txt" if not exist "%%~nF.short" if not exist "%%~nF.caption" if not exist "%%~nF.json" (
        rem "x.*" also finds "x.y.txt", whose base name is "x.y": compare the base names
        for %%S in ("%%~nF.*") do if /i "%%~nS"=="%%~nF" move /y "%%S" "single_files\" >nul
        set /a count+=1
    )
)

call echo Moved %%count%% image name(s) to single_files.
pause
