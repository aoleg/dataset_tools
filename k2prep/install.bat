@echo off
setlocal
cd /d "%~dp0"
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Install complete. Use run.bat to run k2prep.
pause
