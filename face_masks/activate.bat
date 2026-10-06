@echo off
rem Activates the project venv in the current cmd window.
rem Started by double-click, it opens a new cmd window with the venv active.

if not exist "%~dp0..\venv\Scripts\activate.bat" (
    echo venv not found. Run install.bat first.
    exit /b 1
)

echo %CMDCMDLINE% | find /i "%~nx0" >nul && (
    cmd /k ""%~dp0..\venv\Scripts\activate.bat""
    exit /b
)

call "%~dp0..\venv\Scripts\activate.bat"
