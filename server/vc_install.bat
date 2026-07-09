@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo Voice Changer Server Installation Script (RTX 5080 CUDA)
echo ===============================================
echo.

REM Check if we're in the server directory
if not exist "main.py" (
    echo Error: This script must be run from the server directory.
    echo Please navigate to the server directory and run the script again.
    pause
    exit /b 1
)
if not exist "requirements-common.txt" (
    echo Error: This script must be run from the server directory.
    pause
    exit /b 1
)

REM Function to check if Python is available
:check_python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.12 and try again.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% lss 3 (
    echo Error: Python 3.12 is recommended. Found: %PYTHON_VERSION%
    pause
    exit /b 1
)
if %PYTHON_MAJOR% equ 3 if %PYTHON_MINOR% lss 10 (
    echo Error: Python 3.10 or higher is required. Found: %PYTHON_VERSION%
    pause
    exit /b 1
)

echo Found Python: %PYTHON_VERSION%
set BACKEND=cuda-rtx5080
set REQUIREMENTS_FILE=requirements-cuda-5080.txt

REM Function to create virtual environment
:create_venv
echo.
echo Selected backend: %BACKEND%
echo.
echo Creating virtual environment...

if exist "venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment created and activated
goto install_requirements

REM Function to install requirements
:install_requirements
echo.
echo Installing requirements...

REM Upgrade pip first
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Warning: Failed to upgrade pip
)

REM Install common requirements
echo Installing common requirements...
pip install -r requirements-common.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install common requirements
    pause
    exit /b 1
)

REM Install backend-specific requirements
if exist "%REQUIREMENTS_FILE%" (
    echo Installing %BACKEND%-specific requirements (%REQUIREMENTS_FILE%)...
    pip install -r "%REQUIREMENTS_FILE%"
    if %errorlevel% neq 0 (
        echo Error: Failed to install %BACKEND%-specific requirements
        pause
        exit /b 1
    )
) else (
    echo Error: %REQUIREMENTS_FILE% not found.
    pause
    exit /b 1
)

echo Requirements installed successfully
goto show_completion

REM Function to show completion message
:show_completion
echo.
echo ===============================================
echo Installation completed successfully!
echo ===============================================
echo.
echo To start the voice changer server:
echo.
echo Run .\vc_startup.bat
echo.
echo Backend: %BACKEND% (RTX 5080 Optimized)
echo.
echo Press any key to exit...
pause >nul
exit /b 0