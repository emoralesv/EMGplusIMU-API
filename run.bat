@echo off
echo 🚀 Starting MioTracker GUI...

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run setup first:
    echo   python setup.py
    pause
    exit /b 1
)

REM Check if Python executable exists in venv
if not exist "venv\Scripts\python.exe" (
    echo ❌ Virtual environment appears to be corrupted!
    echo Please recreate it by running:
    echo   python setup.py
    pause
    exit /b 1
)

REM Check if miotracker.py exists
if not exist "miotracker.py" (
    echo ❌ miotracker.py not found in current directory!
    echo Make sure you're running this from the miotracker-gui directory
    pause
    exit /b 1
)

echo 📱 Launching MioTracker GUI...
venv\Scripts\python.exe miotracker.py

if errorlevel 1 (
    echo ❌ Failed to start application
    pause
    exit /b 1
)

echo 👋 Application closed
pause 