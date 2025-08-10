#!/bin/bash

echo "🚀 Starting MioTracker GUI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first:"
    echo "  python3 setup.py"
    exit 1
fi

# Check if Python executable exists in venv
if [ ! -f "venv/bin/python" ]; then
    echo "❌ Virtual environment appears to be corrupted!"
    echo "Please recreate it by running:"
    echo "  python3 setup.py"
    exit 1
fi

# Check if miotracker.py exists
if [ ! -f "miotracker.py" ]; then
    echo "❌ miotracker.py not found in current directory!"
    echo "Make sure you're running this from the miotracker-gui directory"
    exit 1
fi

echo "📱 Launching MioTracker GUI..."

# Activate virtual environment and run the application
source venv/bin/activate
python miotracker.py

if [ $? -eq 0 ]; then
    echo "👋 Application closed successfully"
else
    echo "❌ Application exited with error code $?"
fi 