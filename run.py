#!/usr/bin/env python3
"""
MioTracker GUI Run Script
Activates virtual environment and launches the application
"""

import os
import sys
import subprocess
import platform

def main():
    print("🚀 Starting MioTracker GUI...")
    
    # Check if virtual environment exists
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found!")
        print("Please run setup first:")
        print("  python setup.py")
        sys.exit(1)
    
    # Determine platform-specific commands
    system = platform.system()
    if system == "Windows":
        activate_script = os.path.join(venv_path, "Scripts", "activate")
        python_exe = os.path.join(venv_path, "Scripts", "python.exe")
    else:  # macOS/Linux
        activate_script = os.path.join(venv_path, "bin", "activate")
        python_exe = os.path.join(venv_path, "bin", "python")
    
    # Check if Python executable exists in venv
    if not os.path.exists(python_exe):
        print("❌ Virtual environment appears to be corrupted!")
        print("Please recreate it by running:")
        print("  python setup.py")
        sys.exit(1)
    
    # Launch the application
    try:
        print("📱 Launching MioTracker GUI...")
        subprocess.run([python_exe, "miotracker.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
    except FileNotFoundError:
        print("❌ miotracker.py not found in current directory!")
        print("Make sure you're running this from the miotracker-gui directory")
        sys.exit(1)

if __name__ == "__main__":
    main() 