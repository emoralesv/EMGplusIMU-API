#!/usr/bin/env python3
"""
MioTracker GUI Setup Script
Automates virtual environment creation and dependency installation
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"⚙️  {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 MioTracker GUI Setup")
    print("=" * 30)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or newer is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Determine platform-specific commands
    system = platform.system()
    if system == "Windows":
        venv_activate = "venv\\Scripts\\activate"
        python_cmd = "python"
        activate_cmd = f"{venv_activate} && "
    else:  # macOS/Linux
        venv_activate = "source venv/bin/activate"
        python_cmd = "python3"
        activate_cmd = f"{venv_activate} && "
    
    print(f"🖥️  Platform: {system}")
    
    # Create virtual environment
    if not run_command(f"{python_cmd} -m venv venv", "Creating virtual environment"):
        sys.exit(1)
    
    # Install dependencies
    if not run_command(f"{activate_cmd}pip install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    if not run_command(f"{activate_cmd}pip install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"1. Activate virtual environment:")
    print(f"   {venv_activate}")
    print("2. Run the application:")
    print("   python miotracker.py")
    print("\n📖 For detailed usage instructions, see README.md")
    print("🚀 For quick start, see QUICKSTART.md")

if __name__ == "__main__":
    main() 