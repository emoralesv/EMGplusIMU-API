# MioTracker GUI - Quick Start Guide

## 🚀 Get Running in 5 Minutes

### 1. Prerequisites
- Python 3.8+ installed
- MioTracker device (optional - can use test data)

### 2. Installation

**Option A: Automated Setup (Recommended)**
```bash
# Clone and navigate to directory
git clone <your-repository-url>
cd miotracker-gui

# Run automated setup
python3 setup.py  # macOS/Linux
# OR
python setup.py   # Windows
```

**Option B: Manual Setup**
```bash
# Clone and navigate to directory
git clone <your-repository-url>
cd miotracker-gui

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Application

**Option A: Using Run Scripts**
```bash
# macOS/Linux:
./run.sh

# Windows:
run.bat

# Cross-platform Python script:
python3 run.py
```

**Option B: Manual Run**
```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Run application
python miotracker.py
```

### 4. Test Without Hardware
1. Right-click in the raw data area
2. Select "Generate Test Data"
3. Right-click again → "Export All Data to CSV"
4. Choose save location

### 5. Connect Real Device

**USB Connection:**
1. Connect MioTracker via USB
2. Click "Scan" → Select port → "USB Connect"

**WebSocket Connection:**
1. Ensure device on same network
2. Click "Connect" (uses `ws://miotracker.local/start`)

### 6. Export Your Data
- Right-click raw data area
- Choose export option
- Data saved with timestamps in CSV format

## 📊 Data Access
```python
# Get data as pandas DataFrames
imu_df = main_window.getIMUDataFrame()
emg_df = main_window.getEMGDataFrame()

print(f"Collected {len(imu_df)} IMU samples and {len(emg_df)} EMG samples")
```

## 🛠️ Available Scripts
- `setup.py` - Automated environment setup
- `run.py` - Cross-platform run script
- `run.sh` - macOS/Linux run script
- `run.bat` - Windows run script

## 🆘 Problems?
- **No data?** → Generate test data first
- **Connection issues?** → Check USB/network connections  
- **Import errors?** → Ensure virtual environment is activated
- **Setup issues?** → Try manual installation method

**Need help?** See full [README.md](README.md) for detailed instructions. 