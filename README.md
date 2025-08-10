# MioTracker GUI

A comprehensive graphical user interface for the MioTracker device, providing real-time data visualization, data storage with timestamps, CSV export functionality, and Wi-Fi provisioning capabilities.

## Features

- **Real-time Data Visualization**: Live plotting of EMG and IMU sensor data
- **Data Storage**: Automatic storage of sensor data with pandas DatetimeIndex timestamps
- **CSV Export**: Export IMU and EMG data to CSV files for analysis
- **Dual Connectivity**: Support for both USB/Serial and WebSocket connections
- **Wi-Fi Provisioning**: Configure device Wi-Fi settings through the GUI
- **Test Data Generation**: Built-in test data generation for debugging and testing

## Prerequisites

- Python 3.8 or newer
- An internet connection for downloading dependencies
- MioTracker device (for real data collection)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd miotracker-gui
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a Python virtual environment to manage dependencies:

**On macOS/Linux:**
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

After activation, your terminal prompt should be prefixed with `(venv)`.

### 3. Install Dependencies

**Method 1: Automated Setup (Easiest)**
```bash
python3 setup.py  # macOS/Linux
# OR
python setup.py   # Windows
```

**Method 2: Using requirements.txt**
```bash
pip install -r requirements.txt
```

**Method 3: Manual installation**
```bash
pip install PyQt5 pyqtgraph numpy pandas pyserial websocket-client matplotlib scipy
```

## Running the Application

**Method 1: Using Run Scripts (Easiest)**
```bash
# macOS/Linux:
./run.sh

# Windows:
run.bat

# Cross-platform Python script:
python3 run.py
```

**Method 2: Manual Run**
```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Run application
python miotracker.py
```

The application window will appear with the MioTracker GUI interface.

## Usage Guide

### Data Collection

#### USB/Serial Connection

1. Connect the MioTracker device via USB
2. Click **"Scan"** to find available serial ports
3. Select the correct port from the dropdown menu
4. Click **"USB Connect"** to start receiving data
5. The graphs will display live EMG and IMU signals
6. Click **"USB Disconnect"** to stop the connection

#### WebSocket Connection

1. Ensure the MioTracker device is connected to the same network
2. Click **"Connect"** for WebSocket connection
3. Data will be received and displayed in real-time
4. Click **"Disconnect"** to stop the WebSocket connection

### Data Storage and Export

The application automatically stores all received sensor data with precise timestamps using pandas DatetimeIndex.

#### Accessing Data as Variables

```python
# Get data as pandas DataFrames
imu_df = main_window.getIMUDataFrame()
emg_df = main_window.getEMGDataFrame()

# Check data
print(f"IMU data shape: {imu_df.shape}")
print(f"EMG data shape: {emg_df.shape}")

# Get latest samples
latest_imu = main_window.getLatestIMUData()
latest_emg = main_window.getLatestEMGData()
```

#### Export to CSV

**Using the Context Menu:**
1. Right-click on the raw data text area
2. Select export options:
   - "Export IMU Data to CSV"
   - "Export EMG Data to CSV"
   - "Export All Data to CSV"
3. Choose save location in the file dialog

**Programmatically:**
```python
# Export individual datasets
main_window.exportIMUData("my_imu_data.csv")
main_window.exportEMGData("my_emg_data.csv")

# Export both with timestamp-based filenames
main_window.exportAllData("experiment_001")
```

#### CSV File Format

**IMU Data (IMU_Data_YYYYMMDD_HHMMSS.csv):**
```
Timestamp,Accel_X_g,Accel_Y_g,Accel_Z_g,Gyro_X_dps,Gyro_Y_dps,Gyro_Z_dps
2024-01-15 14:30:25.123456,0.123,-0.456,9.801,12.34,-5.67,2.13
...
```

**EMG Data (EMG_Data_YYYYMMDD_HHMMSS.csv):**
```
Timestamp,Channel_0_mV,Channel_1_mV
2024-01-15 14:30:25.123456,45.2,67.8
2024-01-15 14:30:25.127456,44.1,68.2
...
```

### Data Management

**Right-click Context Menu Options:**
- **Clear Raw Data**: Clear the raw data display
- **Export Data**: Various export options
- **Clear Sensor Data**: Clear stored IMU/EMG data
- **Show Data Summary**: Display data statistics
- **Generate Test Data**: Create sample data for testing

### Test Data Generation

For testing and debugging without hardware:

1. Right-click on raw data area → "Generate Test Data"
2. Confirm the dialog to generate sample IMU and EMG data
3. Test export functionality with the generated data

### Wi-Fi Provisioning

Configure the MioTracker device to connect to your Wi-Fi network:

1. **Connect to Device's Access Point:**
   - When in provisioning mode, the device creates a Wi-Fi network (SSID: `PROV_XXXXXX`)
   - Connect your computer to this network

2. **Enter Wi-Fi Credentials:**
   - In the application, enter the target network **SSID** and **Password**

3. **Start Provisioning:**
   - Click **"Start Provisioning"**
   - Wait for success/failure message
   - Device will connect to the specified network

4. **Reset Provisioning:**
   - Click **"Reset Provisioning"** to clear stored Wi-Fi settings

## Data Analysis

The exported CSV files use pandas DatetimeIndex, making them ideal for time series analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load exported data
imu_data = pd.read_csv('IMU_Data_20240115_143025.csv', index_col='Timestamp', parse_dates=True)
emg_data = pd.read_csv('EMG_Data_20240115_143025.csv', index_col='Timestamp', parse_dates=True)

# Analyze data
print(f"Data collection duration: {imu_data.index[-1] - imu_data.index[0]}")
print(f"IMU sampling rate: {len(imu_data) / (imu_data.index[-1] - imu_data.index[0]).total_seconds():.1f} Hz")

# Plot data
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
imu_data[['Accel_X_g', 'Accel_Y_g', 'Accel_Z_g']].plot()
plt.title('Accelerometer Data')
plt.ylabel('Acceleration (g)')

plt.subplot(2, 1, 2)
emg_data[['Channel_0_mV', 'Channel_1_mV']].plot()
plt.title('EMG Data')
plt.ylabel('Voltage (mV)')
plt.show()
```

## Troubleshooting

### Common Issues

**"No data to export" message:**
1. Ensure device is properly connected and sending data
2. Check console output for debugging information
3. Try generating test data to verify export functionality works

**Serial port connection issues:**
1. Check that the device is properly connected via USB
2. Ensure no other applications are using the serial port
3. Try different USB ports or cables
4. Verify device drivers are installed

**WebSocket connection issues:**
1. Ensure device and computer are on the same network
2. Check device IP address (default: `miotracker.local`)
3. Verify firewall settings allow WebSocket connections

**Import errors:**
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version (3.8+ required)

### Debug Mode

Enable debug output by checking console messages when data is received. The application provides detailed logging for:
- Data packet parsing
- Storage operations
- Export processes
- Connection status

## Development

### File Structure
- `miotracker.py` - Main application and GUI logic
- `MainWindow.py` - UI class (generated from .ui file)
- `MainWindow.ui` - Qt Designer UI file
- `WebSocketClient.py` - WebSocket communication handler
- `serialCoder.py` - Serial communication utilities
- `esp_prov/` - ESP32 provisioning modules
- `protocomm/` - Protocol communication modules
- `provisioning/` - Wi-Fi provisioning protocol buffers

### Dependencies
- **PyQt5**: GUI framework
- **pyqtgraph**: Real-time plotting
- **pandas**: Data manipulation and export with DatetimeIndex
- **numpy**: Numerical computations
- **pyserial**: Serial communication
- **websocket-client**: WebSocket communication

## Deactivating Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

## License

[Add your license information here]

## Support

For issues, questions, or contributions, please [add contact information or issue tracker link].