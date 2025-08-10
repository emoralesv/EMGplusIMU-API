import sys

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg

from pyqtgraph import PlotWidget
import pyqtgraph as pg
import numpy as np

import serial
import serial.tools.list_ports as portList

from serialCoder import SerialCoder

import ctypes
import threading
import time
from datetime import datetime
from collections import deque
from enum import Enum, auto
import random

import struct
import pandas as pd

from MainWindow import Ui_MainWindow
from WebSocketClient import WebSocketClient



# IMU Data Type Constants (matching C enum)
DATA_TYPE_EMG = 0
DATA_TYPE_IMU = 1
DATA_TYPE_STATUS = 2



class RxState (Enum):
    Sync1 = auto()
    Sync2 = auto()
    Type = auto()
    Size = auto()
    Payload = auto()

class ParserState (Enum):
    Type = auto()
    Size = auto()
    payload = auto()

class UsbSignal(qtc.QObject):
    usbSig = qtc.pyqtSignal(object)

class UsbThread(qtc.QThread):
    def __init__(self, s, sCoder, parent = None):
        qtc.QThread.__init__(self, parent)
        self.s = s
        self.sCoder = sCoder
        self.exiting = False
        self.bytesToRead = 0
        self.rxstate = RxState.Sync1
        self.signal = UsbSignal()
        super().__init__()
    
    def run(self):
        try:
            while not self.exiting:
                try:
                    if(self.rxstate == RxState.Sync1):
                        self.bytesToRead = 0
                        dataRead = self.sCoder.read_u08(self.s)
                        if(dataRead == 0xA5):
                            self.rxstate = RxState.Sync2
                        else:
                            self.rxstate = RxState.Sync1
                    elif(self.rxstate == RxState.Sync2):
                        dataRead = self.sCoder.read_u08(self.s)
                        if(dataRead == 0x5A):
                            self.rxstate = RxState.Type
                        else:
                            self.rxstate = RxState.Sync1
                    elif(self.rxstate == RxState.Type):
                        dataRead = self.sCoder.read_u08(self.s)
                        self.rxstate = RxState.Size
                        self.signal.usbSig.emit(dataRead)
                    elif(self.rxstate == RxState.Size):
                        dataRead = self.sCoder.read_u08(self.s)
                        self.bytesToRead = dataRead - 4
                        self.rxstate = RxState.Payload
                        self.signal.usbSig.emit(dataRead)
                    elif(self.rxstate == RxState.Payload):
                        dataRead = self.s.read(self.bytesToRead)
                        self.rxstate = RxState.Sync1  
                        self.signal.usbSig.emit(dataRead)   
                except:
                    dataRead = {}
        finally:
            print("not running")

class MainWindow(qtw.QMainWindow):
    """
    Main application window for MioTracker GUI.
    
    Features:
    - Real-time IMU and EMG data visualization
    - Data storage with pandas DatetimeIndex timestamps
    - CSV export functionality for both data types
    - WebSocket and USB connectivity
    - Test data generation for debugging
    
    Data Storage:
    - IMU data: self.imuDataHistory, self.imuTimestamps (accelerometer + gyroscope)
    - EMG data: self.emgDataHistory, self.emgTimestamps (2-channel EMG signals)
    
    Data Access:
    - getIMUDataFrame(): Returns pandas DataFrame with DatetimeIndex
    - getEMGDataFrame(): Returns pandas DataFrame with DatetimeIndex  
    - getLatestIMUData(): Returns most recent IMU sample with timestamp
    - getLatestEMGData(): Returns most recent EMG sample with timestamp
    
    Export Methods:
    - exportIMUData(): Export IMU data to CSV
    - exportEMGData(): Export EMG data to CSV
    - exportAllData(): Export both datasets to separate CSV files
    
    Usage Example:
        # Get data as pandas DataFrames
        imu_df = main_window.getIMUDataFrame()
        emg_df = main_window.getEMGDataFrame()
        
        # Check data
        print(f"IMU data shape: {imu_df.shape}")
        print(f"EMG data shape: {emg_df.shape}")
        
        # Export data
        main_window.exportAllData("my_experiment")
        
        # Generate test data for debugging
        main_window.generateTestData(100, 50)
    """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.channel1 = list()
        self.channelData = [deque(maxlen=5000), deque(maxlen=5000)]  # This list now holds both channel1Data and channel2Data
        self.tBase = 0  # Initialize time base
        self.Fs = 250  # Sampling frequency in Hz
        self.timeData = deque(maxlen=5000)
        self.parserState = ParserState.Type

        # EMG Data storage with datetime timestamps
        self.emgDataHistory = deque(maxlen=10000)  # Store EMG data with timestamps
        self.emgTimestamps = deque(maxlen=10000)   # Store datetime objects for EMG data
        
        # IMU Data storage - MPU6050 ESP-IDF component
        # Accelerometer: typically ±2g, ±4g, ±8g, or ±16g range (in g units where 1g = 9.8 m/s²)
        # Gyroscope: typically ±250°/s, ±500°/s, ±1000°/s, or ±2000°/s range (in degrees per second)
        self.imuData = []  # Current IMU reading [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        self.imuDataHistory = deque(maxlen=10000)  # Historical IMU data (increased capacity)
        self.imuTimestamps = deque(maxlen=10000)   # Store datetime objects for IMU data

        # Connection
        self.s = ""
        self.sCoder = SerialCoder()
        self.sPorts = list(portList.comports())
        self.ui.baudComboBox.addItem("1000000")

        self.type = 0
        self.size = 0

        self.curve1 = self.ui.graphWidget.plot()

        self.plotArea = self.ui.graphWidget.getPlotItem()
        self.plotArea.setAxisItems({"left": pg.AxisItem(
            orientation='left', text=' Amplitude (mV)')})
        self.plotArea.setAxisItems(
            {"bottom": pg.AxisItem(orientation='bottom', text=' Time (s)')})
        self.plotArea.setTitle(title="EMG")
        self.plotArea.showLabel('left', show=True)
        self.plotArea.showLabel('bottom', show=True)

        # IMU 2D Plot Setup
        self.setupIMUPlots()

        self.ui.connectButton.clicked.connect(self.onConnectButtonClicked)
        self.ui.disconnectButton.clicked.connect(self.onDisconnectButtonClicked)
        self.ui.usbConnectButton.clicked.connect(self.onUsbConnectButtonClicked)
        self.ui.usbDisconnectButton.clicked.connect(self.onUsbDisconnectButtonClicked)
        self.ui.scanButton.clicked.connect(self.onScanButtonClicked)
        
        # Add context menu to raw data text area for clearing
        self.ui.rawDataTextEdit.setContextMenuPolicy(qtc.Qt.CustomContextMenu)
        self.ui.rawDataTextEdit.customContextMenuRequested.connect(self.showRawDataContextMenu)

        self.client = None

        self.wst = None



    def setupIMUPlots(self):
        """Initialize the IMU plots for gyroscope and accelerometer data"""
        # Setup Gyroscope Plot - MPU6050 outputs in degrees per second (°/s)
        self.gyroPlotArea = self.ui.gyroPlotWidget.getPlotItem()
        self.gyroPlotArea.setAxisItems({"left": pg.AxisItem(orientation='left', text='Angular Velocity (°/s)')})
        self.gyroPlotArea.setAxisItems({"bottom": pg.AxisItem(orientation='bottom', text='Time (s)')})
        self.gyroPlotArea.setTitle(title="Gyroscope Data")
        self.gyroPlotArea.showLabel('left', show=True)
        self.gyroPlotArea.showLabel('bottom', show=True)
        self.gyroPlotArea.addLegend(offset=(10, 10))
        
        # Setup Accelerometer Plot - MPU6050 outputs in gravity units (g) where 1g = 9.8 m/s²
        self.accelPlotArea = self.ui.accelPlotWidget.getPlotItem()
        self.accelPlotArea.setAxisItems({"left": pg.AxisItem(orientation='left', text='Acceleration (g)')})
        self.accelPlotArea.setAxisItems({"bottom": pg.AxisItem(orientation='bottom', text='Time (s)')})
        self.accelPlotArea.setTitle(title="Accelerometer Data")
        self.accelPlotArea.showLabel('left', show=True)
        self.accelPlotArea.showLabel('bottom', show=True)
        self.accelPlotArea.addLegend(offset=(10, 10))
        
        # Create gyroscope curves (X=Red, Y=Green, Z=Blue)
        self.gyroCurveX = self.ui.gyroPlotWidget.plot(pen=pg.mkPen(color=(255, 0, 0), width=2), name='Gyro X')
        self.gyroCurveY = self.ui.gyroPlotWidget.plot(pen=pg.mkPen(color=(0, 255, 0), width=2), name='Gyro Y')
        self.gyroCurveZ = self.ui.gyroPlotWidget.plot(pen=pg.mkPen(color=(0, 0, 255), width=2), name='Gyro Z')
        
        # Create accelerometer curves (X=Red, Y=Green, Z=Blue)
        self.accelCurveX = self.ui.accelPlotWidget.plot(pen=pg.mkPen(color=(255, 0, 0), width=2), name='Accel X')
        self.accelCurveY = self.ui.accelPlotWidget.plot(pen=pg.mkPen(color=(0, 255, 0), width=2), name='Accel Y')
        self.accelCurveZ = self.ui.accelPlotWidget.plot(pen=pg.mkPen(color=(0, 0, 255), width=2), name='Accel Z')
        
        # Initialize time data for IMU plots
        self.imuTimeDataPlot = deque(maxlen=1000)
        
        # Set maximum lines for raw data display to prevent memory issues
        self.ui.rawDataTextEdit.document().setMaximumBlockCount(1000)

    def updateIMUPlots(self, accel_data, gyro_data):
        """Update the IMU plots with new accelerometer and gyroscope data"""
        # Add timestamp for this data point
        current_time = time.time()
        if not hasattr(self, 'imu_start_time'):
            self.imu_start_time = current_time
        
        relative_time = current_time - self.imu_start_time
        self.imuTimeDataPlot.append(relative_time)
        
        # Store data for plotting (separate deques for each axis)
        if not hasattr(self, 'gyroDataX'):
            self.gyroDataX = deque(maxlen=1000)
            self.gyroDataY = deque(maxlen=1000) 
            self.gyroDataZ = deque(maxlen=1000)
            self.accelDataX = deque(maxlen=1000)
            self.accelDataY = deque(maxlen=1000)
            self.accelDataZ = deque(maxlen=1000)
        
        # Add new data points
        self.gyroDataX.append(gyro_data[0])
        self.gyroDataY.append(gyro_data[1])
        self.gyroDataZ.append(gyro_data[2])
        self.accelDataX.append(accel_data[0])
        self.accelDataY.append(accel_data[1])
        self.accelDataZ.append(accel_data[2])
        
        # Update gyroscope plots
        self.gyroCurveX.setData(x=list(self.imuTimeDataPlot), y=list(self.gyroDataX))
        self.gyroCurveY.setData(x=list(self.imuTimeDataPlot), y=list(self.gyroDataY))
        self.gyroCurveZ.setData(x=list(self.imuTimeDataPlot), y=list(self.gyroDataZ))
        
        # Update accelerometer plots
        self.accelCurveX.setData(x=list(self.imuTimeDataPlot), y=list(self.accelDataX))
        self.accelCurveY.setData(x=list(self.imuTimeDataPlot), y=list(self.accelDataY))
        self.accelCurveZ.setData(x=list(self.imuTimeDataPlot), y=list(self.accelDataZ))

    def resetIMUPlots(self):
        """Clear the IMU plots and reset data"""
        # Clear all data deques
        if hasattr(self, 'gyroDataX'):
            self.gyroDataX.clear()
            self.gyroDataY.clear()
            self.gyroDataZ.clear()
            self.accelDataX.clear()
            self.accelDataY.clear()
            self.accelDataZ.clear()
            self.imuTimeDataPlot.clear()
            
        # Clear the plot curves
        if hasattr(self, 'gyroCurveX'):
            self.gyroCurveX.setData([], [])
            self.gyroCurveY.setData([], [])
            self.gyroCurveZ.setData([], [])
            self.accelCurveX.setData([], [])
            self.accelCurveY.setData([], [])
            self.accelCurveZ.setData([], [])
            

    def onScanButtonClicked(self):
        self.sPorts.clear()
        self.sPorts = list(portList.comports())
        self.ui.portComboBox.clear()
        for p in self.sPorts:
            print(p)
            self.ui.portComboBox.addItem(p[0])

    def onConnectButtonClicked(self):
        self.client = WebSocketClient("ws://miotracker.local/start")
        self.client.frameSignal.sig.connect(self.updateUI)
        self.client.start()
        self.wst = threading.Thread(target=self.client.ws.run_forever)
        self.wst.start()

    def onDisconnectButtonClicked(self):
        self.client.close()

    def onUsbConnectButtonClicked(self):
        self.s = serial.Serial(self.ui.portComboBox.currentText(), baudrate=1000000, timeout=100)
        if(self.s.is_open):
            self.worker = UsbThread(self.s, self.sCoder)
            self.worker.exiting = False
            self.worker.signal.usbSig.connect(self.updateUsbUI)
            self.worker.start()
    
    def onUsbDisconnectButtonClicked(self):
        self.s.close()
    
    def onUpGainButtonClicked(self):
        print("up")
    
    def onDownGainButtonClicked(self):
        print("Down")
    
    def showRawDataContextMenu(self, position):
        """Show context menu for raw data text area"""
        menu = qtw.QMenu(self)
        clear_action = menu.addAction("Clear Raw Data")
        copy_action = menu.addAction("Copy All")
        save_action = menu.addAction("Save to File")
        
        # Add separator and data export options
        menu.addSeparator()
        export_imu_action = menu.addAction("Export IMU Data to CSV")
        export_emg_action = menu.addAction("Export EMG Data to CSV")
        export_all_action = menu.addAction("Export All Data to CSV")
        
        # Add separator and data clearing options
        menu.addSeparator()
        clear_imu_action = menu.addAction("Clear IMU Data")
        clear_emg_action = menu.addAction("Clear EMG Data")
        clear_all_data_action = menu.addAction("Clear All Sensor Data")
        
        # Add data summary option
        menu.addSeparator()
        summary_action = menu.addAction("Show Data Summary")
        
        # Add test data generation option
        menu.addSeparator()
        generate_test_action = menu.addAction("Generate Test Data")
        
        action = menu.exec_(self.ui.rawDataTextEdit.mapToGlobal(position))
        
        if action == clear_action:
            self.ui.rawDataTextEdit.clear()
        elif action == copy_action:
            clipboard = qtw.QApplication.clipboard()
            clipboard.setText(self.ui.rawDataTextEdit.toPlainText())
        elif action == save_action:
            self.saveRawDataToFile()
        elif action == export_imu_action:
            self.exportIMUData()
        elif action == export_emg_action:
            self.exportEMGData()
        elif action == export_all_action:
            self.exportAllData()
        elif action == clear_imu_action:
            self.clearIMUData()
            qtw.QMessageBox.information(self, "Data Cleared", "IMU data has been cleared.")
        elif action == clear_emg_action:
            self.clearEMGData()
            qtw.QMessageBox.information(self, "Data Cleared", "EMG data has been cleared.")
        elif action == clear_all_data_action:
            reply = qtw.QMessageBox.question(self, "Clear All Data", 
                                           "Are you sure you want to clear all sensor data?",
                                           qtw.QMessageBox.Yes | qtw.QMessageBox.No)
            if reply == qtw.QMessageBox.Yes:
                self.clearAllData()
                qtw.QMessageBox.information(self, "Data Cleared", "All sensor data has been cleared.")
        elif action == summary_action:
            summary = self.getDataSummary()
            summary_text = f"""Data Summary:
            
IMU Data:
- Samples: {summary['imu_samples']}
- Duration: {summary['imu_duration']:.2f} seconds
- Start Time: {summary['imu_start_time']}

EMG Data:
- Packets: {summary['emg_packets']}
- Duration: {summary['emg_duration']:.2f} seconds  
- Start Time: {summary['emg_start_time']}"""
            qtw.QMessageBox.information(self, "Data Summary", summary_text)
        elif action == generate_test_action:
            reply = qtw.QMessageBox.question(self, "Generate Test Data", 
                                           "Generate test data for testing export functionality?\n"
                                           "This will clear existing data.",
                                           qtw.QMessageBox.Yes | qtw.QMessageBox.No)
            if reply == qtw.QMessageBox.Yes:
                success = self.generateTestData()
                if success:
                    qtw.QMessageBox.information(self, "Test Data Generated", 
                                              "Test data has been generated successfully!\n"
                                              "You can now test the export functionality.")
                else:
                    qtw.QMessageBox.warning(self, "Error", "Failed to generate test data.")
    
    def saveRawDataToFile(self):
        """Save raw data to a text file"""
        from datetime import datetime
        default_filename = f"RawData_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filename, _ = qtw.QFileDialog.getSaveFileName(
            self, "Save Raw Data", default_filename, "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.ui.rawDataTextEdit.toPlainText())
                print(f"Raw data saved to {filename}")
            except Exception as e:
                qtw.QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
    
    def updateUI(self, dataStream):
        data = dataStream
        
        # Display raw WebSocket data in the text box with timestamp
        if isinstance(data, bytes) and len(data) > 0:
            import time
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
            hex_data = data.hex(' ').upper()
            # Limit hex display to first 50 bytes for readability
            if len(data) > 50:
                display_hex = hex_data[:150] + "..."  # 50 bytes * 3 chars per byte
                self.ui.rawDataTextEdit.appendPlainText(f"[{timestamp}] WebSocket Raw ({len(data)} bytes): {display_hex}")
            else:
                self.ui.rawDataTextEdit.appendPlainText(f"[{timestamp}] WebSocket Raw ({len(data)} bytes): {hex_data}")
        
        # Add debugging for data length
        print(f"WebSocket updateUI called with data length: {len(data) if isinstance(data, bytes) else 'Not bytes'}")
        
        try:
            sampleInterval = 1 / self.Fs  # Time interval between samples
            offset = 2
            self.type = struct.unpack_from('<B', data, offset)[0]
            offset = offset + 1
            self.size = struct.unpack_from('<B', data, offset)[0]
            offset = offset + 1
            
            # Add packet type info to raw data display
            packet_types = {DATA_TYPE_EMG: "EMG", DATA_TYPE_IMU: "IMU", DATA_TYPE_STATUS: "STATUS"}
            packet_type_name = packet_types.get(self.type, f"UNKNOWN({self.type})")
            import time
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]
            self.ui.rawDataTextEdit.appendPlainText(f"[{timestamp}] WebSocket Packet: {packet_type_name}, Size: {self.size}")
            
            # Debug packet info
            print(f"WebSocket Packet - Type: {self.type} ({packet_type_name}), Size: {self.size}")
            
            # Handle different packet types
            if self.type == DATA_TYPE_EMG:
                print("Processing EMG data packet...")
                # Handle EMG data (existing logic)
                current_timestamp = datetime.now()  # Get current datetime for this packet
                emg_data_packet = {'ch0': [], 'ch1': []}  # Store data for this packet
                
                for ch in range(2):
                    chid =  struct.unpack_from('<H', data, offset)[0]
                    offset = offset + 2
                    nwords =  struct.unpack_from('<H', data, offset)[0]
                    offset = offset + 2
                    print(f"Channel {ch}: ID={chid}, Words={nwords}")
                    for _ in range(nwords):
                        value = struct.unpack_from('<f', data, offset)[0]
                        self.channelData[ch].append(value)
                        emg_data_packet[f'ch{ch}'].append(value)
                        offset = offset + 4
                
                # Store EMG data with timestamp for export
                self.emgDataHistory.append(emg_data_packet)
                self.emgTimestamps.append(current_timestamp)
                print(f"Stored EMG data - Total packets: {len(self.emgDataHistory)}")
                
                sync = struct.unpack_from('<I', data, offset)[0]
                if sync:
                    # Ensure a vertical line is created or move the existing one
                    if not hasattr(self, 'vLine'):
                        self.vLine = pg.InfiniteLine(angle=90, movable=False)  # Create a vertical line
                        self.plotArea.addItem(self.vLine)  # Assuming 'plotWidget' is your PlotWidget instance
                    self.vLine.setPos(self.tBase)  # Position line at the last time point
                for _ in range(nwords):
                    self.timeData.append(self.tBase)
                    self.tBase += sampleInterval

                self.curve1.setData(x = self.timeData, y=self.channelData[0])
                
            elif self.type == DATA_TYPE_IMU:
                print("Processing IMU data packet...")
                # Handle IMU data from WebSocket
                if len(data) >= (offset + 28):  # Need at least 24 bytes for data + 4 for sync
                    # Parse 6 IMU data values (accelerometer + gyroscope) as floats
                    imu_values = []
                    for i in range(6):
                        value = struct.unpack_from('<f', data, offset)[0]
                        imu_values.append(value)
                        offset += 4
                    
                    # Parse sync
                    sync = struct.unpack_from('<I', data, offset)[0]
                    
                    # Store IMU data [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                    self.imuData = imu_values
                    
                    # Add to history with datetime timestamp
                    self.imuDataHistory.append(imu_values.copy())
                    self.imuTimestamps.append(datetime.now()) # Store datetime object
                    print(f"Stored IMU data - Total samples: {len(self.imuDataHistory)}")
                    
                    # Print IMU data for debugging - MPU6050 ESP-IDF component units
                    print(f"WebSocket IMU Data - Sync: 0x{sync:08X}")
                    print(f"Accel (g): ({imu_values[0]:.3f}, {imu_values[1]:.3f}, {imu_values[2]:.3f}) "
                          f"Gyro (°/s): ({imu_values[3]:.3f}, {imu_values[4]:.3f}, {imu_values[5]:.3f})")
                    
                    # Update IMU plots
                    self.updateIMUPlots(imu_values[:3], imu_values[3:])
                    
                    # Update UI labels if they exist
                    try:
                        if hasattr(self.ui, 'imuAccelLabel'):
                            self.ui.imuAccelLabel.setText(f"Accel: ({imu_values[0]:.2f}, {imu_values[1]:.2f}, {imu_values[2]:.2f})")
                        if hasattr(self.ui, 'imuGyroLabel'):
                            self.ui.imuGyroLabel.setText(f"Gyro: ({imu_values[3]:.2f}, {imu_values[4]:.2f}, {imu_values[5]:.2f})")
                    except AttributeError:
                        pass
                else:
                    print(f"WebSocket IMU packet too short: {len(data)} bytes, need {offset + 28}")
            else:
                print(f"Unknown WebSocket packet type: {self.type}")
        except Exception as e:
            print(f"Error processing WebSocket data: {e}")
            import traceback
            traceback.print_exc()

    def updateUsbUI(self, dataStream):
        data = dataStream
        
        # Display raw bytes in the text box with timestamp and type info
        if isinstance(data, bytes):
            import time
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
            hex_data = data.hex(' ').upper()
            # Limit hex display to first 50 bytes for readability
            if len(data) > 50:
                display_hex = hex_data[:150] + "..."  # 50 bytes * 3 chars per byte
                self.ui.rawDataTextEdit.appendPlainText(f"[{timestamp}] USB Raw ({len(data)} bytes): {display_hex}")
            else:
                self.ui.rawDataTextEdit.appendPlainText(f"[{timestamp}] USB Raw ({len(data)} bytes): {hex_data}")
        elif isinstance(data, int):
            # Display single byte values (like type and size)
            import time
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]
            self.ui.rawDataTextEdit.appendPlainText(f"[{timestamp}] USB Byte: 0x{data:02X} ({data})")

        # Add debugging for USB data
        print(f"USB updateUsbUI called with data: {data} (type: {type(data)})")

        sampleInterval = 1 / self.Fs
        if(self.parserState == ParserState.Type):
            self.type = int(data)
            self.parserState = ParserState.Size
            print(f"USB Parser - Got Type: {self.type}")
        elif(self.parserState == ParserState.Size):
            self.size = data
            self.parserState = ParserState.payload
            print(f"USB Parser - Got Size: {self.size}")
        elif(self.parserState == ParserState.payload):
            self.payload = data
            print(f"USB Parser - Got Payload: {len(data) if isinstance(data, bytes) else data} bytes")
            
            try:
                # Add packet type info to raw data display
                packet_types = {DATA_TYPE_EMG: "EMG", DATA_TYPE_IMU: "IMU", DATA_TYPE_STATUS: "STATUS"}
                packet_type_name = packet_types.get(self.type, f"UNKNOWN({self.type})")
                import time
                timestamp = time.strftime("%H:%M:%S.%f")[:-3]
                self.ui.rawDataTextEdit.appendPlainText(f"[{timestamp}] USB Packet: {packet_type_name}, Size: {self.size}")
                
                # Debug packet info
                print(f"USB Packet - Type: {self.type} ({packet_type_name}), Size: {self.size}")
                
                # Handle different packet types
                if self.type == DATA_TYPE_EMG:
                    print("Processing USB EMG data packet...")
                    # Handle EMG data (existing logic)
                    current_timestamp = datetime.now()  # Get current datetime for this packet
                    emg_data_packet = {'ch0': [], 'ch1': []}  # Store data for this packet
                    
                    offset = 0
                    for ch in range(2):
                        chid =  struct.unpack_from('<H', data, offset)[0]
                        offset = offset + 2
                        nwords =  struct.unpack_from('<H', data, offset)[0]
                        offset = offset + 2
                        print(f"USB Channel {ch}: ID={chid}, Words={nwords}")
                        for _ in range(nwords):
                            value = struct.unpack_from('<f', data, offset)[0]
                            self.channelData[ch].append(value)
                            emg_data_packet[f'ch{ch}'].append(value)
                            offset = offset + 4
                    
                    # Store EMG data with timestamp for export
                    self.emgDataHistory.append(emg_data_packet)
                    self.emgTimestamps.append(current_timestamp)
                    print(f"Stored USB EMG data - Total packets: {len(self.emgDataHistory)}")
                    
                    sync = struct.unpack_from('<I', data, offset)[0]
                    if sync:
                        # Ensure a vertical line is created or move the existing one
                        if not hasattr(self, 'vLine'):
                            self.vLine = pg.InfiniteLine(angle=90, movable=False)  # Create a vertical line
                            self.plotArea.addItem(self.vLine)  # Assuming 'plotWidget' is your PlotWidget instance
                        self.vLine.setPos(self.tBase)  # Position line at the last time point

                    for _ in range(nwords):
                        self.timeData.append(self.tBase)
                        self.tBase += sampleInterval
                    
                    self.curve1.setData(x = self.timeData, y=self.channelData[0])
                    
                elif self.type == DATA_TYPE_IMU:
                    print("Processing USB IMU data packet...")
                    # Handle IMU data from USB - payload contains only data(24) + sync(4) = 28 bytes
                    # USB protocol strips sync/header bytes in UsbThread before sending payload
                    if len(data) >= 28:
                        offset = 0
                        
                        # Parse 6 IMU data values (accelerometer + gyroscope) as floats
                        imu_values = []
                        for i in range(6):
                            value = struct.unpack_from('<f', data, offset)[0]
                            imu_values.append(value)
                            offset += 4
                        
                        # Parse sync
                        sync = struct.unpack_from('<I', data, offset)[0]
                        
                        # Store IMU data [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                        self.imuData = imu_values
                        
                        # Add to history with datetime timestamp
                        self.imuDataHistory.append(imu_values.copy())
                        self.imuTimestamps.append(datetime.now()) # Store datetime object
                        print(f"Stored USB IMU data - Total samples: {len(self.imuDataHistory)}")
                        
                        # Print IMU data for debugging - MPU6050 ESP-IDF component units
                        print(f"USB IMU Data - Sync: 0x{sync:08X}")
                        print(f"Accel (g): ({imu_values[0]:.3f}, {imu_values[1]:.3f}, {imu_values[2]:.3f}) "
                              f"Gyro (°/s): ({imu_values[3]:.3f}, {imu_values[4]:.3f}, {imu_values[5]:.3f})")
                        
                        # Update IMU plots
                        self.updateIMUPlots(imu_values[:3], imu_values[3:])
                        
                        # Update UI labels if they exist
                        try:
                            if hasattr(self.ui, 'imuAccelLabel'):
                                self.ui.imuAccelLabel.setText(f"Accel: ({imu_values[0]:.2f}, {imu_values[1]:.2f}, {imu_values[2]:.2f})")
                            if hasattr(self.ui, 'imuGyroLabel'):
                                self.ui.imuGyroLabel.setText(f"Gyro: ({imu_values[3]:.2f}, {imu_values[4]:.2f}, {imu_values[5]:.2f})")
                        except AttributeError:
                            pass
                    else:
                        print(f"USB IMU packet too short: {len(data)} bytes, need 28")
                
                else:
                    print(f"Unknown USB packet type: {self.type}")
                
            except Exception as e:
                print(f"Error processing USB data: {e}")
                import traceback
                traceback.print_exc()
            
            self.parserState = ParserState.Type



    def getIMUDataFrame(self):
        """Get IMU data as a pandas DataFrame with DatetimeIndex"""
        if len(self.imuDataHistory) == 0:
            return pd.DataFrame()
        
        try:
            imu_array = np.array(list(self.imuDataHistory))
            timestamps = list(self.imuTimestamps)
            
            df = pd.DataFrame({
                "Accel_X_g": imu_array[:, 0],
                "Accel_Y_g": imu_array[:, 1], 
                "Accel_Z_g": imu_array[:, 2],
                "Gyro_X_dps": imu_array[:, 3],
                "Gyro_Y_dps": imu_array[:, 4],
                "Gyro_Z_dps": imu_array[:, 5]
            }, index=pd.DatetimeIndex(timestamps, name='Timestamp'))
            
            return df
        except Exception as e:
            print(f"Error creating IMU DataFrame: {e}")
            return pd.DataFrame()
    
    def getEMGDataFrame(self):
        """Get EMG data as a pandas DataFrame with DatetimeIndex"""
        if len(self.emgDataHistory) == 0:
            return pd.DataFrame()
        
        try:
            data_rows = []
            timestamps = []
            
            for emg_packet, timestamp in zip(self.emgDataHistory, self.emgTimestamps):
                ch0_data = emg_packet['ch0']
                ch1_data = emg_packet['ch1']
                
                # Calculate time offset for each sample within the packet
                sample_interval = pd.Timedelta(milliseconds=1000/(self.Fs * len(ch0_data))) if len(ch0_data) > 1 else pd.Timedelta(0)
                
                for j in range(len(ch0_data)):
                    sample_timestamp = timestamp + (j * sample_interval)
                    data_rows.append({
                        'Channel_0_mV': ch0_data[j],
                        'Channel_1_mV': ch1_data[j] if j < len(ch1_data) else np.nan
                    })
                    timestamps.append(sample_timestamp)
            
            df = pd.DataFrame(data_rows, index=pd.DatetimeIndex(timestamps, name='Timestamp'))
            return df
        except Exception as e:
            print(f"Error creating EMG DataFrame: {e}")
            return pd.DataFrame()
    
    def getLatestIMUData(self):
        """Returns the most recent IMU data as a dictionary with timestamp"""
        if self.imuData and len(self.imuData) == 6 and self.imuTimestamps:
            return {
                'timestamp': self.imuTimestamps[-1],
                'accel_x_g': self.imuData[0],
                'accel_y_g': self.imuData[1], 
                'accel_z_g': self.imuData[2],
                'gyro_x_dps': self.imuData[3],
                'gyro_y_dps': self.imuData[4],
                'gyro_z_dps': self.imuData[5]
            }
        return None
    
    def getLatestEMGData(self):
        """Returns the most recent EMG data as a dictionary with timestamp"""
        if self.emgDataHistory and self.emgTimestamps:
            latest_packet = self.emgDataHistory[-1]
            return {
                'timestamp': self.emgTimestamps[-1],
                'channel_0_mV': latest_packet['ch0'][-1] if latest_packet['ch0'] else None,
                'channel_1_mV': latest_packet['ch1'][-1] if latest_packet['ch1'] else None,
                'packet_data': latest_packet
            }
        return None
    
    def clearIMUData(self):
        """Clears stored IMU data history"""
        self.imuDataHistory.clear()
        self.imuTimestamps.clear()
        self.imuData = []
        # Also reset the IMU plots
        self.resetIMUPlots()
    
    def clearEMGData(self):
        """Clears stored EMG data history"""
        self.emgDataHistory.clear()
        self.emgTimestamps.clear()
        self.channelData[0].clear()
        self.channelData[1].clear()
        self.timeData.clear()
        self.tBase = 0
        # Clear the EMG plot
        if hasattr(self, 'curve1'):
            self.curve1.setData([], [])
    
    def clearAllData(self):
        """Clears both IMU and EMG data"""
        self.clearIMUData()
        self.clearEMGData()
    
    def exportIMUData(self, filename=None):
        """Export IMU data to CSV file with DatetimeIndex"""
        if len(self.imuDataHistory) == 0:
            print("No IMU data to export")
            qtw.QMessageBox.warning(self, "No Data", "No IMU data available to export.")
            return False
            
        if filename is None:
            default_filename = f"IMU_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filename, _ = qtw.QFileDialog.getSaveFileName(
                self, "Export IMU Data", default_filename, "CSV Files (*.csv);;All Files (*)"
            )
            if not filename:  # User cancelled
                return False
        
        try:
            # Convert data to numpy arrays
            imu_array = np.array(list(self.imuDataHistory))
            timestamps = list(self.imuTimestamps)
            
            # Create DataFrame with DatetimeIndex
            df = pd.DataFrame({
                "Accel_X_g": imu_array[:, 0],
                "Accel_Y_g": imu_array[:, 1], 
                "Accel_Z_g": imu_array[:, 2],
                "Gyro_X_dps": imu_array[:, 3],
                "Gyro_Y_dps": imu_array[:, 4],
                "Gyro_Z_dps": imu_array[:, 5]
            }, index=pd.DatetimeIndex(timestamps, name='Timestamp'))
            
            df.to_csv(filename)
            message = f"IMU data exported to {filename} with {len(df)} samples"
            print(message)
            qtw.QMessageBox.information(self, "Export Successful", message)
            return True
            
        except Exception as e:
            error_msg = f"Error exporting IMU data: {e}"
            print(error_msg)
            qtw.QMessageBox.critical(self, "Export Error", error_msg)
            return False
    
    def exportEMGData(self, filename=None):
        """Export EMG data to CSV file with DatetimeIndex"""
        if len(self.emgDataHistory) == 0:
            print("No EMG data to export")
            qtw.QMessageBox.warning(self, "No Data", "No EMG data available to export.")
            return False
            
        if filename is None:
            default_filename = f"EMG_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filename, _ = qtw.QFileDialog.getSaveFileName(
                self, "Export EMG Data", default_filename, "CSV Files (*.csv);;All Files (*)"
            )
            if not filename:  # User cancelled
                return False
        
        try:
            # Prepare data for DataFrame
            data_rows = []
            timestamps = []
            
            for i, (emg_packet, timestamp) in enumerate(zip(self.emgDataHistory, self.emgTimestamps)):
                # Each packet can contain multiple samples
                ch0_data = emg_packet['ch0']
                ch1_data = emg_packet['ch1']
                
                # Calculate time offset for each sample within the packet (assuming uniform sampling)
                sample_interval = pd.Timedelta(milliseconds=1000/(self.Fs * len(ch0_data))) if len(ch0_data) > 1 else pd.Timedelta(0)
                
                for j in range(len(ch0_data)):
                    sample_timestamp = timestamp + (j * sample_interval)
                    data_rows.append({
                        'Channel_0_mV': ch0_data[j],
                        'Channel_1_mV': ch1_data[j] if j < len(ch1_data) else np.nan
                    })
                    timestamps.append(sample_timestamp)
            
            # Create DataFrame with DatetimeIndex
            df = pd.DataFrame(data_rows, index=pd.DatetimeIndex(timestamps, name='Timestamp'))
            
            df.to_csv(filename)
            message = f"EMG data exported to {filename} with {len(df)} samples"
            print(message)
            qtw.QMessageBox.information(self, "Export Successful", message)
            return True
            
        except Exception as e:
            error_msg = f"Error exporting EMG data: {e}"
            print(error_msg)
            qtw.QMessageBox.critical(self, "Export Error", error_msg)
            return False
    
    def exportAllData(self, base_filename=None):
        """Export both IMU and EMG data to separate CSV files"""
        if base_filename is None:
            base_filename = f"SensorData_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        imu_success = self.exportIMUData(f"{base_filename}_IMU.csv")
        emg_success = self.exportEMGData(f"{base_filename}_EMG.csv")
        
        if imu_success and emg_success:
            print(f"All sensor data exported successfully with base filename: {base_filename}")
        elif imu_success:
            print("Only IMU data exported successfully")
        elif emg_success:
            print("Only EMG data exported successfully")
        else:
            print("No data was exported")
        
        return imu_success, emg_success
    
    def getDataSummary(self):
        """Get summary of stored data"""
        return {
            'imu_samples': len(self.imuDataHistory),
            'emg_packets': len(self.emgDataHistory),
            'imu_duration': (self.imuTimestamps[-1] - self.imuTimestamps[0]).total_seconds() if len(self.imuTimestamps) > 1 else 0,
            'emg_duration': (self.emgTimestamps[-1] - self.emgTimestamps[0]).total_seconds() if len(self.emgTimestamps) > 1 else 0,
            'imu_start_time': self.imuTimestamps[0] if self.imuTimestamps else None,
            'emg_start_time': self.emgTimestamps[0] if self.emgTimestamps else None
        }

    def generateTestData(self, num_imu_samples=100, num_emg_packets=50):
        """Generate test data for debugging and testing export functionality"""
        import random
        
        print(f"Generating test data: {num_imu_samples} IMU samples, {num_emg_packets} EMG packets")
        
        # Clear existing data
        self.clearAllData()
        
        # Generate test IMU data
        base_time = datetime.now()
        for i in range(num_imu_samples):
            # Generate realistic IMU values
            # Accelerometer: simulate gravity + small movements (-2g to +2g)
            accel_x = random.uniform(-2.0, 2.0)
            accel_y = random.uniform(-2.0, 2.0) 
            accel_z = random.uniform(8.0, 11.0)  # Gravity + noise
            
            # Gyroscope: simulate rotation (-200 to +200 degrees/sec)
            gyro_x = random.uniform(-200.0, 200.0)
            gyro_y = random.uniform(-200.0, 200.0)
            gyro_z = random.uniform(-200.0, 200.0)
            
            imu_sample = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            
            # Add timestamp with realistic intervals (assume 100Hz IMU sampling)
            timestamp = base_time + pd.Timedelta(milliseconds=i*10)
            
            self.imuDataHistory.append(imu_sample)
            self.imuTimestamps.append(timestamp)
        
        # Generate test EMG data
        for i in range(num_emg_packets):
            # Each EMG packet contains multiple samples (simulate varying packet sizes)
            samples_per_packet = random.randint(5, 20)
            
            emg_packet = {'ch0': [], 'ch1': []}
            
            for j in range(samples_per_packet):
                # Generate realistic EMG values (typically -500 to +500 mV)
                ch0_value = random.uniform(-500.0, 500.0)
                ch1_value = random.uniform(-500.0, 500.0)
                
                emg_packet['ch0'].append(ch0_value)
                emg_packet['ch1'].append(ch1_value)
            
            # Add timestamp with realistic intervals (assume packets every 20ms)
            timestamp = base_time + pd.Timedelta(milliseconds=i*20)
            
            self.emgDataHistory.append(emg_packet)
            self.emgTimestamps.append(timestamp)
        
        print(f"Test data generated successfully!")
        print(f"IMU samples: {len(self.imuDataHistory)}")
        print(f"EMG packets: {len(self.emgDataHistory)}")
        
        # Update the UI to show test data was generated
        self.ui.rawDataTextEdit.appendPlainText(f"[TEST] Generated {num_imu_samples} IMU samples and {num_emg_packets} EMG packets for testing")
        
        return True

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    main = MainWindow()
    widget = qtw.QStackedWidget()
    widget.addWidget(main)
    # widget.addWidget(dbWin)
    widget.show()
    sys.exit(app.exec_())
    # main()
        

