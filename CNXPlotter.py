import sys
import zlib

from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget
import pyqtgraph as pg

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg

# from PyQt5.QtWidgets import QDialog, QStackedWidget

import serial
import serial.tools.list_ports as portList

from serialCoder import SerialCoder

import numpy as np
import ctypes
import csv
from collections import deque
import time
import random
import array as arr
import struct
import string

from enum import Enum, auto


import pandas as pd
import libscrc
from datetime import datetime

#from MainWindow import Ui_MainWindow
from MainWindow import Ui_MainWindow

custom_crc_table = {}
poly = 0x04C11DB7

class State (Enum):
    IdleDisconnected = auto()
    IdleConnected = auto()
    InSession = auto()

class RxState (Enum):
    Sync1 = auto()
    Sync2 = auto()
    Sync3 = auto()
    Sync4 = auto()
    PacketType1 = auto()
    PacketSize1 = auto()
    CRC = auto()
    Payload = auto()

class ParserState (Enum):
    Type = auto()
    Size = auto()
    CRC = auto()
    payload = auto()


class SystemState:
    def __init__(self):
        self.__state = State.IdleDisconnected

    def setState(self, state):
        self.__state = state

    def getState(self):
        return self.__state

class MySignal(qtc.QObject):
    sig = qtc.pyqtSignal(object)

class WorkerThread(qtc.QThread):
    def __init__(self, s, sCoder, parent=None):
        qtc.QThread.__init__(self, parent)
        self.s = s
        self.sCoder = sCoder
        self.exiting = False
        self.bytesToRead = 0
        self.rxstate = RxState.Sync1
        self.signal = MySignal()

        super().__init__()

    @qtc.pyqtSlot()
    def run(self):
        try:
            while not self.exiting:
                try:
                    if(self.rxstate == RxState.Sync1):
                        dataRead = self.sCoder.read_u08(self.s)
                        if(dataRead == 0xAA):
                            self.rxstate = RxState.Sync2
                        else:
                            self.rxstate = RxState.Sync1
                    elif(self.rxstate == RxState.Sync2):
                        dataRead = self.sCoder.read_u08(self.s)
                        if(dataRead == 0x55):
                            self.rxstate = RxState.Sync3
                        else:
                            self.rxstate = RxState.Sync1
                    elif(self.rxstate == RxState.Sync3):
                       
                        dataRead = self.sCoder.read_u08(self.s)
                        if(dataRead == 0xFA):
                            self.rxstate = RxState.Sync4
                        else:
                            self.rxstate = RxState.Sync1
                    elif(self.rxstate == RxState.Sync4):
                        dataRead = self.sCoder.read_u08(self.s)
                        if(dataRead == 0xAF):
                            self.rxstate = RxState.PacketType1
                        else:
                            self.rxstate = RxState.Sync1
                    elif(self.rxstate == RxState.PacketType1):
                        dataRead = self.sCoder.read_u16(self.s)
                        self.rxstate = RxState.PacketSize1
                        self.signal.sig.emit(dataRead)
                    elif(self.rxstate == RxState.PacketSize1):
                        dataRead = self.sCoder.read_u16(self.s)
                        self.bytesToRead = dataRead - 12
                        self.rxstate = RxState.CRC
                        self.signal.sig.emit(dataRead)
                    elif(self.rxstate == RxState.CRC):
                        dataRead = self.sCoder.read_u32(self.s)
                        self.rxstate = RxState.Payload
                        self.signal.sig.emit(dataRead)
                    elif(self.rxstate == RxState.Payload):
                        dataRead = self.s.read(self.bytesToRead)
                        self.bytesToRead = 0
                        self.signal.sig.emit(dataRead)
                        self.rxstate = RxState.Sync1        
                except:
                    dataRead = {}
        finally:
            print("not running")

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        # self.ui = uic.loadUi('MainWindow.ui', self)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #Session
        self.crcErrorCounter = 0
        self.sessionToken = 0
        self.sessionSequence = 0
        self.sequenceArray = np.zeros(shape=[1,1])
        self.commandToken = 0

        # Data
        self.fs = 2000
        self.t = 1/2000
        self.timeLinSpace = list()
        self.tBase = 0
        self.payload = list()
        self.channel1 = []
        self.channel2 = list()
        self.channel3 = list()
        self.channel4 = list()
        self.channel5 = list()
        self.channel6 = list()
        self.channel7 = list()
        self.channel8 = list()
        self.eegData = [self.channel1, self.channel2,self.channel3,self.channel4,self.channel5,self.channel6,self.channel7,self.channel8]
        self.eegDataAgg = np.zeros(shape=[8,1])
        self.parserState = ParserState.Type
        self.crc = 0
        self.type = 0
        self.size = 0
        self.pendingPayload = 0
        self.sequence = 0
        self.sessionToken = 0
        self.responseCode = 0
        self.originPacketType = 0
        self.OriginToken = 0
        self.moduleError = 0
        self.sn = 0
        self.rate = 0
        self.channelMask = 0
        self.messageInBytes = []
        self.timer=qtc.QTimer()
        self.timer.timeout.connect(self.sendKeepAlive)
        #self.timer.start(5000)

        #Graph
        self.channel1Data = deque()
        self.channel2Data = deque()
        self.channel3Data = deque()
        self.channel4Data = deque()
        self.channel5Data = deque()
        self.channel6Data = deque()
        self.channel7Data = deque()
        self.channel8Data = deque()
        self.timeData = deque()
        self.curve1 = self.ui.graphWidget.plot()
        self.curve2 = self.ui.graphWidget.plot()
        self.curve3 = self.ui.graphWidget.plot()
        self.curve4 = self.ui.graphWidget.plot()
        self.curve5 = self.ui.graphWidget.plot()
        self.curve6 = self.ui.graphWidget.plot()
        self.curve7 = self.ui.graphWidget.plot()
        self.curve8 = self.ui.graphWidget.plot()
        self.plotArea = self.ui.graphWidget.getPlotItem()
        self.plotArea.setAxisItems({"left": pg.AxisItem(
            orientation='left', text=' Amplitude (V)')})
        self.plotArea.setAxisItems(
            {"bottom": pg.AxisItem(orientation='bottom', text=' Time (s)')})
        self.plotArea.setTitle(title="EEG")
        self.plotArea.showLabel('left', show=True)
        self.plotArea.showLabel('bottom', show=True)
        self.calculateImpInMCU = 0
        self.onChannels = 0

        self.lastUpdate = time.time()
        self.lastUiUpdate = time.time()

        # Connection
        self.s = ""
        self.sCoder = SerialCoder()
        self.sPorts = list(portList.comports())
        self.addPorts()
        self.sConnected = False
        self.exceptionCnt = 0
        self.packetOffset = 0
        self.mcuError = 0
        self.temp = 0
        self.vref = 0
        self.adcOvr = 0
        self.serialOvr = 0
        self.crcMismatches = 0
        self.spiOverrun = 0
        self.moduleList = ['GPIO','UART','SPI','FLASH','IWDG','ADC','ADS1299','EEPROM','SERIAL','SIGNAL_ACQ','SIGNAL_PROC','NVS','SETTINGS','COMMANDS','DATA','SESSION','SYSTEM','CLOCK','CRC']
        self.errorList = ['NONE','NO_INIT','WRONG_PARAM','BUSY','PERIPH_FAILURE','COMPONENT_FAILURE','UNKNOWN_FAILURE','UNKNOWN_COMPONENT','BUS_FAILURE','CLOCK_FAILURE','MSP_FAILURE','FEATURE_NOT_SUPPORTED','TIMEOUT']
        self.ui.baudComboBox.addItem("1000000")
        self.ui.rateComboBox.addItem("250")
        self.ui.rateComboBox.addItem("500")
        self.ui.rateComboBox.addItem("1000")
        self.ui.rateComboBox.addItem("2000")
        self.ui.rateComboBox.addItem("5000")
        self.custom_crc_table = {}
        self.poly = 0x04C11DB7
        self.generate_crc32_table(self.poly)

        #self.ui.graphWidget.setYRange(-500, 500)
        self.impedance = list()
        for i in range(8):
            self.impedance.append(0)

        # System
        self.state = SystemState()

        # Signals, slot
        self.ui.scanButton.clicked.connect(self.onScanButtonClicked)
        self.ui.connectButton.clicked.connect(self.onConnectButtonClicked)
        self.ui.startEegButton.clicked.connect(self.onStartEegButtonClicked)
        self.ui.stopEegButton.clicked.connect(self.onStopEegButtonClicked)
        self.ui.startImpButton.clicked.connect(self.onStartImpedanceButtonClicked)
        self.ui.stopImpButton.clicked.connect(self.onStopImpedanceButtonClicked)
        self.ui.stopAllButton.clicked.connect(self.onStopAllButtonClicked)
        self.ui.disconnectButton.clicked.connect(self.onDisconnectButtonClicked)
        self.ui.setSnButton.clicked.connect(self.onSetSnButtonClicked)
        self.ui.getSnButton.clicked.connect(self.onGetSnButtonClicked)
        self.ui.handshakeButton.clicked.connect(self.onSendHandsakeButtonClicked)
        self.ui.testButton.clicked.connect(self.onStartTestButtonClicked)
        self.ui.ch1CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        self.ui.ch2CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        self.ui.ch3CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        self.ui.ch4CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        self.ui.ch5CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        self.ui.ch6CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        self.ui.ch7CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        self.ui.ch8CheckBox.stateChanged.connect(self.onChCheckboxStateChanged)
        
        

    def addPorts(self):
        for p in self.sPorts:
            self.ui.portComboBox.addItem(p[0])

    def generate_crc32_table(self, _poly):
        for i in range(256):
            c = i << 24
            for j in range(8):
                c = (c << 1) ^ _poly if (c & 0x80000000) else c << 1
            self.custom_crc_table[i] = c & 0xffffffff

    def crc32_stm(self, bytes_arr):
        length = len(bytes_arr)
        crc = 0xffffffff
        k = 0
        while length >= 4:
            v = ((bytes_arr[k] << 24) & 0xFF000000) | ((bytes_arr[k+1] << 16) & 0xFF0000) | \
            ((bytes_arr[k+2] << 8) & 0xFF00) | (bytes_arr[k+3] & 0xFF)
            crc = ((crc << 8) & 0xffffffff) ^ self.custom_crc_table[0xFF & ((crc >> 24) ^ v)]
            crc = ((crc << 8) & 0xffffffff) ^ self.custom_crc_table[0xFF & ((crc >> 24) ^ (v >> 8))]
            crc = ((crc << 8) & 0xffffffff) ^ self.custom_crc_table[0xFF & ((crc >> 24) ^ (v >> 16))]
            crc = ((crc << 8) & 0xffffffff) ^ self.custom_crc_table[0xFF & ((crc >> 24) ^ (v >> 24))]
            k += 4
            length -= 4

        if length > 0:
            v = 0
            for i in range(length):
                v |= (bytes_arr[k+i] << 24-i*8)
            if length == 1:
                v &= 0xFF000000
            elif length == 2:
                v &= 0xFFFF0000
            elif length == 3:
                v &= 0xFFFFFF00

            crc = (( crc << 8 ) & 0xffffffff) ^ self.custom_crc_table[0xFF & ( (crc >> 24) ^ (v ) )]
            crc = (( crc << 8 ) & 0xffffffff) ^ self.custom_crc_table[0xFF & ( (crc >> 24) ^ (v >> 8) )]
            crc = (( crc << 8 ) & 0xffffffff) ^ self.custom_crc_table[0xFF & ( (crc >> 24) ^ (v >> 16) )]
            crc = (( crc << 8 ) & 0xffffffff) ^ self.custom_crc_table[0xFF & ( (crc >> 24) ^ (v >> 24) )]

        return crc
    
    def generateRandomString(self, length):
        digits = string.digits
        result_str = ''.join(random.choice(digits) for i in range(length))
        return result_str

    def onScanButtonClicked(self):
        self.sPorts.clear()
        self.sPorts = list(portList.comports())
        self.ui.portComboBox.clear()
        self.addPorts()

    def onConnectButtonClicked(self):
        self.s = serial.Serial(
            self.ui.portComboBox.currentText(), baudrate=1000000, timeout=100)

        self.sConnected = self.s.is_open
        if(self.sConnected):
            self.sCoder.write_u08(self.s, 0x05)
            self.state = (State.IdleConnected)
            self.worker = WorkerThread(self.s, self.sCoder)
            self.worker.exiting = False
            self.state = State.IdleConnected
            # self.timer.timeout.connect(lambda:self.updateGraph(random.choice(self.data)))
            self.worker.signal.sig.connect(self.updateUI)
            self.worker.start()

    def onDisconnectButtonClicked(self):
        self.sCoder.write_u08(self.s, 0x15)
        self.s.close()
        self.state.setState(State.IdleDisconnected)
        self.ui.startEegButton.setEnabled(False)

    def onStartEegButtonClicked(self):
        self.getChannelMask()
        self.rate = int(self.ui.rateComboBox.currentText())
        self.fs = int(self.ui.rateComboBox.currentText())
        self.t = 1/self.fs
        self.state = State.InSession
        self.sessionToken = self.sessionToken + 1
        data_buf = []
        buf_tmp = []
        for i in range(30):
            data_buf.append(0)
        data_buf[0] = 0xAA #sync
        data_buf[1] = 0x55 #sync
        data_buf[2] = 0xFA #sync
        data_buf[3] = 0xAF #sync
        data_buf[4] = 0x01 #type
        data_buf[5] = 0xB0 #type
        data_buf[6] = 0x18 #size
        data_buf[7] = 0x00 #size
        data_buf[8] = 0 
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        data_buf[12] = 0x01 #token
        data_buf[13] = 0x00 #token
        data_buf[14] = 0x00 #token
        data_buf[15] = 0x01 #token
        bytes_val = self.sessionToken.to_bytes(2,'little')
        data_buf[16] = bytes_val[0] #session token
        data_buf[17] = bytes_val[1] #session token
        data_buf[18] = self.channelMask #bitmask
        bytes_val = self.rate.to_bytes(4,'little')
        data_buf[19] = bytes_val[0] #rate
        data_buf[20] = bytes_val[1] #rate
        data_buf[21] = bytes_val[2] #rate
        data_buf[22] = bytes_val[3] #rate
        data_buf[23] = 0x00 #rate
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:24])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4,'little')
        data_buf[8] = bytes_val[0] #crc
        data_buf[9] = bytes_val[1] #crc
        data_buf[10] = bytes_val[2] #crc
        data_buf[11] = bytes_val[3] #crc
        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:24]:
            self.sCoder.write_u08(self.s, i)
    
    def onStartTestButtonClicked(self):
        self.getChannelMask()
        self.rate = int(self.ui.rateComboBox.currentText())
        self.fs = int(self.ui.rateComboBox.currentText())
        print(self.fs)
        self.t = 1/self.fs
        self.state = State.InSession
        self.sessionToken = self.sessionToken + 1
        data_buf = []
        buf_tmp = []
        for i in range(30):
            data_buf.append(0)
        data_buf[0] = 0xAA #sync
        data_buf[1] = 0x55 #sync
        data_buf[2] = 0xFA #sync
        data_buf[3] = 0xAF #sync
        data_buf[4] = 0x12 #type
        data_buf[5] = 0xB0 #type
        data_buf[6] = 0x18 #size
        data_buf[7] = 0x00 #size
        data_buf[8] = 0 
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        data_buf[12] = 0x01 #token
        data_buf[13] = 0x00 #token
        data_buf[14] = 0x00 #token
        data_buf[15] = 0x01 #token
        bytes_val = self.sessionToken.to_bytes(2,'little')
        data_buf[16] = bytes_val[0] #session token
        data_buf[17] = bytes_val[1] #session token
        data_buf[18] = self.channelMask #bitmask
        bytes_val = self.rate.to_bytes(4,'little')
        data_buf[19] = bytes_val[0] #rate
        data_buf[20] = bytes_val[1] #rate
        data_buf[21] = bytes_val[2] #rate
        data_buf[22] = bytes_val[3] #rate
        data_buf[23] = 0x00 #padding
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:24])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4,'little')
        data_buf[8] = bytes_val[0] #crc
        data_buf[9] = bytes_val[1] #crc
        data_buf[10] = bytes_val[2] #crc
        data_buf[11] = bytes_val[3] #crc
        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:24]:
            self.sCoder.write_u08(self.s, i)
    
    
    def appendData(self):
        pd.DataFrame(self.sequenceArray).to_csv("sequence" +  datetime.now().strftime("%H_%M_%S") + ".csv")
        pd.DataFrame({
            "Channel 1" : np.array(self.eegDataAgg[0]),
            "Channel 2" : np.array(self.eegDataAgg[1]),
            "Channel 3" : np.array(self.eegDataAgg[2]),
            "Channel 4" : np.array(self.eegDataAgg[3]),
            "Channel 5" : np.array(self.eegDataAgg[4]),
            "Channel 6" : np.array(self.eegDataAgg[5]),
                      }).to_csv(
            "Data_" +  datetime.now().strftime("%H_%M_%S") + ".csv")
        # print(np.array(self.eegDataAgg))
        
        # df = pd.DataFrame({"name1" : a, "name2" : b})
        # pd.DataFrame(np.append(self.eegData, self.sequenceArray)).to_csv("DATA_Seqeuence" +  datetime.now().strftime("%H:%M:%S") + ".csv")
        
    
    def onStopEegButtonClicked(self):
        self.state = State.IdleConnected
        
        # pd.DataFrame(self.sequenceArray).to_csv("sequence" +  datetime.now().strftime("%H:%M:%S") + ".csv")
        self.appendData()
        data_buf = []
        buf_tmp = []
        cmd_token = 1
        for i in range(30):
            data_buf.append(0)
        data_buf[0] = 0xAA
        data_buf[1] = 0x55
        data_buf[2] = 0xFA
        data_buf[3] = 0xAF
        data_buf[4] = 0x02
        data_buf[5] = 0xB0
        data_buf[6] = 0x10
        data_buf[7] = 0x00
        data_buf[8] = 0
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        bytes_val = cmd_token.to_bytes(4, 'little')
        data_buf[12] = bytes_val[0]
        data_buf[13] = bytes_val[1]
        data_buf[14] = bytes_val[2]
        data_buf[15] = bytes_val[3]
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:16])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4, 'little')
        data_buf[8] = bytes_val[0]
        data_buf[9] = bytes_val[1]
        data_buf[10] = bytes_val[2]
        data_buf[11] = bytes_val[3]
        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:16]:
            self.sCoder.write_u08(self.s, i)

    def onStopAllButtonClicked(self):
        self.state = State.IdleConnected
        data_buf = []
        buf_tmp = []
        cmd_token = 1
        for i in range(30):
            data_buf.append(0)
        data_buf[0] = 0xAA
        data_buf[1] = 0x55
        data_buf[2] = 0xFA
        data_buf[3] = 0xAF
        data_buf[4] = 0x13
        data_buf[5] = 0xB0
        data_buf[6] = 0x10
        data_buf[7] = 0x00
        data_buf[8] = 0
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        bytes_val = cmd_token.to_bytes(4, 'little')
        data_buf[12] = bytes_val[0]
        data_buf[13] = bytes_val[1]
        data_buf[14] = bytes_val[2]
        data_buf[15] = bytes_val[3]
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:16])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4, 'little')
        data_buf[8] = bytes_val[0]
        data_buf[9] = bytes_val[1]
        data_buf[10] = bytes_val[2]
        data_buf[11] = bytes_val[3]
        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:16]:
            self.sCoder.write_u08(self.s, i)

    def onStartImpedanceButtonClicked(self):
        self.getChannelMask()
        self.rate = int(self.ui.rateComboBox.currentText())
        self.fs = int(self.ui.rateComboBox.currentText())
        self.t = 1/self.fs
        self.state = State.InSession
        self.sessionToken = self.sessionToken + 1
        data_buf = []
        buf_tmp = []
        for i in range(30):
            data_buf.append(0)
        data_buf[0] = 0xAA #sync
        data_buf[1] = 0x55 #sync
        data_buf[2] = 0xFA #sync
        data_buf[3] = 0xAF #sync
        data_buf[4] = 0x03 #type
        data_buf[5] = 0xB0 #type
        data_buf[6] = 0x18 #size
        data_buf[7] = 0x00 #size
        data_buf[8] = 0 
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        data_buf[12] = 0x01 #token
        data_buf[13] = 0x00 #token
        data_buf[14] = 0x00 #token
        data_buf[15] = 0x01 #token
        bytes_val = self.sessionToken.to_bytes(2,'little')
        data_buf[16] = bytes_val[0] #session token
        data_buf[17] = bytes_val[1] #session token
        data_buf[18] = self.channelMask #bitmask
        bytes_val = self.rate.to_bytes(4,'little')
        data_buf[19] = bytes_val[0] #rate
        data_buf[20] = bytes_val[1] #rate
        data_buf[21] = bytes_val[2] #rate
        data_buf[22] = bytes_val[3] #rate
        data_buf[23] = 0x00 #padding
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:24])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4,'little')
        data_buf[8] = bytes_val[0] #crc
        data_buf[9] = bytes_val[1] #crc
        data_buf[10] = bytes_val[2] #crc
        data_buf[11] = bytes_val[3] #crc
        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:24]:
            self.sCoder.write_u08(self.s, i)
    
    def onStopImpedanceButtonClicked(self):
        data_buf = []
        buf_tmp = []
        cmd_token = 1
        for i in range(30):
            data_buf.append(0)
        data_buf[0] = 0xAA #sync
        data_buf[1] = 0x55 #sync
        data_buf[2] = 0xFA #sync
        data_buf[3] = 0xAF #sync
        data_buf[4] = 0x04 #type
        data_buf[5] = 0xB0 #type
        data_buf[6] = 0x10 #size
        data_buf[7] = 0x00 #size
        data_buf[8] = 0 
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        bytes_val = cmd_token.to_bytes(4, 'little')
        data_buf[12] = bytes_val[0]
        data_buf[13] = bytes_val[1]
        data_buf[14] = bytes_val[2]
        data_buf[15] = bytes_val[3]
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:16])
        data = bytearray(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4,'little')
        data_buf[8] = bytes_val[0] #crc
        data_buf[9] = bytes_val[1] #crc
        data_buf[10] = bytes_val[2] #crc
        data_buf[11] = bytes_val[3] #crc
        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:16]:
            self.sCoder.write_u08(self.s, i)

    
    def onSetSnButtonClicked(self):
        data_buf = []
        buf_tmp = []
        for i in range(16):
            data_buf.append(0)
        data_buf[0] = 0xAA #sync
        data_buf[1] = 0x55 #sync
        data_buf[2] = 0xFA #sync
        data_buf[3] = 0xAF #sync
        data_buf[4] = 0x05 #type
        data_buf[5] = 0xB0 #type
        data_buf[6] = 0x20 #size
        data_buf[7] = 0x00 #size
        data_buf[8] = 0 
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        data_buf[12] = 0x01 #token
        data_buf[13] = 0x00 #token
        data_buf[14] = 0x00 #token
        data_buf[15] = 0x01 #token
        s = self.generateRandomString(16)
        self.ui.setSnLabel.setText(f"Set Sn:{s}")
        res = []
        for ele in s:
            res.extend(ord(num) for num in ele)
        data_buf.extend(res)
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:32])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4,'little')
        data_buf[8] = bytes_val[0] #crc
        data_buf[9] = bytes_val[1] #crc
        data_buf[10] = bytes_val[2] #crc
        data_buf[11] = bytes_val[3] #crc

        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:32]:
            self.sCoder.write_u08(self.s, i)
    
    def onGetSnButtonClicked(self):
        data_buf = []
        buf_tmp = []
        for i in range(16):
            data_buf.append(0)
        data_buf[0] = 0xAA #sync
        data_buf[1] = 0x55 #sync
        data_buf[2] = 0xFA #sync
        data_buf[3] = 0xAF #sync
        data_buf[4] = 0x06 #type
        data_buf[5] = 0xB0 #type
        data_buf[6] = 0x10 #size
        data_buf[7] = 0x00 #size
        data_buf[8] = 0 
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        data_buf[12] = 0x01 #token
        data_buf[13] = 0x00 #token
        data_buf[14] = 0x00 #token
        data_buf[15] = 0x01 #token
        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:16])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4,'little')
        data_buf[8] = bytes_val[0] #crc
        data_buf[9] = bytes_val[1] #crc
        data_buf[10] = bytes_val[2] #crc
        data_buf[11] = bytes_val[3] #crc

        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:16]:
            self.sCoder.write_u08(self.s, i)
    
    def sendKeepAlive(self):
        if(self.state == State.IdleConnected):
            self.ui.aliveLabel.setText("Alive ack = false")
            cmd_token = 1
            data_buf = []
            buf_tmp = []
            for i in range(30):
                data_buf.append(0)
            data_buf[0] = 0xAA
            data_buf[1] = 0x55
            data_buf[2] = 0xFA
            data_buf[3] = 0xAF
            data_buf[4] = 0x11
            data_buf[5] = 0xB0
            data_buf[6] = 0x10
            data_buf[7] = 0x00
            data_buf[8] = 0
            data_buf[9] = 0
            data_buf[10] = 0
            data_buf[11] = 0
            bytes_val = cmd_token.to_bytes(4, 'little')
            data_buf[12] = bytes_val[0]
            data_buf[13] = bytes_val[1]
            data_buf[14] = bytes_val[2]
            data_buf[15] = bytes_val[3]
            buf_tmp.extend(data_buf[0:8])
            buf_tmp.extend(data_buf[12:16])
            data = bytes(buf_tmp)
            crc32 = self.crc32_stm(data)
            bytes_val = crc32.to_bytes(4, 'little')
            data_buf[8] = bytes_val[0]
            data_buf[9] = bytes_val[1]
            data_buf[10] = bytes_val[2]
            data_buf[11] = bytes_val[3]
            self.sCoder.write_u08(self.s, data_buf[0])
            for i in data_buf[1:16]:
                self.sCoder.write_u08(self.s, i)
    
    def onSendHandsakeButtonClicked(self):
        data_buf = []
        buf_tmp = []
        for i in range(32):
            data_buf.append(0)
        data_buf[0] = 0xAA #sync
        data_buf[1] = 0x55 #sync
        data_buf[2] = 0xFA #sync
        data_buf[3] = 0xAF #sync
        data_buf[4] = 0x08 #type
        data_buf[5] = 0xB0 #type
        data_buf[6] = 0x18 #size
        data_buf[7] = 0x00 #size
        data_buf[8] = 0 
        data_buf[9] = 0
        data_buf[10] = 0
        data_buf[11] = 0
        data_buf[12] = 0x01 #token
        data_buf[13] = 0x00 #token
        data_buf[14] = 0x00 #token
        data_buf[15] = 0x00 #token
        data_buf[16] = 0x01 #token
        data_buf[17] = 0x00 #token
        data_buf[18] = 0x00 #token
        data_buf[19] = 0x00 #token
        data_buf[20] = 0x00 #token
        data_buf[21] = 0x00 #token
        data_buf[22] = 0x00 #token
        data_buf[23] = 0x00 #token

        buf_tmp.extend(data_buf[0:8])
        buf_tmp.extend(data_buf[12:24])
        data = bytes(buf_tmp)
        crc32 = self.crc32_stm(data)
        bytes_val = crc32.to_bytes(4,'little')
        data_buf[8] = bytes_val[0] #crc
        data_buf[9] = bytes_val[1] #crc
        data_buf[10] = bytes_val[2] #crc
        data_buf[11] = bytes_val[3] #crc

        self.sCoder.write_u08(self.s, data_buf[0])
        for i in data_buf[1:24]:
            self.sCoder.write_u08(self.s, i)
    def onChCheckboxStateChanged(self):
        self.curve1.setData(x=None,y=None)
        self.curve2.setData(x=None,y=None)
        self.curve3.setData(x=None,y=None)
        self.curve4.setData(x=None,y=None)
        self.curve5.setData(x=None,y=None)
        self.curve6.setData(x=None,y=None)
        self.curve7.setData(x=None,y=None)
        self.curve8.setData(x=None,y=None)
        self.channel1Data.clear()
        self.channel2Data.clear()
        self.channel3Data.clear()
        self.channel4Data.clear()
        self.channel5Data.clear()
        self.channel6Data.clear()
        self.channel7Data.clear()
        self.channel8Data.clear()
        self.timeData.clear()
    
    def getChannelMask(self):
        self.channelMask = 0
        if(self.ui.ch1CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 1
        if(self.ui.ch2CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 2
        if(self.ui.ch3CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 4
        if(self.ui.ch4CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 8
        if(self.ui.ch5CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 16
        if(self.ui.ch6CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 32
        if(self.ui.ch7CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 64
        if(self.ui.ch8CheckBox.isChecked() == True):
            self.channelMask = self.channelMask | 128
        self.onChannels = bin(self.channelMask).count("1")
   
    def updateUI(self, dataStream):
        data = dataStream
        if(self.parserState == ParserState.Type):
            self.type = int(data)
            bytes_val = self.type.to_bytes(2,'little')
            self.messageInBytes.extend(bytearray([0xaa, 0x55, 0xfa, 0xaf]))
            self.messageInBytes.extend(bytes_val)
            self.parserState = ParserState.Size
        elif(self.parserState == ParserState.Size):
            self.size = data
            bytes_val = self.size.to_bytes(2,'little')
            self.messageInBytes.extend(bytes_val)
            self.parserState = ParserState.CRC
            self.pendingPayload = self.size
        elif(self.parserState == ParserState.CRC):
            self.crc = data
            self.parserState = ParserState.payload
        elif(self.parserState == ParserState.payload):
            self.payload = data
            self.messageInBytes.extend(self.payload )
            #self.messageInBytes.insert(0, bytearray([0xaa, 0x55, 0xfa, 0xaf]))
            calculatedCRC = self.crc32_stm(self.messageInBytes)
            self.messageInBytes.clear()
            #print(hex(self.crc))
            #print(hex(calculatedCRC))
            if(self.crc == calculatedCRC):
                if(self.type == 0xA001):
                    offset = 0
                    stoken= struct.unpack_from('<H', self.payload, offset)[0]
                    self.ui.sessionLabel.setText(f'Session: {stoken}')
                    offset = offset + 2
                    self.sequence = struct.unpack_from('<H', self.payload, offset)[0]
                    self.ui.sequenceLabel.setText(f'Sequence: {self.sequence}')
                    offset = offset + 2
                    self.sequenceArray = np.append(self.sequenceArray, self.sequence+1)
                    #print(f"on channels = {self.onChannels}")
                    for ch in range (self.onChannels):
                        chid =  struct.unpack_from('<B', self.payload, offset)[0]
                        offset = offset + 1
                        nwords =  struct.unpack_from('<B', self.payload, offset)[0]
                        offset = offset + 1
                        #print(f"on channels = {nwords}")
                        for i in range(nwords):
                            self.eegData[ch].append(struct.unpack_from('<f', self.payload, offset)[0])
                            offset = offset + 4
                    for i in range(nwords):
                        self.timeLinSpace.append((i+1)/self.fs + self.tBase)
                    self.tBase = self.timeLinSpace[-1]
                    if((self.channelMask & 1) == 1):
                        while len(self.channel1Data) > 1500:
                            #print(f"removing channel 1")
                            self.channel1Data.popleft()
                    if((self.channelMask & 2) == 2):
                        while len(self.channel2Data) > 1500:
                            #print(f"removing channel 2")
                            self.channel2Data.popleft()
                    if((self.channelMask & 4) == 4):
                        while len(self.channel3Data) > 1500:
                            #print(f"removing channel 3")
                            self.channel3Data.popleft()
                    if((self.channelMask & 8) == 8):
                        while len(self.channel4Data) > 1500:
                            #print(f"removing channel 4")
                            self.channel4Data.popleft()
                    if((self.channelMask & 16) == 16):
                        while len(self.channel5Data) > 1500:
                            #print(f"removing channel 5")
                            self.channel5Data.popleft()
                    if((self.channelMask & 32) == 32):
                        while len(self.channel6Data) > 1500:
                            #print(f"removing channel 6")
                            self.channel6Data.popleft()
                    if((self.channelMask & 64) == 64):
                        while len(self.channel7Data) > 1500:
                            #print(f"removing channel 7")
                            self.channel7Data.popleft()
                    if((self.channelMask & 128) == 128):
                        while len(self.channel8Data) > 1500:
                            #print(f"removing channel 8")
                            self.channel8Data.popleft()
                    while len(self.timeData) > 1500:
                        self.timeData.popleft()
                    self.channel1Data.extend(self.eegData[0])
                    self.channel2Data.extend(self.eegData[1])
                    self.channel3Data.extend(self.eegData[2])
                    self.channel4Data.extend(self.eegData[3])
                    self.channel5Data.extend(self.eegData[4])
                    self.channel6Data.extend(self.eegData[5])
                    self.channel7Data.extend(self.eegData[6])
                    self.channel8Data.extend(self.eegData[7])
                    self.timeData.extend(self.timeLinSpace)
                    
                              
                    self.eegDataAgg = np.append(self.eegDataAgg, np.array(self.eegData), axis= 1)
                    # print("EEG Data Size: " , np.array(self.eegData).shape)
                    
                  
                    
                    if(self.ui.ch1CheckBox.isChecked() == True):
                        self.curve1.setData(x = self.timeData, y = self.channel1Data, pen = ({'color': (0, 63,92), 'width': 1}))
                    if(self.ui.ch2CheckBox.isChecked() == True):
                        self.curve2.setData(x = self.timeData, y = self.channel2Data, pen = ({'color': (32,75,124), 'width': 1}))
                    if(self.ui.ch3CheckBox.isChecked() == True):
                        self.curve3.setData(x = self.timeData, y = self.channel3Data, pen = ({'color': (102, 81, 145), 'width': 1}))
                    if(self.ui.ch4CheckBox.isChecked() == True):
                        self.curve4.setData(x = self.timeData, y = self.channel4Data, pen = ({'color': (160,81,49), 'width': 1}))
                    if(self.ui.ch5CheckBox.isChecked() == True):
                        self.curve5.setData(x = self.timeData, y = self.channel5Data, pen = ({'color': (212,80,135), 'width': 1}))
                    if(self.ui.ch6CheckBox.isChecked() == True):
                        self.curve6.setData(x = self.timeData, y = self.channel6Data, pen = ({'color': (249,93,106), 'width': 1}))
                    if(self.ui.ch7CheckBox.isChecked() == True):
                        self.curve7.setData(x = self.timeData, y = self.channel7Data, pen = ({'color': (255,125,67), 'width': 1}))
                    if(self.ui.ch8CheckBox.isChecked() == True):
                        self.curve8.setData(x = self.timeData, y = self.channel8Data, pen = ({'color': (255,166,192), 'width': 1}))
                    self.channel1.clear()
                    self.channel2.clear()
                    self.channel3.clear()
                    self.channel4.clear()
                    self.channel5.clear()
                    self.channel6.clear()
                    self.channel7.clear()
                    self.channel8.clear()
                    self.timeLinSpace.clear()
                    #else:
                        #self.ui.errorLabel.setText("Error: No crc match")
                
                elif(self.type == 0xB009):
                    offset = 0
                    self.originPacketType = struct.unpack_from('<H', self.payload, offset)[0]
                    offset = offset + 2
                    self.OriginToken = struct.unpack_from('<H', self.payload, offset)[0]
                    offset = offset + 4
                    self.responseCode = struct.unpack_from('<H', self.payload, offset)[0]
                    offset = offset + 2

                    if(self.originPacketType == 0xB011):
                        self.ui.aliveLabel.setText("Alive ack = true")
                    if(self.originPacketType == 0xB006):
                        self.sn = struct.unpack_from('16s', self.payload, offset)[0]
                        self.sn = self.sn.decode("utf-8")
                        self.ui.getSnLabel.setText(f"Get Sn:{self.sn}")
                    if(self.originPacketType == 0xB008):
                        if(self.responseCode == 1):
                            self.ui.handLabel.setText(f'Hand: Ok')
                        else:
                            self.ui.handLabel.setText(f'Hand: wrong')
                    if(self.originPacketType == 0xB001):
                        print("start eeg reply received")
                    if(self.originPacketType == 0xB002):
                        print("stop eeg reply received")
                    if(self.originPacketType == 0xB003):
                        print("start imp reply received")
                    if(self.originPacketType == 0xB004):
                        print("stop imp reply received")
                    if(self.originPacketType == 0xB013):
                        print("stop all reply received")

                elif(self.type == 0xA002):
                    offset = 0
                    stoken= struct.unpack_from('<H', self.payload, offset)[0]
                    self.ui.sessionLabel.setText(f'Session: {stoken}')
                    offset = offset + 2
                    self.sequence = struct.unpack_from('<H', self.payload, offset)[0]
                    self.ui.sequenceLabel.setText(f'Sequence: {self.sequence}')
                    offset = offset + 2
                    for ch in range (self.onChannels):
                        chid =  struct.unpack_from('<B', self.payload, offset)[0]
                        offset = offset + 1
                        nwords =  struct.unpack_from('<B', self.payload, offset)[0]
                        offset = offset + 1
                        for i in range(nwords):
                            self.eegData[ch].append(struct.unpack_from('<f', self.payload, offset)[0])
                            offset = offset + 4
                    for i in range(nwords):
                        self.timeLinSpace.append(i/self.fs + self.tBase)
                    self.tBase = self.timeLinSpace[-1]
                    if((self.channelMask & 1) == 1):
                        while len(self.channel1Data) > 1500:
                            #print(f"removing channel 1")
                            self.channel1Data.popleft()
                    if((self.channelMask & 2) == 2):
                        while len(self.channel2Data) > 1500:
                            #print(f"removing channel 2")
                            self.channel2Data.popleft()
                    if((self.channelMask & 4) == 4):
                        while len(self.channel3Data) > 1500:
                            #print(f"removing channel 3")
                            self.channel3Data.popleft()
                    if((self.channelMask & 8) == 8):
                        while len(self.channel4Data) > 1500:
                            #print(f"removing channel 4")
                            self.channel4Data.popleft()
                    if((self.channelMask & 16) == 16):
                        while len(self.channel5Data) > 1500:
                            #print(f"removing channel 5")
                            self.channel5Data.popleft()
                    if((self.channelMask & 32) == 32):
                        while len(self.channel6Data) > 1500:
                            #print(f"removing channel 6")
                            self.channel6Data.popleft()
                    if((self.channelMask & 64) == 64):
                        while len(self.channel7Data) > 1500:
                            #print(f"removing channel 7")
                            self.channel7Data.popleft()
                    if((self.channelMask & 128) == 128):
                        while len(self.channel8Data) > 1500:
                            #print(f"removing channel 8")
                            self.channel8Data.popleft()
                    while len(self.timeData) > 1500:
                        self.timeData.popleft()
                    #print(self.eegData[0])
                    self.channel1Data.extend(self.eegData[0])
                    self.channel2Data.extend(self.eegData[1])
                    self.channel3Data.extend(self.eegData[2])
                    self.channel4Data.extend(self.eegData[3])
                    self.channel5Data.extend(self.eegData[4])
                    self.channel6Data.extend(self.eegData[5])
                    self.channel7Data.extend(self.eegData[6])
                    self.channel8Data.extend(self.eegData[7])
                    self.timeData.extend(self.timeLinSpace)
                    
                    if(self.ui.ch1CheckBox.isChecked() == True):
                        self.curve1.setData(x = self.timeData, y = self.channel1Data, pen = ({'color': (0, 63,92), 'width': 1}))
                    if(self.ui.ch2CheckBox.isChecked() == True):
                        self.curve2.setData(x = self.timeData, y = self.channel2Data, pen = ({'color': (32,75,124), 'width': 1}))
                    if(self.ui.ch3CheckBox.isChecked() == True):
                        self.curve3.setData(x = self.timeData, y = self.channel3Data, pen = ({'color': (102, 81, 145), 'width': 1}))
                    if(self.ui.ch4CheckBox.isChecked() == True):
                        self.curve4.setData(x = self.timeData, y = self.channel4Data, pen = ({'color': (160,81,49), 'width': 1}))
                    if(self.ui.ch5CheckBox.isChecked() == True):
                        self.curve5.setData(x = self.timeData, y = self.channel5Data, pen = ({'color': (212,80,135), 'width': 1}))
                    if(self.ui.ch6CheckBox.isChecked() == True):
                        self.curve6.setData(x = self.timeData, y = self.channel6Data, pen = ({'color': (249,93,106), 'width': 1}))
                    if(self.ui.ch7CheckBox.isChecked() == True):
                        self.curve7.setData(x = self.timeData, y = self.channel7Data, pen = ({'color': (255,125,67), 'width': 1}))
                    if(self.ui.ch8CheckBox.isChecked() == True):
                        self.curve8.setData(x = self.timeData, y = self.channel8Data, pen = ({'color': (255,166,192), 'width': 1}))
                    self.channel1.clear()
                    self.channel2.clear()
                    self.channel3.clear()
                    self.channel4.clear()
                    self.channel5.clear()
                    self.channel6.clear()
                    self.channel7.clear()
                    self.channel8.clear()
                    self.timeLinSpace.clear()

                elif(self.type == 0xF010):
                    offset = 4
                    self.payload = data
                    self.mcuError = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.errorMcuLabel.setText(f'Error mcu: {self.mcuError}')
                    offset = offset + 4
                    self.temp = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.tempLabel.setText(f'Temp: {self.temp} C')
                    offset = offset + 4
                    self.vref = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.vrefLabel.setText(f'Vref: {self.vref} mV')
                    offset = offset + 4
                    self.serialOvr = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.serialOvrLabel.setText(f'Serial ovr: {self.serialOvr}')
                    offset = offset + 4
                    self.adcOvr = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.adcOvrLabel.setText(f'Adc ovr: {self.adcOvr}')
                    offset = offset + 4
                    self.crcMismatches = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.crcMismatchLabel.setText(f'Crc err: {self.crcMismatches}')
                    offset = offset + 4
                    self.spiOverrun = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.spiOvrLabel.setText(f'Spi ovr: {self.spiOverrun}')

                elif(self.type == 0xF011):
                    offset = 4
                    self.payload = data
                    self.moduleError = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.moduleLabel.setText(f'Mod err: {self.moduleList[self.moduleError]}')
                    offset = offset + 4
                    self.mcuError = struct.unpack_from('<I', self.payload, offset)[0]
                    self.ui.errorMcuLabel.setText(f'Error mcu: {self.errorList[self.mcuError]}')

                else:
                    self.ui.errorLabel.setText("Error: No type match")
            else:
                print("wrong crc")
                self.crcErrorCounter =  self.crcErrorCounter +1
                print(self.crcErrorCounter)
            self.parserState = ParserState.Type


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(main)
    # widget.addWidget(dbWin)
    widget.show()
    sys.exit(app.exec_())
    # main()



