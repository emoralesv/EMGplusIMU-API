# MiotrackerSDK.py
import threading
import time
import struct
from datetime import datetime
from collections import deque
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
import sys
import pandas as pd
import serial
import serial.tools.list_ports as portList

from serialCoder import SerialCoder
from scipy.signal import iirnotch, filtfilt, butter

from SDK._utils.WebSocketClient import WebSocketClient


# Tipos de paquete (idénticos al GUI)
DATA_TYPE_EMG = 0
DATA_TYPE_IMU = 1
DATA_TYPE_STATUS = 2


def list_serial_devices():
    """Lista puertos serial tal como hace onScanButtonClicked (pero sin GUI)."""
    ports = list(portList.comports())
    devices = []
    for p in ports:
        print(p.device, p.name, p.description)
        devices.append({
            "device": p.device,
            "name": p.name,
            "description": p.description,
            "manufacturer": getattr(p, "manufacturer", None),
            "vid": getattr(p, "vid", None),
            "pid": getattr(p, "pid", None),
            "serial_number": getattr(p, "serial_number", None),
        })
    return devices
class OnlineEMGOffsetComp:
    """
    Compensador de offset dinámico por canal usando media exponencial (EMA).
    baseline[t] = (1-alpha)*baseline[t-1] + alpha*x[t]
    x_corr = x - baseline
    alpha = dt / tau  (dt=1/fs, tau=constante de tiempo)
    """
    def __init__(self, zero_center = 1, n_channels=2, fs=250, tau_ms=300.0):
        self.zero_center = zero_center
        self.n = n_channels
        self.fs = float(fs)
        self.tau = max(1e-3, tau_ms/1000.0)
        self.alpha = min(1.0, (1.0/self.fs)/self.tau)
        self.baseline = np.zeros(self.n, dtype=float)

    def reset(self):
        self.baseline[:] = 0.0

    def update(self, sample_vec):
        x = np.array(sample_vec, dtype=float)
        """
        sample_vec: iterable de tamaño n_channels (muestra instantánea)
        retorna: sample_corr (np.ndarray)
        """
        if not self.zero_center:
            return x

        self.baseline = (1.0 - self.alpha)*self.baseline + self.alpha*x

        return x - self.baseline

class _UsbReader(threading.Thread):
    """
    Reproducción del UsbThread del GUI pero sin PyQt.
    Usa la misma máquina de estados: Sync1 -> Sync2 -> Type -> Size -> Payload.
    Emite el paquete completo al callback on_packet(ptype, payload).
    """
    def __init__(self, ser, sCoder, on_packet):
        super().__init__(daemon=True)
        self.ser = ser
        self.sCoder = sCoder
        self.on_packet = on_packet
        self._stop_event = threading.Event()
        self.bytes_to_read = 0
        self.rxstate = "Sync1"

    def stop(self):
        self._stop_event.set()

    def _read_exact(self, n, timeout_s=1.0):
        """Lee exactamente n bytes o None si no logra completar."""
        out = bytearray()
        t0 = time.perf_counter()
        while len(out) < n and not self._stop_event.is_set():
            chunk = self.ser.read(n - len(out))
            if chunk:
                out.extend(chunk)
            elif (time.perf_counter() - t0) > timeout_s:
                return None
        return bytes(out)

    def run(self):
        try:
            while not self._stop_event.is_set():
                try:
                    if self.rxstate == "Sync1":
                        self.bytes_to_read = 0
                        b = self.sCoder.read_u08(self.ser)
                        if b == 0xA5:
                            self.rxstate = "Sync2"
                        else:
                            self.rxstate = "Sync1"

                    elif self.rxstate == "Sync2":
                        b = self.sCoder.read_u08(self.ser)
                        if b == 0x5A:
                            self.rxstate = "Type"
                        else:
                            self.rxstate = "Sync1"

                    elif self.rxstate == "Type":
                        self._ptype = self.sCoder.read_u08(self.ser)
                        self.rxstate = "Size"

                    elif self.rxstate == "Size":
                        size = self.sCoder.read_u08(self.ser)
                        # en tu GUI: bytesToRead = size - 4 (payload sin cabecera)
                        if size < 4:
                            self.rxstate = "Sync1"
                            continue
                        self.bytes_to_read = size - 4
                        self.rxstate = "Payload"

                    elif self.rxstate == "Payload":
                        payload = self._read_exact(self.bytes_to_read, timeout_s=1.0)
                        self.rxstate = "Sync1"
                        if payload is None:
                            continue
                        # entrega al parser principal
                        self.on_packet(self._ptype, payload)

                except Exception:
                    # silencioso, como el GUI
                    pass
        finally:
            try:
                self.ser.close()
            except Exception:
                pass

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import sys
import pandas as pd
import numpy as np  # si no lo tienes importado arriba

class MioTracker:
    """
    SDK sin GUI que replica la lógica de miotracker.py:

    - Parsing USB (igual a UsbThread) y WebSocket (offset=2, type y size de 1 byte)
    - Almacenamiento:
        EMG -> emgDataHistory: deque de {'ch0':[], 'ch1':[]}, emgTimestamps: deque de datetime
        IMU -> imuDataHistory: deque de [ax, ay, az, gx, gy, gz], imuTimestamps: deque de datetime
    - Exportadores a CSV con DatetimeIndex
    - Generador de datos de prueba (API compatible con "Generate Test Data")
    - Gráficos simples con matplotlib

    API principal:
        mt = MioTracker()
        mt.connect_serial("COM8"); mt.start_serial(); time.sleep(5); mt.stop_serial()
        mt.connect_socket("ws://miotracker.local/start"); mt.start_socket(); time.sleep(5); mt.stop_socket()
        mt.save_emg_csv("emg.csv"); mt.save_imu_csv("imu.csv")
        mt.plot_emg(); mt.plot_imu()
    """
    def __init__(self,zero_center = 1, fs_emg=100, maxlen=10000):
        self.zero_center = zero_center
        self.Fs = fs_emg
        self._lock = threading.Lock()

        # Buffers (idénticos en intención al GUI)
        self.channelData = [deque(maxlen=5000), deque(maxlen=5000)]  # para plotting rápido
        self.timeData = deque(maxlen=5000)
        self.tBase = 0.0

        self.emgDataHistory = deque(maxlen=maxlen)
        self.emgTimestamps = deque(maxlen=maxlen)

        self.imuDataHistory = deque(maxlen=maxlen)
        self.imuTimestamps = deque(maxlen=maxlen)
        self.imuData = []  # último vector IMU

        # Serial
        self._ser = None
        self._usb_thread = None
        self._sCoder = SerialCoder()

        # WebSocket
        self._ws_client = None
        self._ws_thread = None

    # =============== Conexión SERIAL ===============
    def connect_serial(self, port: str, baudrate=1000000, timeout=1.0):
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        # similar a GUI refresh
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
    
    def disconnect_serial(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            self._ser = None

    def start_serial(self):
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("Puerto serial no abierto. Llama a connect_serial() primero.")
        if self._usb_thread and self._usb_thread.is_alive():
            return
        self._usb_thread = _UsbReader(self._ser, self._sCoder, on_packet=self._on_usb_packet)
        self._usb_thread.start()

    def stop_serial(self):
        if self._usb_thread:
            self._usb_thread.stop()
            self._usb_thread.join(timeout=1.0)
            self._usb_thread = None

    # =============== Conexión WebSocket ===============
    def connect_socket(self, uri="ws://miotracker.local/start"):
        if WebSocketClient is None:
            raise RuntimeError("WebSocketClient no está disponible en el entorno.")
        self._ws_client = WebSocketClient(uri)

        # En el GUI, conectan frameSignal.sig a updateUI; aquí simulamos ese callback:
        self._ws_client.frameSignal.sig.connect(self._on_ws_frame)

    def start_socket(self):
        if not self._ws_client:
            self.connect_socket()
        self._ws_client.start()
        # Igual al GUI: ws.run_forever en un hilo aparte
        self._ws_thread = threading.Thread(target=self._ws_client.ws.run_forever, daemon=True)
        self._ws_thread.start()

    def stop_socket(self):
        if self._ws_client:
            try:
                self._ws_client.close()
            except Exception:
                pass
        self._ws_client = None
        self._ws_thread = None

    # =============== Parsing de paquetes ===============
    # USB: recibe ptype y payload (sin cabecera A5 5A type size)
    def _on_usb_packet(self, ptype: int, payload: bytes):
        try:
            if ptype == DATA_TYPE_EMG:
                self._parse_emg_payload(payload)
            elif ptype == DATA_TYPE_IMU:
                self._parse_imu_payload(payload)
            else:
                # STATUS/UNKNOWN: ignorar
                pass
        except Exception:
            pass



    

    # WS: los bytes traen offset=2, luego <type:u8>, <size:u8>, payload=(size-4)
    def _on_ws_frame(self, data: bytes):
        try:
            if not isinstance(data, (bytes, bytearray)) or len(data) < 4:
                return
            offset = 2
            ptype = struct.unpack_from('<B', data, offset)[0]; offset += 1
            size = struct.unpack_from('<B', data, offset)[0]; offset += 1
            need = size - 4
            payload = data[offset: offset + need] if len(data) >= offset + need else data[offset:]
            if ptype == DATA_TYPE_EMG:
                self._parse_emg_payload(payload)
            elif ptype == DATA_TYPE_IMU:
                self._parse_imu_payload(payload)
        except Exception:
            pass

    # ====== Estructura de payloads (idéntico al archivo) ======
    def _parse_emg_payload(self, buf: bytes):
        """
        EMG payload:
          canal 0: <u16 chid>, <u16 nwords>, nwords * <float32>
          canal 1: <u16 chid>, <u16 nwords>, nwords * <float32>
          sync: <u32>
        """
        off = 0
        pkt = {'ch0': [], 'ch1': []}
        try:
            for ch in range(2):
                if off + 4 > len(buf):
                    break
                chid = struct.unpack_from('<H', buf, off)[0]; off += 2
                nwords = struct.unpack_from('<H', buf, off)[0]; off += 2
                for _ in range(nwords):
                    if off + 4 > len(buf): break
                    val = struct.unpack_from('<f', buf, off)[0]; off += 4
                    pkt[f'ch{ch}'].append(val)

            # sync (si está)
            # if off + 4 <= len(buf):
            #     sync = struct.unpack_from('<I', buf, off)[0]

            with self._lock:
                self.emgDataHistory.append(pkt)
                if not hasattr(self, "_ema"):
                    self._ema = OnlineEMGOffsetComp(n_channels=2, zero_center=self.zero_center, fs=self.Fs, tau_ms=300)

                ch0 = np.array(pkt['ch0'], float)
                ch1 = np.array(pkt['ch1'], float)
                stack = np.vstack([ch0, ch1]).T  # shape (N,2)

                corr = np.vstack([self._ema.update(s) for s in stack])  # aplica EMA por muestra
                pkt['ch0'] = corr[:,0].tolist()
                pkt['ch1'] = corr[:,1].tolist()
                
                self.emgTimestamps.append(datetime.now())

                # para plotting básico (como GUI usa channelData[0])
                for i, v in enumerate(pkt.get('ch0', [])):
                    self.channelData[0].append(v)
                    # timebase equivalente
                    sampleInterval = 1.0 / float(self.Fs)
                    self.timeData.append(self.tBase + i*sampleInterval)
                if len(pkt.get('ch0', [])) > 0:
                    self.tBase = self.timeData[-1] if self.timeData else self.tBase

        except Exception:
            pass

    def _parse_imu_payload(self, buf: bytes):
        """
        IMU payload:
          6 * <float32> (Accel XYZ en g, Gyro XYZ en °/s)
          + <u32 sync>
        """
        try:
            if len(buf) < 28:
                return
            off = 0
            vals = []
            for _ in range(6):
                vals.append(struct.unpack_from('<f', buf, off)[0]); off += 4
            # sync = struct.unpack_from('<I', buf, off)[0]
            with self._lock:
                self.imuData = vals
                self.imuDataHistory.append(vals.copy())
                self.imuTimestamps.append(datetime.now())
        except Exception:
            pass

    # =============== DataFrames (idénticos en intención) ===============
    def getIMUDataFrame(self,trim_sec=1) -> pd.DataFrame:
        with self._lock:
            if len(self.imuDataHistory) == 0:
                return pd.DataFrame()
            imu_array = np.array(list(self.imuDataHistory))
            timestamps = list(self.imuTimestamps)
        try:
            df = pd.DataFrame({
                "Accel_X_g": imu_array[:, 0],
                "Accel_Y_g": imu_array[:, 1],
                "Accel_Z_g": imu_array[:, 2]-1,
                "Gyro_X_dps": imu_array[:, 3],
                "Gyro_Y_dps": imu_array[:, 4],
                "Gyro_Z_dps": imu_array[:, 5]
            }, index=pd.DatetimeIndex(timestamps, name='Timestamp'))
            if trim_sec and not df.empty:
                t0 = df.index[0] + timedelta(seconds=float(trim_sec))
                df = df[df.index >= t0]
            return df
        except Exception:
            return pd.DataFrame()

    def getEMGDataFrame(self, trim_sec=1) -> pd.DataFrame:
        with self._lock:
            if len(self.emgDataHistory) == 0:
                return pd.DataFrame()
            packets = list(self.emgDataHistory)
            timestamps = list(self.emgTimestamps)
        try:
            rows, ts = [], []
            for pkt, t0 in zip(packets, timestamps):
                ch0 = pkt['ch0']
                ch1 = pkt['ch1']
                n = max(len(ch0), len(ch1))
                if n <= 1:
                    sample_dt = pd.Timedelta(0)
                else:
                    sample_dt = pd.Timedelta(milliseconds=1000/(self.Fs * len(ch0)))
                for j in range(n):
                    ts.append(t0 + j*sample_dt)
                    val_ch0 = ch0[j] if j < len(ch0) else np.nan
                    val_ch1 = ch1[j] if j < len(ch1) else np.nan
                    rows.append({
                        'Channel_0_mV': val_ch0,
                        'Channel_1_mV': val_ch1,
                        'EMG_diff_mV': val_ch0 - val_ch1  # diferencial
                    })
            df = pd.DataFrame(rows, index=pd.DatetimeIndex(ts, name='Timestamp'))
            if trim_sec and not df.empty:
                t0 = df.index[0] + timedelta(seconds=float(trim_sec))
                df = df[df.index >= t0] 
            return df
        except Exception:
            return pd.DataFrame()


    # =============== Exportadores (mismo formato) ===============
    def save_imu_csv(self, filename: str) -> bool:
        df = self.getIMUDataFrame()
        if df.empty:
            print("No IMU data to export")
            return False
        df.to_csv(filename)
        print(f"IMU data exported to {filename} ({len(df)} samples)")
        return True

    def save_emg_csv(self, filename: str) -> bool:
        df = self.getEMGDataFrame()
        if df.empty:
            print("No EMG data to export")
            return False
        df.to_csv(filename)
        print(f"EMG data exported to {filename} ({len(df)} samples)")
        return True

    # =============== Gráficas rápidas (equivalente a curve1 + IMU plots) ===============
    def plot_emg(self):
        df = self.getEMGDataFrame()
        if df.empty:
            print("No hay datos EMG.")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        df[["Channel_0_mV", "Channel_1_mV"]].plot(ax=ax1)
        plt.title("EMG (mV)")
        plt.xlabel("Tiempo")
        plt.ylabel("mV")
        plt.legend()


        df[["EMG_diff_mV"]].plot(ax = ax2)
        plt.title("EMG (mV)")
        plt.xlabel("Tiempo")
        plt.ylabel("mV")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_imu(self):
        df = self.getIMUDataFrame()
        if df.empty:
            print("No hay datos IMU.")
            return
        # Acelerómetro
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        df[["Accel_X_g", "Accel_Y_g", "Accel_Z_g"]].plot(ax=ax1)
        plt.title("IMU - Acelerómetro (g)")
        plt.xlabel("Tiempo")
        plt.ylabel("g")
        plt.legend()

        # Giroscopio
        df[["Gyro_X_dps", "Gyro_Y_dps", "Gyro_Z_dps"]].plot(ax = ax2)
        plt.title("IMU - Giroscopio (°/s)")
        plt.xlabel("Tiempo")
        plt.ylabel("°/s")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    # =============== Utilidades (equivalentes a limpiar/exportar/generar) ===============
    def clearIMUData(self):
        with self._lock:
            self.imuDataHistory.clear()
            self.imuTimestamps.clear()
            self.imuData = []

    def clearEMGData(self):
        with self._lock:
            self.emgDataHistory.clear()
            self.emgTimestamps.clear()
            self.channelData[0].clear()
            self.channelData[1].clear()
            self.timeData.clear()
            self.tBase = 0.0

    def clearAllData(self):
        self.clearIMUData()
        self.clearEMGData()

    def generateTestData(self, num_imu_samples=100, num_emg_packets=50):
        """Replica la opción de 'Generate Test Data' del GUI (sin UI)."""
        import random
        self.clearAllData()
        base_time = datetime.now()

        # IMU (100 Hz)
        for i in range(num_imu_samples):
            imu_sample = [
                random.uniform(-2.0, 2.0),
                random.uniform(-2.0, 2.0),
                random.uniform(8.0, 11.0),
                random.uniform(-200.0, 200.0),
                random.uniform(-200.0, 200.0),
                random.uniform(-200.0, 200.0),
            ]
            timestamp = base_time + pd.Timedelta(milliseconds=i*10)
            with self._lock:
                self.imuDataHistory.append(imu_sample)
                self.imuTimestamps.append(timestamp)

        # EMG (paquetes con 5–20 muestras por canal)
        for i in range(num_emg_packets):
            samples_per_packet = random.randint(5, 20)
            emg_packet = {'ch0': [], 'ch1': []}
            for _ in range(samples_per_packet):
                emg_packet['ch0'].append(random.uniform(-500.0, 500.0))
                emg_packet['ch1'].append(random.uniform(-500.0, 500.0))
            timestamp = base_time + pd.Timedelta(milliseconds=i*20)
            with self._lock:
                self.emgDataHistory.append(emg_packet)
                self.emgTimestamps.append(timestamp)

        return True

    # =============== (Opcional) Envío de comandos crudos al dispositivo ===============
    def send_raw(self, data: bytes):
        if self._ser and self._ser.is_open:
            self._ser.write(data)
            self._ser.flush()

    # Helpers de “última muestra” (equivalentes a getLatest*)
    def getLatestIMUData(self):
        with self._lock:
            if self.imuData and self.imuTimestamps:
                return {
                    'timestamp': self.imuTimestamps[-1],
                    'accel_x_g': self.imuData[0],
                    'accel_y_g': self.imuData[1],
                    'accel_z_g': self.imuData[2],
                    'gyro_x_dps': self.imuData[3],
                    'gyro_y_dps': self.imuData[4],
                    'gyro_z_dps': self.imuData[5],
                }
        return None

    def getLatestEMGData(self):
        with self._lock:
            if self.emgDataHistory and self.emgTimestamps:
                latest_packet = self.emgDataHistory[-1]
                return {
                    'timestamp': self.emgTimestamps[-1],
                    'channel_0_mV': latest_packet['ch0'][-1] if latest_packet['ch0'] else None,
                    'channel_1_mV': latest_packet['ch1'][-1] if latest_packet['ch1'] else None,
                    'packet_data': latest_packet
                }
        return None

    def live_plot_emg_pg(self, window_sec=5, refresh_hz=30, differential=False):
        created_app = False
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
            created_app = True

        win = pg.GraphicsLayoutWidget(show=True, title="EMG Live")
        win.resize(500, 300)

        if differential:
            p1 = win.addPlot(title="EMG diferencial (mV)")
            curve_diff = p1.plot(pen=pg.mkPen(width=2))
            # guarda referencias para que no se GC
            self._live_curves = {"diff": curve_diff}
        else:
            p1 = win.addPlot(title="EMG canales (mV)")
            curve0 = p1.plot(pen=pg.mkPen(width=2))
            curve1 = p1.plot(pen=pg.mkPen(width=2, style=QtCore.Qt.PenStyle.DotLine))
            self._live_curves = {"ch0": curve0, "ch1": curve1}

        p1.showGrid(x=True, y=True)

        def update():
            df = self.getEMGDataFrame()
            print(len(df))
            if df.empty:
                return
            tmax = df.index.max()
            df = df[df.index >= tmax - pd.Timedelta(seconds=window_sec)]
            x = (df.index - df.index[0]).total_seconds().to_numpy()

            if differential and "EMG_diff_mV" in df.columns:
                y = df["EMG_diff_mV"].astype(float).to_numpy()
                self._live_curves["diff"].setData(x, y)
            else:
                y0 = df["Channel_0_mV"].astype(float).to_numpy()
                y1 = df["Channel_1_mV"].astype(float).to_numpy()
                self._live_curves["ch0"].setData(x, y0)
                self._live_curves["ch1"].setData(x, y1)

        timer = QtCore.QTimer(win)  # <- parent = win para que no lo recoja el GC
        timer.timeout.connect(update)
        timer.start(int(1000/refresh_hz))
        self._live_timer = timer   # referencia extra por si cierras/abres

        # cierra limpio
        def _on_close(_evt):
            try:
                self._live_timer.stop()
            except Exception:
                pass
            _evt.accept()
        win.closeEvent = _on_close

        # Solo corre el event loop si lo creamos aquí
        if created_app:
            app.exec()

