# devices/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List
import pandas as pd

EMGCallback = Callable[[pd.DataFrame], None]
IMUCallback = Callable[[pd.DataFrame], None]

class Device(ABC):
    """Common interface for any EMG+IMU device."""

    def __init__(self) -> None:
        self._emg_subs: List[EMGCallback] = []
        self._imu_subs: List[IMUCallback] = []

    # ---- lifecycle ----
    @abstractmethod
    def connect(self, **kwargs) -> None: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...

    # ---- data access ----
    @abstractmethod
    def get_all_data(self) -> pd.DataFrame:
        emg_df = self.get_emg_df()
        imu_df = self.get_imu_df()
        if emg_df.empty and imu_df.empty:
            return pd.DataFrame()
        return pd.concat([emg_df, imu_df], axis=1)

    def get_emg_df(self) -> pd.DataFrame:
        return pd.DataFrame()
    def get_imu_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    # ---- optional control ----
    def send_command(self, data: bytes | str) -> None:
        pass

    # ---- pub/sub for live consumers (GUI/recorder) ----
    def subscribe_emg(self, fn: EMGCallback) -> None:
        self._emg_subs.append(fn)

    def subscribe_imu(self, fn: IMUCallback) -> None:
        self._imu_subs.append(fn)

    # Helpers for subclasses to notify subscribers
    def _emit_emg(self, df_chunk: pd.DataFrame) -> None:
        for fn in list(self._emg_subs):
            try: fn(df_chunk)
            except Exception: pass

    def _emit_imu(self, df_chunk: pd.DataFrame) -> None:
        for fn in list(self._imu_subs):
            try: fn(df_chunk)
            except Exception: pass