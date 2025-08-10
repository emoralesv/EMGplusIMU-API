# Re-writing the WebSocket client as a class after the reset

import websocket
import threading
from PyQt5 import QtCore as qtc

class FrameSignal(qtc.QObject):
    sig = qtc.pyqtSignal(object)
    
class WebSocketClient(qtc.QObject):
    def __init__(self, url):
        self.url = url
        self.frameSignal = FrameSignal()
        self.ws = None
        #super().__init__()

    def on_message(self, ws, message):
        """Handle incoming messages as binary data."""
        self.frameSignal.sig.emit(message)
        #print("Binary message received as a list: ", binary_data_list)

    def on_error(self, ws, error):
        """Handle errors."""
        print("Error occurred: ", error)

    def on_close(self, ws, close_status_code, close_msg):
        """Handle closure of the connection."""
        print("### WebSocket closed ###")

    def on_open(self, ws):
        """Handle opening of the connection."""
        print("WebSocket opened")
    
    def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            self.ws.close()
            print("WebSocket connection closed")

    def start(self):
        """Create and run the WebSocket client."""
        print(self.url)
        self.ws = websocket.WebSocketApp(self.url,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                        on_close=self.on_close)