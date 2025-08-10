import websocket
import threading
import time

def on_message(ws, message):
    print("Received message: ", message)

def on_error(ws, error):
    print("Error: ", error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Connection opened")

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://miotracker.local/start",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    wst = threading.Thread(target=ws.run_forever)
    wst.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ws.close()
        print("Connection closed")