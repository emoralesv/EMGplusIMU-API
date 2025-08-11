from MiotrackerSDK import MioTracker,list_serial_devices
import time

list_serial_devices()


mt = MioTracker(fs_emg=100,zero_center = 1 )

mt.connect_serial("COM6")   
time.sleep(1)  
mt.start_serial()

mt.live_plot_emg_pg(window_sec=10, refresh_hz=30, differential=False)
       
mt.stop_serial()
mt.disconnect_serial()

#mt.connect_socket("ws://miotracker.local/start")
#mt.start_socket()
#time.sleep(20)
#mt.stop_socket()

mt.plot_emg()
mt.plot_imu()

