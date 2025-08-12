



from _devices.MioTracker import MioTracker


class DeviceFactory:
    @staticmethod
    def create(name: str, **kwargs):
        name = name.lower()
        if name == "miotracker":
            return MioTracker(**kwargs)
        #elif name == "freeemg":
        #    return FreeEMGDevice()
        #else:
            #raise ValueError(f"Unknown device: {name}")
            
            
            
if __name__ == "__main__":
    dev = DeviceFactory.create("miotracker", transport="websocket", websocketuri="ws://miotracker.local/start")