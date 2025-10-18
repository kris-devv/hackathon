from datetime import datetime

class DronePositionRaport():
    def __init__(self, x, y, timestamp=None):
        self.x = x
        self.y = y
        self.timestamp = timestamp if timestamp is not None else datetime.now()