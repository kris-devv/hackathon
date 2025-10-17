from fixed_sized_queue import FixedSizeQueue
from numpy import polyfit

class Drone:

    last_known_positions:FixedSizeQueue = FixedSizeQueue(max_size=10)

    def __init__(self, x: float, y: float):
        self.coordinate: tuple[float] = (x, y)
        self.last_known_positions.push(self.coordinate)

    def move(self, x: float, y: float):
        self.coordinate = (self.coordinate[0] + x, self.coordinate[1] + y)
        self.last_known_positions.push(self.coordinate)

    def estimate_mathematical_trajectory(self):
        points = self.last_known_positions.get_items()
        if len(points) < 2:
            return None

        if len(points) == 2:
            return polyfit([p[0] for p in points], [p[1] for p in points], 1)
        else:
            return polyfit([p[0] for p in points], [p[1] for p in points], 2)
