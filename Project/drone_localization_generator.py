import random
from drone import Drone
import schedule
from datetime import datetime

class DroneLocalizationGenerator:
    """
    Generates drone localization events at regular intervals.
    """
    
    def __init__(self, drones, generation_interval=0, new_drone_chance=0.3, drone_move_chance=0.3):
        self.drones = drones
        self.generation_interval = generation_interval
        self.new_drone_chance = new_drone_chance
        self.drone_move_chance = drone_move_chance
        self.event_listeners = []
        
        schedule.every(self.generation_interval).seconds.do(self.tick)

    def add_event_listener(self, callback):
        """Add an event listener callback."""
        self.event_listeners.append(callback)

    def remove_event_listener(self, callback):
        """Remove an event listener callback."""
        if callback in self.event_listeners:
            self.event_listeners.remove(callback)

    def emit_tick_event(self):
        """Emit tick event to all listeners."""
        event_data = {
            'current_time': datetime.now(),
            'drones': self.drones.copy(),
            'drone_count': len(self.drones)
        }
        
        for listener in self.event_listeners:
            listener(event_data)

    def tick(self):
        """Process one tick - move drones and potentially spawn new ones."""
        for drone in self.drones:
            drone.move(random.randint(0, 10), random.randint(0, 2))
        
        if random.random() < self.new_drone_chance:
            new_drone = Drone(x=0, y=random.randint(0, 300))
            self.drones.append(new_drone)

        self.emit_tick_event()


