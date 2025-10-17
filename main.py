from drone_localization_generator import DroneLocalizationGenerator
from drone import Drone
import time 
import schedule

starting_drones = [
    Drone(10, 10)
]

drone_localization_generator = DroneLocalizationGenerator(
    drones=starting_drones,
    new_drone_chance=0,
    generation_interval=2 
)

def callback(event_data):
    for drone in event_data['drones']:
        trajectory = drone.estimate_mathematical_trajectory()
        print(f"Drone at {drone.coordinate} estimated trajectory coefficients: {trajectory}")

drone_localization_generator.add_event_listener(callback)


while True:
    schedule.run_pending()  
    time.sleep(0.1)