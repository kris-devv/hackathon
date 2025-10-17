from drone_localization_generator import DroneLocalizationGenerator
from drone import Drone
import time 
import schedule

if __name__ == "__main__":
    starting_drones = [Drone(10, 10)]

    drone_localization_generator = DroneLocalizationGenerator(
        drones=starting_drones,
        new_drone_chance=0,
        generation_interval=2 
    )

    def callback(event_data):
        for drone in event_data['drones']:
            trajectory = drone.estimate_mathematical_trajectory()
            if trajectory is not None:
                print(f"Drone at {drone.coordinate} - Polynomial coefficients: {trajectory}")
                print(f"  - Degree: {'Linear (ax + b)' if len(trajectory) == 2 else 'Quadratic (ax² + bx + c)'}")
            else:
                print(f"Drone at {drone.coordinate} - Insufficient data for prediction")

    drone_localization_generator.add_event_listener(callback)

    print("Starting drone localization generator...")
    print("Using polynomial fitting: Linear (2 points) or Quadratic (3+ points)")
    
    while True:
        schedule.run_pending()  
        time.sleep(0.1)