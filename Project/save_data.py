import json
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

class DroneDataCollector:
    """
    Collects and saves drone simulation data to JSON file.
    """
    
    def __init__(self, output_file: str = "data.json"):
        """
        Initialize the data collector.
        
        Args:
            output_file (str): Name of the output JSON file
        """
        self.output_file = output_file
        self.simulation_data = {
            "simulation_info": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_drones": 0,
                "total_moves": 0,
                "total_trajectories": 0
            },
            "drones": [],
            "events": []
        }
        self.drone_id_counter = 0
        print(f"✓ DroneDataCollector initialized. Will save to: {self.output_file}")
    
    def add_drone(self, drone, drone_color: str) -> int:
        """
        Add a new drone to the data collection.
        
        Args:
            drone: Drone object
            drone_color (str): Color of the drone
            
        Returns:
            int: Assigned drone ID
        """
        self.drone_id_counter += 1
        drone_id = self.drone_id_counter
        
        drone_data = {
            "id": drone_id,
            "color": drone_color,
            "spawn_position": {
                "x": float(drone.coordinate[0]),
                "y": float(drone.coordinate[1])
            },
            "spawn_time": datetime.now().isoformat(),
            "trajectory": [],
            "statistics": {
                "total_distance": 0.0,
                "average_speed": 0.0,
                "max_speed": 0.0,
                "total_moves": 0,
                "active_duration": 0.0
            },
            "predictions": [],
            "zones_visited": {
                "cities": [],
                "radars": []
            },
            "end_position": None,
            "end_time": None,
            "removal_reason": None
        }
        
        self.simulation_data["drones"].append(drone_data)
        self.simulation_data["simulation_info"]["total_drones"] += 1
        
        return drone_id
    
    def update_drone_position(self, drone_id: int, x: float, y: float, 
                             speed: float = None, direction: float = None):
        """
        Update drone position in trajectory.
        
        Args:
            drone_id (int): ID of the drone
            x (float): X coordinate
            y (float): Y coordinate
            speed (float): Current speed (optional)
            direction (float): Current direction in degrees (optional)
        """
        drone_data = self._get_drone_data(drone_id)
        if drone_data is None:
            return
        
        position_data = {
            "x": float(x),
            "y": float(y),
            "timestamp": datetime.now().isoformat()
        }
        
        if speed is not None:
            position_data["speed"] = float(speed)
        
        if direction is not None:
            position_data["direction"] = float(direction)
        
        drone_data["trajectory"].append(position_data)
        drone_data["statistics"]["total_moves"] += 1
        self.simulation_data["simulation_info"]["total_moves"] += 1
        
        # Update max speed
        if speed is not None and speed > drone_data["statistics"]["max_speed"]:
            drone_data["statistics"]["max_speed"] = float(speed)
    
    def add_prediction(self, drone_id: int, predicted_points: List[Dict], 
                      coefficients: List[float] = None, polynomial_type: str = None):
        """
        Add prediction data for a drone.
        
        Args:
            drone_id (int): ID of the drone
            predicted_points (list): List of predicted points
            coefficients (list): Polynomial coefficients
            polynomial_type (str): Type of polynomial (linear/quadratic)
        """
        drone_data = self._get_drone_data(drone_id)
        if drone_data is None:
            return
        
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "predicted_points": predicted_points,
            "polynomial_type": polynomial_type,
            "coefficients": [float(c) for c in coefficients] if coefficients else None
        }
        
        drone_data["predictions"].append(prediction_data)
    
    def add_zone_visit(self, drone_id: int, zone_type: str, zone_name: str, 
                      entry_time: str = None, exit_time: str = None):
        """
        Record when a drone visits a zone (city or radar).
        
        Args:
            drone_id (int): ID of the drone
            zone_type (str): Type of zone ('city' or 'radar')
            zone_name (str): Name of the zone
            entry_time (str): Entry timestamp
            exit_time (str): Exit timestamp (optional)
        """
        drone_data = self._get_drone_data(drone_id)
        if drone_data is None:
            return
        
        visit_data = {
            "zone_name": zone_name,
            "entry_time": entry_time or datetime.now().isoformat(),
            "exit_time": exit_time,
            "duration": None
        }
        
        if zone_type == "city":
            drone_data["zones_visited"]["cities"].append(visit_data)
        elif zone_type == "radar":
            drone_data["zones_visited"]["radars"].append(visit_data)
    
    def remove_drone(self, drone_id: int, end_x: float, end_y: float, 
                    reason: str, total_distance: float = None, 
                    average_speed: float = None):
        """
        Mark a drone as removed and record final statistics.
        
        Args:
            drone_id (int): ID of the drone
            end_x (float): Final X coordinate
            end_y (float): Final Y coordinate
            reason (str): Reason for removal
            total_distance (float): Total distance traveled
            average_speed (float): Average speed
        """
        drone_data = self._get_drone_data(drone_id)
        if drone_data is None:
            return
        
        drone_data["end_position"] = {
            "x": float(end_x),
            "y": float(end_y)
        }
        drone_data["end_time"] = datetime.now().isoformat()
        drone_data["removal_reason"] = reason
        
        if total_distance is not None:
            drone_data["statistics"]["total_distance"] = float(total_distance)
        
        if average_speed is not None:
            drone_data["statistics"]["average_speed"] = float(average_speed)
        
        # Calculate active duration
        if drone_data.get("spawn_time"):
            start = datetime.fromisoformat(drone_data["spawn_time"])
            end = datetime.fromisoformat(drone_data["end_time"])
            duration = (end - start).total_seconds()
            drone_data["statistics"]["active_duration"] = duration
        
        self.simulation_data["simulation_info"]["total_trajectories"] += 1
        
        # Add event
        self.add_event("drone_removed", {
            "drone_id": drone_id,
            "color": drone_data["color"],
            "reason": reason,
            "position": {"x": float(end_x), "y": float(end_y)}
        })
    
    def add_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Add a general event to the simulation log.
        
        Args:
            event_type (str): Type of event
            event_data (dict): Event details
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": event_data
        }
        self.simulation_data["events"].append(event)
    
    def finalize_simulation(self):
        """
        Finalize the simulation and mark end time.
        """
        self.simulation_data["simulation_info"]["end_time"] = datetime.now().isoformat()
        
        # Calculate total simulation duration
        start = datetime.fromisoformat(self.simulation_data["simulation_info"]["start_time"])
        end = datetime.fromisoformat(self.simulation_data["simulation_info"]["end_time"])
        duration = (end - start).total_seconds()
        self.simulation_data["simulation_info"]["total_duration_seconds"] = duration
        
        print(f"\n{'='*60}")
        print(f"SIMULATION FINALIZED")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total Drones: {self.simulation_data['simulation_info']['total_drones']}")
        print(f"Total Moves: {self.simulation_data['simulation_info']['total_moves']}")
        print(f"Total Trajectories: {self.simulation_data['simulation_info']['total_trajectories']}")
        print(f"{'='*60}\n")
    
    def save_to_file(self, pretty_print: bool = True):
        """
        Save collected data to JSON file.
        
        Args:
            pretty_print (bool): Whether to format JSON with indentation
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                if pretty_print:
                    json.dump(self.simulation_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(self.simulation_data, f, ensure_ascii=False)
            
            file_size = os.path.getsize(self.output_file)
            print(f"✓ Data saved to {self.output_file} ({file_size} bytes)")
            return True
        except Exception as e:
            print(f"✗ Error saving data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current simulation statistics.
        
        Returns:
            dict: Statistics summary
        """
        return {
            "total_drones": self.simulation_data["simulation_info"]["total_drones"],
            "total_moves": self.simulation_data["simulation_info"]["total_moves"],
            "total_trajectories": self.simulation_data["simulation_info"]["total_trajectories"],
            "total_events": len(self.simulation_data["events"]),
            "active_drones": sum(1 for d in self.simulation_data["drones"] if d["end_time"] is None)
        }
    
    def _get_drone_data(self, drone_id: int) -> Dict | None:
        """
        Get drone data by ID.
        
        Args:
            drone_id (int): ID of the drone
            
        Returns:
            dict | None: Drone data or None if not found
        """
        for drone_data in self.simulation_data["drones"]:
            if drone_data["id"] == drone_id:
                return drone_data
        print(f"Warning: Drone with ID {drone_id} not found")
        return None
    
    def export_summary(self, summary_file: str = "simulation_summary.txt"):
        """
        Export a human-readable summary of the simulation.
        
        Args:
            summary_file (str): Name of the summary text file
        """
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("DRONE SIMULATION SUMMARY\n")
                f.write("="*60 + "\n\n")
                
                info = self.simulation_data["simulation_info"]
                f.write(f"Start Time: {info['start_time']}\n")
                f.write(f"End Time: {info['end_time']}\n")
                if 'total_duration_seconds' in info:
                    f.write(f"Duration: {info['total_duration_seconds']:.2f} seconds\n")
                f.write(f"\nTotal Drones: {info['total_drones']}\n")
                f.write(f"Total Moves: {info['total_moves']}\n")
                f.write(f"Total Trajectories: {info['total_trajectories']}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("DRONE DETAILS\n")
                f.write("="*60 + "\n\n")
                
                for drone in self.simulation_data["drones"]:
                    f.write(f"Drone #{drone['id']} ({drone['color']})\n")
                    f.write(f"  Spawn: ({drone['spawn_position']['x']:.2f}, {drone['spawn_position']['y']:.2f})\n")
                    if drone['end_position']:
                        f.write(f"  End: ({drone['end_position']['x']:.2f}, {drone['end_position']['y']:.2f})\n")
                    f.write(f"  Moves: {drone['statistics']['total_moves']}\n")
                    f.write(f"  Distance: {drone['statistics']['total_distance']:.2f}\n")
                    f.write(f"  Avg Speed: {drone['statistics']['average_speed']:.2f}\n")
                    f.write(f"  Max Speed: {drone['statistics']['max_speed']:.2f}\n")
                    f.write(f"  Cities Visited: {len(drone['zones_visited']['cities'])}\n")
                    f.write(f"  Radars Detected: {len(drone['zones_visited']['radars'])}\n")
                    if drone['removal_reason']:
                        f.write(f"  Removal Reason: {drone['removal_reason']}\n")
                    f.write("\n")
            
            print(f"✓ Summary exported to {summary_file}")
            return True
        except Exception as e:
            print(f"✗ Error exporting summary: {e}")
            return False