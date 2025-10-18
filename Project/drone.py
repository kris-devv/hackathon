import numpy as np
import pandas as pd
from fixed_sized_queue import FixedSizeQueue
from numpy import polyfit

class Drone:
    """
    Represents a drone with position coordinates and movement capabilities.
    
    The Drone class tracks the position of a drone in a 2D coordinate system
    and allows it to move by applying displacement vectors.
    """
    
    def __init__(self, x: float, y: float):
        """
        Initialize a drone at specified coordinates.
        
        Args:
            x (float): Initial X coordinate
            y (float): Initial Y coordinate
        """
        self.drone_color: str = "blue"
        self.coordinate: tuple[float, float] = (x, y)
        self.trajectory: list[tuple[float, float]] = [(x, y)]
        self.current_direction: float = 0.0
        self.is_active: bool = True
        self.last_known_positions: FixedSizeQueue = FixedSizeQueue(max_size=10)
        self.last_known_positions.push(self.coordinate)

    def move(self, x: float, y: float):
        """
        Move the drone by adding displacement to current coordinates.
        Updates trajectory and calculates current direction.
        
        Args:
            x (float): Displacement in X direction
            y (float): Displacement in Y direction
        """
        self.coordinate = (self.coordinate[0] + x, self.coordinate[1] + y)
        self.trajectory.append(self.coordinate)
        self.last_known_positions.push(self.coordinate)
        
        if x != 0 or y != 0:
            self.current_direction = np.degrees(np.arctan2(y, x))
    
    def is_out_of_bounds(self, map_width: float, map_height: float, margin: float = 0) -> bool:
        """
        Check if drone is outside map boundaries with margin.
        
        Args:
            map_width (float): Width of the map
            map_height (float): Height of the map
            margin (float): Extra margin outside map boundaries
            
        Returns:
            bool: True if drone is out of bounds, False otherwise
        """
        x, y = self.coordinate
        return (x < -margin or x > map_width + margin or 
                y < -margin or y > map_height + margin)
    
    def get_trajectory(self) -> list[tuple[float, float]]:
        """
        Get the complete trajectory history.
        
        Returns:
            list[tuple[float, float]]: List of all positions (x, y) the drone has visited
        """
        return self.trajectory
    
    def get_direction(self) -> float:
        """
        Get the current direction of movement.
        
        Returns:
            float: Direction angle in degrees (0=right, 90=up, 180=left, 270=down)
        """
        return self.current_direction
    
    def get_direction_vector(self) -> tuple[float, float]:
        """
        Get the current direction as a unit vector.
        
        Returns:
            tuple[float, float]: (dx, dy) unit vector showing direction
        """
        angle_rad = np.radians(self.current_direction)
        return (np.cos(angle_rad), np.sin(angle_rad))
    
    def get_trajectory_dataframe(self) -> pd.DataFrame:
        """
        Get trajectory as pandas DataFrame for prediction.
        
        Returns:
            pd.DataFrame: DataFrame with 'x' and 'y' columns
        """
        return pd.DataFrame(self.trajectory, columns=['x', 'y'])
    
    def estimate_mathematical_trajectory(self):
        """
        Estimate mathematical trajectory using polynomial fitting.
        
        Returns:
            numpy array: Polynomial coefficients or None if insufficient data
        """
        points = self.last_known_positions.get_items()
        if len(points) < 2:
            return None

        if len(points) == 2:
            return polyfit([p[0] for p in points], [p[1] for p in points], 1)
        else:
            return polyfit([p[0] for p in points], [p[1] for p in points], 2)
    
    # NEW FUNCTIONS FROM drone_kopia.py
    
    def get_speed(self) -> float:
        """
        Calculate current speed based on last two positions.
        
        Returns:
            float: Speed in units per move, or 0 if insufficient data
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        last_pos = self.trajectory[-1]
        prev_pos = self.trajectory[-2]
        
        distance = np.sqrt((last_pos[0] - prev_pos[0])**2 + (last_pos[1] - prev_pos[1])**2)
        return distance
    
    def get_average_speed(self, last_n_moves: int = 10) -> float:
        """
        Calculate average speed over last N moves.
        
        Args:
            last_n_moves (int): Number of recent moves to consider
            
        Returns:
            float: Average speed in units per move
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        n = min(last_n_moves + 1, len(self.trajectory))
        recent_trajectory = self.trajectory[-n:]
        
        total_distance = 0.0
        for i in range(1, len(recent_trajectory)):
            curr_pos = recent_trajectory[i]
            prev_pos = recent_trajectory[i-1]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
        
        return total_distance / (n - 1) if n > 1 else 0.0
    
    def get_total_distance(self) -> float:
        """
        Calculate total distance traveled by the drone.
        
        Returns:
            float: Total distance in units
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.trajectory)):
            curr_pos = self.trajectory[i]
            prev_pos = self.trajectory[i-1]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
        
        return total_distance
    
    def get_position_at_time(self, time_index: int) -> tuple[float, float] | None:
        """
        Get drone position at specific time index.
        
        Args:
            time_index (int): Index in trajectory (0 = start)
            
        Returns:
            tuple[float, float] | None: Position at that time or None if invalid index
        """
        if 0 <= time_index < len(self.trajectory):
            return self.trajectory[time_index]
        return None
    
    def is_moving(self, threshold: float = 0.1) -> bool:
        """
        Check if drone is currently moving.
        
        Args:
            threshold (float): Minimum speed to consider as moving
            
        Returns:
            bool: True if drone is moving faster than threshold
        """
        return self.get_speed() > threshold
