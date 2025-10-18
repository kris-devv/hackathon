from datetime import datetime
import numpy as np
import pandas as pd
from fixed_sized_queue import FixedSizeQueue
from drone_position_raport import DronePositionRaport

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
        self.coordinate: list[float] = [x, y]
        self.trajectory: list[list[float]] = [[x, y]]
        self.current_direction: float = 0.0
        self.is_active: bool = True
        self.last_known_positions: FixedSizeQueue = FixedSizeQueue(max_size=10)
        self.last_known_positions.push(DronePositionRaport(x, y))

    def move(self, x: float, y: float):
        """
        Move the drone by adding displacement to current coordinates.
        Updates trajectory and calculates current direction.
        
        Args:
            x (float): Displacement in X direction
            y (float): Displacement in Y direction
        """
        self.last_known_positions.push(DronePositionRaport(self.coordinate[0], self.coordinate[1]))
        self.coordinate = [self.coordinate[0] + x, self.coordinate[1] + y]
        self.trajectory.append(self.coordinate.copy())
        
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
    
    def get_trajectory(self) -> list[list[float]]:
        """
        Get the complete trajectory history.
        
        Returns:
            list[list[float]]: List of all positions [x, y] the drone has visited
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
            return np.polyfit([p[0] for p in points], [p[1] for p in points], 1)
        else:
            return np.polyfit([p[0] for p in points], [p[1] for p in points], 2)

    def estimate_current_velocity(self):
        if len(self.last_known_positions) == 0:
            return None

        # Oblicz różnicę pozycji w osiach X i Y
        delta_x = self.coordinate[0] - self.last_known_positions.get_items()[-1].x
        delta_y = self.coordinate[1] - self.last_known_positions.get_items()[-1].y
        
        # Oblicz całkowite przemieszczenie używając twierdzenia Pitagorasa
        position_delta = np.sqrt(delta_x**2 + delta_y**2)

        time_delta = (datetime.now() - self.last_known_positions.get_items()[-1].timestamp).total_seconds()
        if time_delta == 0:
            None

        # Oblicz prędkość w każdej osi
        velocity_x = delta_x / time_delta
        velocity_y = delta_y / time_delta
        
        # Całkowita prędkość (moduł wektora prędkości)
        total_velocity = position_delta / time_delta

        return total_velocity