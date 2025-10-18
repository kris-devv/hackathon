import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class CityZone:
    """
    Represents a city zone with a center point and detection circle.
    """
    
    def __init__(self, name: str, x: float, y: float, radius: float = 10):
        """
        Initialize a city zone.
        
        Args:
            name (str): Name of the city
            x (float): X coordinate of city center
            y (float): Y coordinate of city center
            radius (float): Detection radius (default 10)
        """
        self.name = name
        self.center = (x, y)
        self.radius = radius
        self.is_occupied = False
        self.blink_state = False  # For blinking animation
        self.drones_inside = []  # Track which drones are inside
        
    def check_drone_inside(self, drone_x: float, drone_y: float) -> bool:
        """
        Check if a drone is inside the detection circle.
        
        Args:
            drone_x (float): Drone X coordinate
            drone_y (float): Drone Y coordinate
            
        Returns:
            bool: True if drone is inside the circle
        """
        distance = np.sqrt((drone_x - self.center[0])**2 + (drone_y - self.center[1])**2)
        return distance <= self.radius
    
    def update_occupation(self, drones: list):
        """
        Update occupation status based on drone positions.
        
        Args:
            drones (list): List of active drone objects
        """
        self.drones_inside = []
        for drone in drones:
            if self.check_drone_inside(drone.coordinate[0], drone.coordinate[1]):
                self.drones_inside.append(drone)
        
        self.is_occupied = len(self.drones_inside) > 0
        
        # Toggle blink state if occupied
        if self.is_occupied:
            self.blink_state = not self.blink_state
    
    def get_current_color(self) -> str:
        """
        Get current color based on occupation and blink state.
        
        Returns:
            str: Color name ('green' or 'red')
        """
        if not self.is_occupied:
            return 'green'
        return 'red' if self.blink_state else 'green'
    
    def draw(self, ax):
        """
        Draw the city zone on the plot.
        
        Args:
            ax: Matplotlib axes object
        """
        current_color = self.get_current_color()
        
        # Draw center point
        ax.plot(self.center[0], self.center[1], 'o', 
               color=current_color, markersize=8, 
               markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        
        # Draw detection circle
        circle = patches.Circle(self.center, self.radius, 
                               fill=False, edgecolor=current_color, 
                               linewidth=2.5, linestyle='-', alpha=0.8, zorder=9)
        ax.add_patch(circle)
        
        # Draw city name label
        ax.text(self.center[0], self.center[1] - self.radius - 15, 
               self.name, fontsize=9, fontweight='bold', 
               color=current_color, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=current_color, alpha=0.7))
        
        # If occupied, show drone count
        if self.is_occupied:
            ax.text(self.center[0], self.center[1] + self.radius + 15, 
                   f'{len(self.drones_inside)} drone(s)', 
                   fontsize=8, color=current_color, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                            edgecolor=current_color, alpha=0.6))


class CityZoneManager:
    """
    Manages multiple city zones on the map.
    """
    
    def __init__(self):
        """
        Initialize the city zone manager.
        """
        self.zones = []
    
    def add_zone(self, name: str, x: float, y: float, radius: float = 10):
        """
        Add a new city zone.
        
        Args:
            name (str): Name of the city
            x (float): X coordinate
            y (float): Y coordinate
            radius (float): Detection radius (default 10)
        """
        zone = CityZone(name, x, y, radius)
        self.zones.append(zone)
        return zone
    
    def add_zones_from_list(self, cities: list):
        """
        Add multiple zones from a list.
        
        Args:
            cities (list): List of tuples (name, x, y) or (name, x, y, radius)
        """
        for city_data in cities:
            if len(city_data) == 3:
                name, x, y = city_data
                self.add_zone(name, x, y)
            elif len(city_data) == 4:
                name, x, y, radius = city_data
                self.add_zone(name, x, y, radius)
    
    def update_all_zones(self, drones: list):
        """
        Update all zones with current drone positions.
        
        Args:
            drones (list): List of active drone objects
        """
        for zone in self.zones:
            zone.update_occupation(drones)
    
    def draw_all_zones(self, ax):
        """
        Draw all zones on the plot.
        
        Args:
            ax: Matplotlib axes object
        """
        for zone in self.zones:
            zone.draw(ax)
    
    def get_occupied_zones(self) -> list:
        """
        Get list of currently occupied zones.
        
        Returns:
            list: List of occupied CityZone objects
        """
        return [zone for zone in self.zones if zone.is_occupied]
    
    def get_zone_statistics(self) -> dict:
        """
        Get statistics about all zones.
        
        Returns:
            dict: Statistics including total zones, occupied zones, etc.
        """
        occupied = self.get_occupied_zones()
        total_drones_in_zones = sum(len(zone.drones_inside) for zone in self.zones)
        
        return {
            'total_zones': len(self.zones),
            'occupied_zones': len(occupied),
            'free_zones': len(self.zones) - len(occupied),
            'total_drones_in_zones': total_drones_in_zones
        }