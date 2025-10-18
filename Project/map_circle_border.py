import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class RadarZone:
    """
    Represents a radar detection zone with a center point and detection circle.
    """
    
    def __init__(self, x: float, y: float, radius: float, name: str = "Radar"):
        """
        Initialize a radar zone.
        
        Args:
            x (float): X coordinate of radar center
            y (float): Y coordinate of radar center
            radius (float): Detection radius
            name (str): Name/ID of the radar (default "Radar")
        """
        self.name = name
        self.center = (x, y)
        self.radius = radius
        self.is_detecting = False
        self.drones_detected = []  # Track which drones are inside
        self.detection_count = 0  # Total number of detections
        
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
    
    def update_detection(self, drones: list):
        """
        Update detection status based on drone positions.
        
        Args:
            drones (list): List of active drone objects
        """
        previous_count = len(self.drones_detected)
        self.drones_detected = []
        
        for drone in drones:
            if self.check_drone_inside(drone.coordinate[0], drone.coordinate[1]):
                self.drones_detected.append(drone)
        
        # Update detection status
        self.is_detecting = len(self.drones_detected) > 0
        
        # Increment counter if new detections
        if len(self.drones_detected) > previous_count:
            self.detection_count += (len(self.drones_detected) - previous_count)
    
    def get_current_color(self) -> str:
        """
        Get current color based on detection status.
        
        Returns:
            str: Color name ('green' if clear, 'red' if detecting)
        """
        return 'red' if self.is_detecting else 'green'
    
    def draw(self, ax):
        """
        Draw the radar zone on the plot.
        
        Args:
            ax: Matplotlib axes object
        """
        current_color = self.get_current_color()
        
        # Draw center point (radar position)
        ax.plot(self.center[0], self.center[1], 'o', 
               color=current_color, markersize=10, 
               markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        # Draw detection circle
        circle = patches.Circle(self.center, self.radius, 
                               fill=False, edgecolor=current_color, 
                               linewidth=3, linestyle='-', alpha=0.7, zorder=9)
        ax.add_patch(circle)
        
        # Draw filled circle with low alpha for detection area
        if self.is_detecting:
            filled_circle = patches.Circle(self.center, self.radius, 
                                          fill=True, facecolor='red', 
                                          alpha=0.15, zorder=8)
            ax.add_patch(filled_circle)
        
        # Draw radar name label
        ax.text(self.center[0], self.center[1] - self.radius - 20, 
               self.name, fontsize=10, fontweight='bold', 
               color=current_color, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=current_color, alpha=0.8, linewidth=2))
        
        # If detecting, show drone count and alert
        if self.is_detecting:
            ax.text(self.center[0], self.center[1] + self.radius + 20, 
                   f'⚠️ ALERT: {len(self.drones_detected)} drone(s)', 
                   fontsize=9, fontweight='bold', color='red', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                            edgecolor='red', alpha=0.9, linewidth=2))
            
            # Draw detection lines from radar to drones
            for drone in self.drones_detected:
                ax.plot([self.center[0], drone.coordinate[0]], 
                       [self.center[1], drone.coordinate[1]], 
                       color='red', linewidth=1.5, linestyle='--', alpha=0.6, zorder=7)
        
        # Draw radius label
        ax.text(self.center[0] + self.radius * 0.7, self.center[1] + self.radius * 0.7, 
               f'R={self.radius:.0f}', fontsize=8, color=current_color, 
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        edgecolor=current_color, alpha=0.6))


class RadarBorderManager:
    """
    Manages multiple radar detection zones.
    """
    
    def __init__(self):
        """
        Initialize the radar border manager.
        """
        self.radars = []
    
    def add_radar(self, x: float, y: float, radius: float, name: str = None):
        """
        Add a new radar zone.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            radius (float): Detection radius
            name (str): Name/ID of the radar (optional)
        
        Returns:
            RadarZone: Created radar zone object
        """
        if name is None:
            name = f"Radar-{len(self.radars) + 1}"
        
        radar = RadarZone(x, y, radius, name)
        self.radars.append(radar)
        print(f"✓ Added {name} at ({x}, {y}) with radius {radius}")
        return radar
    
    def add_radars_from_list(self, radar_configs: list):
        """
        Add multiple radars from a list.
        
        Args:
            radar_configs (list): List of tuples (x, y, radius) or (x, y, radius, name)
        """
        for config in radar_configs:
            if len(config) == 3:
                x, y, radius = config
                self.add_radar(x, y, radius)
            elif len(config) == 4:
                x, y, radius, name = config
                self.add_radar(x, y, radius, name)
    
    def update_all_radars(self, drones: list):
        """
        Update all radars with current drone positions.
        
        Args:
            drones (list): List of active drone objects
        """
        for radar in self.radars:
            radar.update_detection(drones)
    
    def draw_all_radars(self, ax):
        """
        Draw all radars on the plot.
        
        Args:
            ax: Matplotlib axes object
        """
        for radar in self.radars:
            radar.draw(ax)
    
    def get_detecting_radars(self) -> list:
        """
        Get list of radars currently detecting drones.
        
        Returns:
            list: List of detecting RadarZone objects
        """
        return [radar for radar in self.radars if radar.is_detecting]
    
    def get_radar_statistics(self) -> dict:
        """
        Get statistics about all radars.
        
        Returns:
            dict: Statistics including total radars, detecting radars, etc.
        """
        detecting = self.get_detecting_radars()
        total_detections = sum(len(radar.drones_detected) for radar in self.radars)
        total_detection_count = sum(radar.detection_count for radar in self.radars)
        
        return {
            'total_radars': len(self.radars),
            'detecting_radars': len(detecting),
            'clear_radars': len(self.radars) - len(detecting),
            'total_drones_detected': total_detections,
            'total_detection_count': total_detection_count
        }
    
    def print_status(self):
        """
        Print current status of all radars.
        """
        stats = self.get_radar_statistics()
        print(f"\n{'='*60}")
        print(f"RADAR SYSTEM STATUS")
        print(f"Total Radars: {stats['total_radars']}")
        print(f"Detecting: {stats['detecting_radars']} | Clear: {stats['clear_radars']}")
        print(f"Current Detections: {stats['total_drones_detected']} drone(s)")
        print(f"Total Detection Count: {stats['total_detection_count']}")
        
        detecting_radars = self.get_detecting_radars()
        if detecting_radars:
            print(f"\n⚠️  ACTIVE DETECTIONS:")
            for radar in detecting_radars:
                print(f"  - {radar.name}: {len(radar.drones_detected)} drone(s) at ({radar.center[0]}, {radar.center[1]})")
        else:
            print(f"\n✓ All radars clear")
        print(f"{'='*60}\n")