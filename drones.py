import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from predict import Prediction

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
        self.trajectory: list[list[float]] = [[x, y]]  # Store full trajectory history
        self.current_direction: float = 0.0  # Current direction angle in degrees
        self.is_active: bool = True  # Track if drone is still within bounds

    def move(self, x: float, y: float):
        """
        Move the drone by adding displacement to current coordinates.
        Updates trajectory and calculates current direction.
        
        Args:
            x (float): Displacement in X direction
            y (float): Displacement in Y direction
        """
        self.coordinate = [self.coordinate[0] + x, self.coordinate[1] + y]
        self.trajectory.append(self.coordinate.copy())
        
        # Calculate current direction based on movement
        if x != 0 or y != 0:
            self.current_direction = np.degrees(np.arctan2(y, x))
    
    def is_out_of_bounds(self, map_width: float, map_height: float, margin: float = 0) -> bool:
        """
        Check if drone is outside map boundaries with margin.
        
        Args:
            map_width (float): Width of the map
            map_height (float): Height of the map
            margin (float): Extra margin outside map boundaries (default 0 for immediate removal)
            
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


class Map:
    """
    Represents a 2D map with optional background image and coordinate system.
    
    The Map class creates a Cartesian coordinate system based on an image's resolution
    or specified dimensions. It handles image loading, transformations (rotation, flipping),
    coordinate conversions between pixel and Cartesian systems, and visualization.
    """
    
    def __init__(self, width: int = None, height: int = None, image_path: str = None, 
                 rotation: int = 0, flip_horizontal: bool = False, flip_vertical: bool = False):
        """
        Initialize the map with optional image and transformations.
        
        Args:
            width (int, optional): Map width in units. If None and image_path provided, uses image width
            height (int, optional): Map height in units. If None and image_path provided, uses image height
            image_path (str, optional): Path to background image file
            rotation (int): Rotation angle in degrees (90, 180, 270, -90)
            flip_horizontal (bool): Whether to flip image horizontally (mirror)
            flip_vertical (bool): Whether to flip image vertically (upside down)
        """
        self.rotation = rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.active_drones: list[Drone] = []  # Track all active drones
        self.drone_counter: int = 0  # Counter for drone IDs
        self.predictor = Prediction()  # Initialize prediction system
        
        if image_path:
            self.load_image(image_path)
            self.width = width if width is not None else float(self.image.shape[1])
            self.height = height if height is not None else float(self.image.shape[0])
        else:
            self.width = width
            self.height = height
            self.grid = np.zeros((height, width))
            self.image = None
    
    def load_image(self, image_path: str):
        """
        Load and process an image file with specified transformations.
        
        Loads the image, applies rotation and flipping transformations,
        converts to grayscale, and stores as numpy array.
        
        Args:
            image_path (str): Path to the image file
        """
        img = Image.open(image_path)
        
        # Rotate image if specified
        if self.rotation != 0:
            if self.rotation in [90, 180, 270, -90]:
                img = img.rotate(self.rotation, expand=True)
        
        # Apply mirror transformations
        if self.flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        img_array = np.array(img.convert('L'))
        self.image = img_array
        self.grid = self.image
        
    def pixel_to_coordinates(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        """
        Convert pixel coordinates to Cartesian coordinates.
        
        Transforms image pixel positions (origin at top-left) to Cartesian
        coordinate system (origin at bottom-left) with float precision.
        
        Args:
            pixel_x (int): X position in pixels
            pixel_y (int): Y position in pixels (0 at top)
            
        Returns:
            tuple[float, float]: (x, y) coordinates in Cartesian system
        """
        image_height, image_width = self.grid.shape
    
        # Convert X coordinate
        x_coord = (pixel_x / image_width) * self.width
        
        # Convert Y coordinate: flip from top-origin to bottom-origin
        y_coord_from_bottom_pixel = image_height - 1 - pixel_y
        y_coord = (y_coord_from_bottom_pixel / image_height) * self.height
    
        return (x_coord, y_coord)
    
    def coordinates_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """
        Convert Cartesian coordinates to pixel coordinates.
        
        Transforms Cartesian coordinate system positions to image pixel
        positions, with bounds checking to ensure valid pixel coordinates.
        
        Args:
            x (float): X coordinate in Cartesian system
            y (float): Y coordinate in Cartesian system
            
        Returns:
            tuple[int, int]: (pixel_x, pixel_y) image coordinates
        """
        image_height, image_width = self.grid.shape
    
        pixel_x = int((x / self.width) * image_width)
        
        # Convert Y: from bottom-origin to top-origin
        pixel_y_from_bottom = int((y / self.height) * image_height)
        pixel_y = image_height - 1 - pixel_y_from_bottom
        
        # Ensure coordinates are within image bounds
        pixel_x = np.clip(pixel_x, 0, image_width - 1)
        pixel_y = np.clip(pixel_y, 0, image_height - 1)
        
        return (pixel_x, pixel_y)
        
    def display(self, drones: list = None):
        """
        Display the map with optional drone positions.
        
        Creates a matplotlib figure showing the map image with Cartesian
        coordinate system and plots drone positions if provided.
        
        Args:
            drones (list, optional): List of Drone objects to display on the map
        """
        plt.figure(figsize=(10, 8))
        
        # Display image with correct coordinate system
        plt.imshow(self.grid, cmap='gray', extent=[0, self.width, 0, self.height])
        
        # Add drones if provided
        if drones:
            for drone in drones:
                plt.plot(drone.coordinate[0], drone.coordinate[1], 
                        'o', color=drone.drone_color, markersize=10, 
                        markeredgecolor='white', markeredgewidth=2)
        
        # Set proper axis limits
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def spawn_new_drone(self, position: str = 'right') -> Drone:
        """
        Spawn a new drone at specified edge position.
        
        Args:
            position (str): Edge position - 'right', 'left', 'top', 'bottom', or 'random'
            
        Returns:
            Drone: Newly created drone object
        """
        self.drone_counter += 1
        
        if position == 'right':
            x = self.width - 50
            y = np.random.uniform(self.height * 0.2, self.height * 0.8)
        elif position == 'left':
            x = 50
            y = np.random.uniform(self.height * 0.2, self.height * 0.8)
        elif position == 'top':
            x = np.random.uniform(self.width * 0.2, self.width * 0.8)
            y = self.height - 50
        elif position == 'bottom':
            x = np.random.uniform(self.width * 0.2, self.width * 0.8)
            y = 50
        elif position == 'random':
            edge = np.random.choice(['right', 'left', 'top', 'bottom'])
            return self.spawn_new_drone(edge)
        else:
            x = self.width / 2
            y = self.height / 2
        
        new_drone = Drone(x, y)
        self.active_drones.append(new_drone)
        
        return new_drone
    
    def remove_drone(self, drone: Drone):
        """
        Remove a drone from active drones list.
        
        Args:
            drone (Drone): Drone object to remove
        """
        if drone in self.active_drones:
            self.active_drones.remove(drone)
            drone.is_active = False
    
    def animate_multiple_drones(self, num_drones: int = 5, spawn_position: str = 'right', 
                               interval: float = 0.02, show_trajectory: bool = True, 
                               show_direction: bool = True, show_prediction: bool = True,
                               speed: float = 10):
        """
        Animate multiple drones simultaneously with trajectory prediction.
        
        Creates an interactive animation showing multiple drones moving at the same time.
        When a drone exits the map, it is removed and its trajectory is saved.
        Uses prediction system to forecast future drone positions.
        Press 'q' to quit.
        
        Args:
            num_drones (int): Number of drones to spawn at start
            spawn_position (str): Position to spawn drones ('right', 'left', 'top', 'bottom', 'random')
            interval (float): Time delay between steps in seconds
            show_trajectory (bool): Whether to show full trajectory path
            show_direction (bool): Whether to show direction arrow
            show_prediction (bool): Whether to show predicted trajectory
            speed (float): Base movement speed per step
        """
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Variables to track statistics
        total_moves = 0
        running = True
        all_trajectories = []  # Store all trajectories from removed drones
        animation_complete = False  # Track if animation has finished
        
        # Spawn initial drones
        drones_data = []
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        
        for i in range(num_drones):
            drone = self.spawn_new_drone(spawn_position)
            drone.drone_color = colors[i % len(colors)]
            moves = generate_flight_path('random_walk', num_moves=100, speed=speed, 
                                        start_angle=180, max_angle_change=8)
            drones_data.append({
                'drone': drone,
                'moves': moves,
                'move_index': 0,
                'active': True,
                'color': drone.drone_color
            })
        
        def on_key(event):
            """Handle keyboard events"""
            nonlocal running
            if event.key == 'q':
                running = False
        
        # Connect keyboard event
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        while running:
            try:
                ax.clear()
                
                # Display image
                ax.imshow(self.grid, cmap='gray', extent=[0, self.width, 0, self.height])
                
                # Draw all previous trajectories (from removed drones)
                for traj_data in all_trajectories:
                    trajectory_array = np.array(traj_data['trajectory'])
                    # Draw line
                    ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                           color=traj_data['color'], alpha=0.4, linewidth=3, linestyle='-')
                    # Draw dots at each position
                    ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                           'o', color=traj_data['color'], markersize=4, alpha=0.3)
                
                # Check if any drones are still active
                any_active = False
                
                # Move all active drones (only if animation not complete)
                if not animation_complete:
                    for drone_data in drones_data:
                        if not drone_data['active']:
                            continue
                        
                        any_active = True
                        drone = drone_data['drone']
                        moves = drone_data['moves']
                        move_index = drone_data['move_index']
                        
                        # Check if we need new moves for this drone
                        if move_index >= len(moves):
                            moves = generate_flight_path('random_walk', num_moves=100, 
                                                        speed=speed, start_angle=180, 
                                                        max_angle_change=8)
                            drone_data['moves'] = moves
                            move_index = 0
                        
                        # Move drone
                        dx, dy = moves[move_index]
                        drone.move(dx, dy)
                        total_moves += 1
                        drone_data['move_index'] = move_index + 1
                        
                        # Check if drone is out of bounds
                        if drone.is_out_of_bounds(self.width, self.height, margin=0):
                            # Save trajectory before removing
                            all_trajectories.append({
                                'trajectory': drone.trajectory.copy(),
                                'color': drone.drone_color
                            })
                            self.remove_drone(drone)
                            drone_data['active'] = False
                            continue
                        
                        # Show current trajectory if enabled
                        if show_trajectory and len(drone.trajectory) > 1:
                            trajectory_array = np.array(drone.trajectory)
                            # Draw line
                            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                                   color=drone.drone_color, alpha=0.6, linewidth=2)
                            # Draw dots at each position
                            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                                   'o', color=drone.drone_color, markersize=4, alpha=0.5)
                        
                        # Show prediction if enabled and drone has enough history
                        if show_prediction and len(drone.trajectory) >= 5:
                            drone_df = drone.get_trajectory_dataframe()
                            prediction = self.predictor.predict(drone_df)
                            
                            if prediction and 'points' in prediction:
                                pred_points = prediction['points']
                                pred_x = [p['x'] for p in pred_points]
                                pred_y = [p['y'] for p in pred_points]
                                
                                # Draw predicted trajectory as dashed line
                                ax.plot(pred_x, pred_y, 
                                       color=drone.drone_color, alpha=0.4, 
                                       linewidth=2, linestyle=':', marker='x', markersize=8)
                        
                        # Draw drone
                        ax.plot(drone.coordinate[0], drone.coordinate[1], 
                               'o', color=drone.drone_color, markersize=12, 
                               markeredgecolor='white', markeredgewidth=2)
                        
                        # Show direction arrow if enabled
                        if show_direction:
                            dir_vec = drone.get_direction_vector()
                            arrow_length = 30
                            ax.arrow(drone.coordinate[0], drone.coordinate[1],
                                    dir_vec[0] * arrow_length, dir_vec[1] * arrow_length,
                                    head_width=15, head_length=15, fc=drone.drone_color, 
                                    ec='white', linewidth=1, alpha=0.6)
                    
                    # If no drones are active, mark animation as complete
                    if not any_active:
                        animation_complete = True
                
                # Draw boundaries
                boundary_rect = plt.Rectangle((0, 0), self.width, self.height, 
                                             fill=False, edgecolor='green', linewidth=3, 
                                             linestyle='--')
                ax.add_patch(boundary_rect)
                
                # Set display parameters
                ax.set_xlim(-100, self.width + 100)
                ax.set_ylim(-100, self.height + 100)
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")
                
                # Update title based on animation state
                if animation_complete:
                    ax.set_title(f"Drone Trajectory Analysis - Completed\nTotal Moves: {total_moves} | Past Trajectories: {len(all_trajectories)}\nPress 'q' to quit", 
                               fontsize=14, fontweight='bold', color='green')
                else:
                    # Count active drones
                    active_count = sum(1 for d in drones_data if d['active'])
                    ax.set_title(f"Drone Trajectory Monitoring System\nActive Drones: {active_count}/{num_drones} | Total Moves: {total_moves} | Past Trajectories: {len(all_trajectories)}\nPress 'q' to quit", 
                               fontsize=12, fontweight='bold')
                
                ax.grid(True, alpha=0.3)
                
                plt.pause(interval)
                
                # Check if window is still open
                if not plt.fignum_exists(fig.number):
                    running = False
                    
            except KeyboardInterrupt:
                running = False
                break
        
        plt.ioff()
        plt.close()


def generate_flight_path(pattern: str, num_moves: int = 100, speed: float = 8.0, 
                        start_angle: float = 180, **kwargs) -> list[tuple[float, float]]:
    """
    Generate a flight path for a drone based on specified pattern.
    
    Creates a sequence of movement vectors (dx, dy) that follow different
    mathematical patterns like sine waves, spirals, zigzags, etc. The function
    supports multiple flight patterns with customizable parameters.
    
    Args:
        pattern (str): Pattern type - 'sine_wave', 'zigzag', 'spiral', 'smooth_curve',
                      'random_walk', 'circle', 'figure_eight', 'wavy_line'
        num_moves (int): Number of movement steps to generate
        speed (float): Base movement speed per step
        start_angle (float): Initial flight direction in degrees (180=left, 0=right)
        **kwargs: Pattern-specific parameters:
            - amplitude (float): Wave amplitude for sine_wave
            - frequency (float): Wave frequency for sine_wave
            - angle_change (float): Angle deviation for zigzag
            - switch_interval (int): Steps between direction changes for zigzag
            - radius_increment (float): Radius growth rate for spiral
            - angular_speed (float): Rotation speed for spiral
            - max_angle_change (float): Maximum angle variation
            - radius (float): Circle radius
            - scale (float): Pattern scale factor
            - wave_amplitude (float): Wave height for wavy_line
            - wave_frequency (float): Wave density for wavy_line
    
    Returns:
        list[tuple[float, float]]: List of movement vectors (dx, dy)
    """
    moves = []
    angle = start_angle
    
    if pattern == 'sine_wave':
        # Sine wave - smooth oscillations
        amplitude = kwargs.get('amplitude', 30)  # Wave amplitude in degrees
        frequency = kwargs.get('frequency', 0.15)  # Oscillation frequency
        
        for i in range(num_moves):
            angle = start_angle + np.sin(i * frequency) * amplitude
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'zigzag':
        # Zigzag - sharp direction changes
        angle_change = kwargs.get('angle_change', 20)  # Angle deviation
        switch_interval = kwargs.get('switch_interval', 10)  # Steps per switch
        
        for i in range(num_moves):
            if (i // switch_interval) % 2 == 0:
                angle = start_angle + angle_change
            else:
                angle = start_angle - angle_change
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'spiral':
        # Spiral - gradually increasing radius
        radius_increment = kwargs.get('radius_increment', 0.5)
        angular_speed = kwargs.get('angular_speed', 0.2)
        
        for i in range(num_moves):
            current_radius = i * radius_increment
            angle = start_angle + i * angular_speed * 360 / num_moves
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad) + current_radius * 0.1
            moves.append((dx, dy))
    
    elif pattern == 'smooth_curve':
        # Smooth curve - gradual angle change
        max_angle_change = kwargs.get('max_angle_change', 90)
        
        for i in range(num_moves):
            progress = i / num_moves
            angle = start_angle + (progress * max_angle_change) - (max_angle_change / 2)
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'random_walk':
        # Random walk - unpredictable changes
        max_angle_change = kwargs.get('max_angle_change', 5)
        
        for i in range(num_moves):
            angle += np.random.uniform(-max_angle_change, max_angle_change)
            angle = np.clip(angle, start_angle - 45, start_angle + 45)
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'circle':
        # Circle - full circular path
        radius = kwargs.get('radius', 5)
        
        for i in range(num_moves):
            angle = (i / num_moves) * 360
            angle_rad = np.radians(angle)
            dx = radius * np.cos(angle_rad)
            dy = radius * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'figure_eight':
        # Figure eight - complex pattern
        scale = kwargs.get('scale', 5)
        
        for i in range(num_moves):
            t = (i / num_moves) * 2 * np.pi
            dx = scale * np.sin(t)
            dy = scale * np.sin(t) * np.cos(t)
            moves.append((dx, dy))
    
    elif pattern == 'wavy_line':
        # Wavy line - main direction with perpendicular waves
        wave_amplitude = kwargs.get('wave_amplitude', 3)
        wave_frequency = kwargs.get('wave_frequency', 0.3)
        
        for i in range(num_moves):
            # Main movement in start_angle direction
            angle_rad = np.radians(start_angle)
            base_dx = speed * np.cos(angle_rad)
            base_dy = speed * np.sin(angle_rad)
            
            # Add perpendicular wave to main direction
            wave_offset = np.sin(i * wave_frequency) * wave_amplitude
            perpendicular_angle = np.radians(start_angle + 90)
            
            dx = base_dx + wave_offset * np.cos(perpendicular_angle)
            dy = base_dy + wave_offset * np.sin(perpendicular_angle)
            moves.append((dx, dy))
    
    elif pattern == 'fully_random':
        # Fully random - random movement in all directions
        for i in range (num_moves):
            angle = np.random.uniform(0, 360)
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    else:
        # Default: straight line
        angle_rad = np.radians(start_angle)
        dx = speed * np.cos(angle_rad)
        dy = speed * np.sin(angle_rad)
        moves = [(dx, dy)] * num_moves
    
    return moves


if __name__ == "__main__":
    image_path = "/Users/sansrozpierducha/hackathon/polish_map.png" 
    
    map_with_image = Map(image_path=image_path, rotation=180, 
                        flip_horizontal=True, flip_vertical=True)
    
    # Animate 5 drones simultaneously with trajectory prediction
    # Slower speed (5 instead of 10) and slower interval (0.05 instead of 0.02)
    map_with_image.animate_multiple_drones(num_drones=5,
                                          spawn_position='right', 
                                          interval=0.05,
                                          show_trajectory=True, 
                                          show_direction=True,
                                          show_prediction=True,
                                          speed=25)