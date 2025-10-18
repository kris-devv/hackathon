import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from drone import Drone
from predict import Prediction
from flight_path_generator import generate_flight_path
from save_data import DroneDataCollector

class Map:
    """
    Represents a 2D map with optional background image and coordinate system.
    """
    
    def __init__(self, width: int = None, height: int = None, image_path: str = None, 
                 rotation: int = 0, flip_horizontal: bool = False, flip_vertical: bool = False,
                 target_width: int = 1920, target_height: int = 1080):
        """
        Initialize the map with optional image and transformations.
        """
        self.rotation = rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.target_width = target_width
        self.target_height = target_height
        self.active_drones: list[Drone] = []
        self.drone_counter: int = 0
        self.predictor = Prediction()
        
        if image_path:
            self.load_image(image_path)
            self.width = float(self.target_width)
            self.height = float(self.target_height)
        else:
            self.width = width if width is not None else self.target_width
            self.height = height if height is not None else self.target_height
            self.grid = np.zeros((int(self.height), int(self.width)))
            self.image = None
    
    def load_image(self, image_path: str):
        """
        Load and process an image file with specified transformations.
        """
        img = Image.open(image_path)
        
        if self.rotation != 0:
            if self.rotation in [90, 180, 270, -90]:
                img = img.rotate(self.rotation, expand=True)
        
        if self.flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        img = img.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(img.convert('L'))
        self.image = img_array
        self.grid = self.image
        
    def pixel_to_coordinates(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        """
        Convert pixel coordinates to Cartesian coordinates.
        """
        image_height, image_width = self.grid.shape
        x_coord = (pixel_x / image_width) * self.width
        y_coord_from_bottom_pixel = image_height - 1 - pixel_y
        y_coord = (y_coord_from_bottom_pixel / image_height) * self.height
        return (x_coord, y_coord)
    
    def coordinates_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """
        Convert Cartesian coordinates to pixel coordinates.
        """
        image_height, image_width = self.grid.shape
        pixel_x = int((x / self.width) * image_width)
        pixel_y_from_bottom = int((y / self.height) * image_height)
        pixel_y = image_height - 1 - pixel_y_from_bottom
        pixel_x = np.clip(pixel_x, 0, image_width - 1)
        pixel_y = np.clip(pixel_y, 0, image_height - 1)
        return (pixel_x, pixel_y)
        
    def display(self, drones: list = None):
        """
        Display the map with optional drone positions.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.grid, cmap='gray', extent=[0, self.width, 0, self.height])
        
        if drones:
            for drone in drones:
                plt.plot(drone.coordinate[0], drone.coordinate[1], 
                        'o', color=drone.drone_color, markersize=10, 
                        markeredgecolor='white', markeredgewidth=2)
        
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def spawn_new_drone(self, position: str = 'right') -> Drone:
        """
        Spawn a new drone at specified edge position.
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
        """
        if drone in self.active_drones:
            self.active_drones.remove(drone)
            drone.is_active = False
    
    def get_drone_count(self) -> int:
        """
        Get the number of active drones.
        
        Returns:
            int: Number of active drones
        """
        return len(self.active_drones)
    
    def get_all_drones(self) -> list[Drone]:
        """
        Get list of all active drones.
        
        Returns:
            list[Drone]: List of active drone objects
        """
        return self.active_drones.copy()
    
    def clear_all_drones(self):
        """
        Remove all drones from the map.
        """
        for drone in self.active_drones[:]:
            self.remove_drone(drone)
    
    def get_drones_in_area(self, x_min: float, y_min: float, x_max: float, y_max: float) -> list[Drone]:
        """
        Get all drones within a rectangular area.
        
        Args:
            x_min (float): Minimum X coordinate
            y_min (float): Minimum Y coordinate
            x_max (float): Maximum X coordinate
            y_max (float): Maximum Y coordinate
            
        Returns:
            list[Drone]: List of drones in the specified area
        """
        drones_in_area = []
        for drone in self.active_drones:
            x, y = drone.coordinate
            if x_min <= x <= x_max and y_min <= y <= y_max:
                drones_in_area.append(drone)
        return drones_in_area
    
    def animate_multiple_drones(self, num_drones: int = 5, spawn_position: str = 'right', 
                               interval: float = 0.02, show_trajectory: bool = True, 
                               show_direction: bool = True, show_prediction: bool = True,
                               speed: float = 10, barrier_x: float = 750, 
                               city_manager=None, radar_manager=None,
                               save_data: bool = True, output_file: str = "data.json"):
        """
        Animate multiple drones simultaneously with polynomial trajectory prediction, city zones and radar detection.
        """
        # Initialize data collector
        data_collector = DroneDataCollector(output_file) if save_data else None
        
        plt.ion()
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.canvas.manager.set_window_title('Drone Trajectory Monitoring System')
        
        total_moves = 0
        running = True
        all_trajectories = []
        animation_complete = False
        
        drones_data = []
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        
        # Tracking for zone visits
        drone_zone_tracking = {}
        
        for i in range(num_drones):
            drone = self.spawn_new_drone(spawn_position)
            drone.drone_color = colors[i % len(colors)]
            moves = generate_flight_path('random_walk', num_moves=100, speed=speed, 
                                        start_angle=180, max_angle_change=8)
            
            # Add drone to data collector
            drone_id = data_collector.add_drone(drone, drone.drone_color) if data_collector else i
            
            drones_data.append({
                'drone': drone,
                'drone_id': drone_id,
                'moves': moves,
                'move_index': 0,
                'active': True,
                'color': drone.drone_color
            })
            
            # Initialize zone tracking
            drone_zone_tracking[drone_id] = {
                'current_cities': set(),
                'current_radars': set()
            }
            
            if data_collector:
                data_collector.add_event("drone_spawned", {
                    "drone_id": drone_id,
                    "color": drone.drone_color,
                    "position": {"x": drone.coordinate[0], "y": drone.coordinate[1]},
                    "spawn_edge": spawn_position
                })
        
        def on_key(event):
            nonlocal running
            if event.key == 'q':
                running = False
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        while running:
            try:
                ax.clear()
                ax.imshow(self.grid, cmap='gray', extent=[0, self.width, 0, self.height], aspect='auto')
                
                # Draw past trajectories
                for traj_data in all_trajectories:
                    trajectory_array = np.array(traj_data['trajectory'])
                    ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                           color=traj_data['color'], alpha=0.4, linewidth=3, linestyle='-')
                    ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                           'o', color=traj_data['color'], markersize=4, alpha=0.3)
                
                any_active = False
                
                if not animation_complete:
                    for drone_data in drones_data:
                        if not drone_data['active']:
                            continue
                        
                        any_active = True
                        drone = drone_data['drone']
                        drone_id = drone_data['drone_id']
                        moves = drone_data['moves']
                        move_index = drone_data['move_index']
                        
                        if move_index >= len(moves):
                            moves = generate_flight_path('random_walk', num_moves=100, 
                                                        speed=speed, start_angle=180, 
                                                        max_angle_change=8)
                            drone_data['moves'] = moves
                            move_index = 0
                        
                        dx, dy = moves[move_index]
                        drone.move(dx, dy)
                        total_moves += 1
                        drone_data['move_index'] = move_index + 1
                        
                        # Update position in data collector
                        if data_collector:
                            data_collector.update_drone_position(
                                drone_id,
                                drone.coordinate[0],
                                drone.coordinate[1],
                                speed=drone.get_speed(),
                                direction=drone.get_direction()
                            )
                        
                        # Check zone visits
                        if city_manager and data_collector:
                            current_cities = set()
                            for zone in city_manager.zones:
                                if zone.check_drone_inside(drone.coordinate[0], drone.coordinate[1]):
                                    current_cities.add(zone.name)
                                    # Check if this is a new entry
                                    if zone.name not in drone_zone_tracking[drone_id]['current_cities']:
                                        data_collector.add_zone_visit(drone_id, "city", zone.name)
                            drone_zone_tracking[drone_id]['current_cities'] = current_cities
                        
                        if radar_manager and data_collector:
                            current_radars = set()
                            for radar in radar_manager.radars:
                                if radar.check_drone_inside(drone.coordinate[0], drone.coordinate[1]):
                                    current_radars.add(radar.name)
                                    # Check if this is a new detection
                                    if radar.name not in drone_zone_tracking[drone_id]['current_radars']:
                                        data_collector.add_zone_visit(drone_id, "radar", radar.name)
                                        data_collector.add_event("radar_detection", {
                                            "drone_id": drone_id,
                                            "radar_name": radar.name,
                                            "position": {"x": drone.coordinate[0], "y": drone.coordinate[1]}
                                        })
                            drone_zone_tracking[drone_id]['current_radars'] = current_radars
                        
                        # Check boundaries and barrier
                        if drone.is_out_of_bounds(self.width, self.height, margin=0) or drone.coordinate[0] <= barrier_x:
                            reason = 'Out of bounds' if drone.is_out_of_bounds(self.width, self.height, margin=0) else f'Crossed barrier at x={barrier_x}'
                            
                            if data_collector:
                                data_collector.remove_drone(
                                    drone_id,
                                    drone.coordinate[0],
                                    drone.coordinate[1],
                                    reason=reason,
                                    total_distance=drone.get_total_distance(),
                                    average_speed=drone.get_average_speed()
                                )
                            
                            all_trajectories.append({
                                'trajectory': drone.trajectory.copy(),
                                'color': drone.drone_color
                            })
                            self.remove_drone(drone)
                            drone_data['active'] = False
                            continue
                        
                        # Show current trajectory
                        if show_trajectory and len(drone.trajectory) > 1:
                            trajectory_array = np.array(drone.trajectory)
                            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                                   color=drone.drone_color, alpha=0.8, linewidth=3)
                            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                                   'o', color=drone.drone_color, markersize=5, alpha=0.7)
                        
                        # Show polynomial prediction
                        if show_prediction and len(drone.trajectory) >= 3:
                            drone_df = drone.get_trajectory_dataframe()
                            prediction = self.predictor.predict(drone_df, steps_ahead=5)
                            
                            if prediction and 'points' in prediction:
                                pred_points = prediction['points']
                                pred_x = [p['x'] for p in pred_points]
                                pred_y = [p['y'] for p in pred_points]
                                
                                # Save prediction to data collector
                                if data_collector and 'coefficients' in prediction:
                                    coeffs = prediction['coefficients']
                                    poly_type = 'Linear' if len(coeffs) == 2 else 'Quadratic'
                                    data_collector.add_prediction(
                                        drone_id,
                                        pred_points,
                                        coefficients=coeffs,
                                        polynomial_type=poly_type
                                    )
                                
                                # Draw predicted trajectory as dotted line WITHOUT markers
                                ax.plot(pred_x, pred_y, 
                                       color=drone.drone_color, alpha=0.7, 
                                       linewidth=3, linestyle=':', 
                                       label=f'Prediction (poly)')
                                
                                if 'coefficients' in prediction:
                                    from numpy import poly1d
                                    coeffs = prediction['coefficients']
                                    poly_func = poly1d(coeffs)
                                    
                                    x_range = np.linspace(drone.trajectory[-5][0] if len(drone.trajectory) >= 5 else drone.trajectory[0][0], 
                                                         pred_x[-1], 50)
                                    y_range = poly_func(x_range)
                                    
                                    ax.plot(x_range, y_range, 
                                           color=drone.drone_color, alpha=0.5, 
                                           linewidth=2, linestyle='--')
                        
                        # Draw drone
                        ax.plot(drone.coordinate[0], drone.coordinate[1], 
                               'o', color=drone.drone_color, markersize=12, 
                               markeredgecolor='white', markeredgewidth=2)
                        
                        # Show direction arrow
                        if show_direction:
                            dir_vec = drone.get_direction_vector()
                            arrow_length = 30
                            ax.arrow(drone.coordinate[0], drone.coordinate[1],
                                    dir_vec[0] * arrow_length, dir_vec[1] * arrow_length,
                                    head_width=15, head_length=15, fc='black', 
                                    ec='black', linewidth=2, alpha=0.9)
                    
                    if not any_active:
                        animation_complete = True
                        
                        if data_collector:
                            data_collector.finalize_simulation()
                            data_collector.save_to_file()
                            data_collector.export_summary("simulation_summary.txt")
                
                # Update and draw city zones if manager is provided
                if city_manager:
                    city_manager.update_all_zones(self.active_drones)
                    city_manager.draw_all_zones(ax)
                
                # Update and draw radar zones if manager is provided
                if radar_manager:
                    radar_manager.update_all_radars(self.active_drones)
                    radar_manager.draw_all_radars(ax)
                
                # Draw boundaries
                boundary_rect = plt.Rectangle((0, 0), self.width, self.height, 
                                             fill=False, edgecolor='green', linewidth=3, 
                                             linestyle='--')
                ax.add_patch(boundary_rect)
                
                # Set display parameters
                ax.set_xlim(0, self.width)
                ax.set_ylim(0, self.height)
                ax.set_xlabel("X Coordinate", fontsize=10)
                ax.set_ylabel("Y Coordinate", fontsize=10)
                
                # Update title
                if animation_complete:
                    ax.set_title(f"Drone Trajectory Analysis - Completed (Polynomial Prediction)\nTotal Moves: {total_moves} | Past Trajectories: {len(all_trajectories)}\nData saved to {output_file}\nPress 'q' to quit", 
                               fontsize=14, fontweight='bold', color='green')
                else:
                    active_count = sum(1 for d in drones_data if d['active'])
                    title_text = f"Drone Trajectory Monitoring System (Polynomial Prediction)\nActive Drones: {active_count}/{num_drones} | Total Moves: {total_moves} | Past Trajectories: {len(all_trajectories)}"
                    
                    if city_manager:
                        stats = city_manager.get_zone_statistics()
                        title_text += f" | Cities: {stats['occupied_zones']}/{stats['total_zones']}"
                    
                    if radar_manager:
                        stats = radar_manager.get_radar_statistics()
                        title_text += f" | Radars: {stats['detecting_radars']}/{stats['total_radars']}"
                    
                    title_text += "\nPress 'q' to quit"
                    ax.set_title(title_text, fontsize=12, fontweight='bold')
                
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.pause(interval)
                
                if not plt.fignum_exists(fig.number):
                    running = False
                    
            except KeyboardInterrupt:
                running = False
                break
        
        # Final save on exit
        if data_collector and not animation_complete:
            data_collector.finalize_simulation()
            data_collector.save_to_file()
            data_collector.export_summary("simulation_summary.txt")
        
        plt.ioff()
        plt.close()