import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from drone import Drone
from predict import Prediction
from flight_path_generator import generate_flight_path

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

    def animate_multiple_drones(self, num_drones: int = 5, spawn_position: str = 'right',
                               interval: float = 0.02, show_trajectory: bool = True,
                               show_direction: bool = True, show_prediction: bool = True,
                               speed: float = 10, barrier_x: float = 900, 
                               city_manager=None, radar_manager=None):
        """
        Animate multiple drones simultaneously with polynomial trajectory prediction, city zones and radar detection.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.canvas.manager.set_window_title('Drone Trajectory Monitoring System')
        
        total_moves = 0
        running = True
        all_trajectories = []
        animation_complete = False
        
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
                        
                        # Check boundaries and barrier
                        if drone.is_out_of_bounds(self.width, self.height, margin=0) or drone.coordinate[0] <= barrier_x:
                            print(f"\n{'*'*60}")
                            print(f"DRONE REMOVED (color: {drone.drone_color})")
                            print(f"Final position: ({drone.coordinate[0]:.2f}, {drone.coordinate[1]:.2f})")
                            print(f"Total trajectory points: {len(drone.trajectory)}")
                            print(f"Reason: {'Out of bounds' if drone.is_out_of_bounds(self.width, self.height, margin=0) else f'Crossed barrier at x={barrier_x}'}")
                            
                            if len(drone.trajectory) >= 2:
                                start_pos = drone.trajectory[0]
                                end_pos = drone.trajectory[-1]
                                distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                                print(f"Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
                                print(f"End position: ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
                                print(f"Total distance traveled: {distance:.2f} units")
                            
                            print(f"Last 5 trajectory points:")
                            for i, point in enumerate(drone.trajectory[-5:], 1):
                                print(f"  {i}. ({point[0]:.2f}, {point[1]:.2f})")
                            
                            print(f"{'*'*60}\n")
                            
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
                                   color=drone.drone_color, alpha=0.6, linewidth=2)
                            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                                   'o', color=drone.drone_color, markersize=4, alpha=0.5)
                        
                        # Show polynomial prediction
                        if show_prediction and len(drone.trajectory) >= 3:
                            drone_df = drone.get_trajectory_dataframe()
                            prediction = self.predictor.predict(drone_df, steps_ahead=5)
                            
                            if prediction and 'points' in prediction:
                                pred_points = prediction['points']
                                pred_x = [p['x'] for p in pred_points]
                                pred_y = [p['y'] for p in pred_points]
                                
                                print(f"\n{'='*60}")
                                print(f"Drone (color: {drone.drone_color}) at position: ({drone.coordinate[0]:.2f}, {drone.coordinate[1]:.2f})")
                                print(f"Current trajectory length: {len(drone.trajectory)} points")
                                
                                if 'coefficients' in prediction:
                                    coeffs = prediction['coefficients']
                                    if len(coeffs) == 2:
                                        print(f"Polynomial type: Linear (y = {coeffs[0]:.4f}x + {coeffs[1]:.4f})")
                                    else:
                                        print(f"Polynomial type: Quadratic (y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f})")
                                    print(f"Coefficients: {coeffs}")
                                
                                print(f"Predicted points:")
                                for i, point in enumerate(pred_points, 1):
                                    print(f"  {i}. x={point['x']:.2f}, y={point['y']:.2f}")
                                print(f"{'='*60}\n")
                                
                                ax.plot(pred_x, pred_y, 
                                       color=drone.drone_color, alpha=0.5, 
                                       linewidth=2, linestyle=':', marker='x', markersize=10,
                                       markeredgewidth=2, label=f'Prediction (poly)')
                                
                                if 'coefficients' in prediction:
                                    from numpy import poly1d
                                    coeffs = prediction['coefficients']
                                    poly_func = poly1d(coeffs)
                                    
                                    x_range = np.linspace(drone.trajectory[-5][0] if len(drone.trajectory) >= 5 else drone.trajectory[0][0], 
                                                         pred_x[-1], 50)
                                    y_range = poly_func(x_range)
                                    
                                    ax.plot(x_range, y_range, 
                                           color=drone.drone_color, alpha=0.3, 
                                           linewidth=1, linestyle='--')
                        
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
                                    head_width=15, head_length=15, fc=drone.drone_color, 
                                    ec='white', linewidth=1, alpha=0.6)
                    
                    if not any_active:
                        animation_complete = True
                        print(f"\n{'#'*60}")
                        print(f"ANIMATION COMPLETED")
                        print(f"Total drones spawned: {self.drone_counter}")
                        print(f"Total moves executed: {total_moves}")
                        print(f"Total trajectories saved: {len(all_trajectories)}")
                        print(f"\nTrajectory summary:")
                        for i, traj_data in enumerate(all_trajectories, 1):
                            traj = traj_data['trajectory']
                            color = traj_data['color']
                            if len(traj) >= 2:
                                distance = np.sqrt((traj[-1][0] - traj[0][0])**2 + (traj[-1][1] - traj[0][1])**2)
                                print(f"  Drone {i} ({color}): {len(traj)} points, distance: {distance:.2f} units")
                        print(f"{'#'*60}\n")
                
                # Update and draw city zones if manager is provided
                if city_manager:
                    city_manager.update_all_zones(self.active_drones)
                    city_manager.draw_all_zones(ax)
                
                # Update and draw radar zones if manager is provided
                if radar_manager:
                    radar_manager.update_all_radars(self.active_drones)
                    radar_manager.draw_all_radars(ax)
                    
                    # Print radar status periodically
                    if total_moves % 100 == 0 and total_moves > 0:
                        radar_manager.print_status()
                
                # Print combined statistics
                if total_moves % 50 == 0 and (city_manager or radar_manager):
                    print(f"\n[Move {total_moves}]")
                    if city_manager:
                        stats = city_manager.get_zone_statistics()
                        occupied_zones = city_manager.get_occupied_zones()
                        if occupied_zones:
                            print(f"City Zones - Occupied: {stats['occupied_zones']}/{stats['total_zones']}")
                    
                    if radar_manager:
                        stats = radar_manager.get_radar_statistics()
                        detecting = radar_manager.get_detecting_radars()
                        if detecting:
                            print(f"Radars - Detecting: {stats['detecting_radars']}/{stats['total_radars']} ({stats['total_drones_detected']} drones)")
                
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
                    ax.set_title(f"Drone Trajectory Analysis - Completed (Polynomial Prediction)\nTotal Moves: {total_moves} | Past Trajectories: {len(all_trajectories)}\nPress 'q' to quit", 
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
        
        plt.ioff()
        plt.close()