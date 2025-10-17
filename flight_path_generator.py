import numpy as np

def generate_flight_path(pattern: str, num_moves: int = 100, speed: float = 8.0, 
                        start_angle: float = 180, **kwargs) -> list[tuple[float, float]]:
    """
    Generate a flight path for a drone based on specified pattern.
    """
    moves = []
    angle = start_angle
    
    if pattern == 'sine_wave':
        amplitude = kwargs.get('amplitude', 30)
        frequency = kwargs.get('frequency', 0.15)
        
        for i in range(num_moves):
            angle = start_angle + np.sin(i * frequency) * amplitude
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'zigzag':
        angle_change = kwargs.get('angle_change', 20)
        switch_interval = kwargs.get('switch_interval', 10)
        
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
        max_angle_change = kwargs.get('max_angle_change', 90)
        
        for i in range(num_moves):
            progress = i / num_moves
            angle = start_angle + (progress * max_angle_change) - (max_angle_change / 2)
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'random_walk':
        max_angle_change = kwargs.get('max_angle_change', 5)
        
        for i in range(num_moves):
            angle += np.random.uniform(-max_angle_change, max_angle_change)
            angle = np.clip(angle, start_angle - 45, start_angle + 45)
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'circle':
        radius = kwargs.get('radius', 5)
        
        for i in range(num_moves):
            angle = (i / num_moves) * 360
            angle_rad = np.radians(angle)
            dx = radius * np.cos(angle_rad)
            dy = radius * np.sin(angle_rad)
            moves.append((dx, dy))
    
    elif pattern == 'figure_eight':
        scale = kwargs.get('scale', 5)
        
        for i in range(num_moves):
            t = (i / num_moves) * 2 * np.pi
            dx = scale * np.sin(t)
            dy = scale * np.sin(t) * np.cos(t)
            moves.append((dx, dy))
    
    elif pattern == 'wavy_line':
        wave_amplitude = kwargs.get('wave_amplitude', 3)
        wave_frequency = kwargs.get('wave_frequency', 0.3)
        
        for i in range(num_moves):
            angle_rad = np.radians(start_angle)
            base_dx = speed * np.cos(angle_rad)
            base_dy = speed * np.sin(angle_rad)
            wave_offset = np.sin(i * wave_frequency) * wave_amplitude
            perpendicular_angle = np.radians(start_angle + 90)
            dx = base_dx + wave_offset * np.cos(perpendicular_angle)
            dy = base_dy + wave_offset * np.sin(perpendicular_angle)
            moves.append((dx, dy))
    
    elif pattern == 'fully_random':
        for i in range(num_moves):
            angle = np.random.uniform(0, 360)
            angle_rad = np.radians(angle)
            dx = speed * np.cos(angle_rad)
            dy = speed * np.sin(angle_rad)
            moves.append((dx, dy))
    
    else:
        angle_rad = np.radians(start_angle)
        dx = speed * np.cos(angle_rad)
        dy = speed * np.sin(angle_rad)
        moves = [(dx, dy)] * num_moves
    
    return moves