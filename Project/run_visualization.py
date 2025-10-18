from map import Map
from map_circle_city import CityZoneManager
from map_circle_border import RadarBorderManager

if __name__ == "__main__":
    image_path = "Project/polish_map_project.png" 
    
    # Initialize map with 1920x1080 resolution
    map_with_image = Map(
        image_path=image_path, 
        rotation=180, 
        flip_horizontal=True, 
        flip_vertical=True,
        target_width=1920, 
        target_height=1080
    )
    
    # Create city zone manager
    city_manager = CityZoneManager()
    
    # Add cities with coordinates (name, x, y, radius)
    cities = [
        ("Warszawa", 1000, 610, 70),
        ("Lublin", 1230, 430, 70),
        ("Rzeszów", 1160, 205, 70),
        ("Białystok", 1280, 790, 70),
    ]
    city_manager.add_zones_from_list(cities)
    
    # Create radar border manager
    radar_manager = RadarBorderManager()
    
    # Add radars with coordinates (x, y, radius, name)
    # You can customize these coordinates
    radars = [
        (1280, 200, 180, "Border-Radar-1"),
        (1280, 550, 180, "Border-Radar-2"),
        (1280, 900, 180, "Border-Radar-3"),
    ]
    radar_manager.add_radars_from_list(radars)
    
    print("\n" + "="*60)
    print("SYSTEM INITIALIZED")
    print(f"Cities: {len(city_manager.zones)}")
    print(f"Radars: {len(radar_manager.radars)}")
    print("="*60 + "\n")
    
    # Animate 5 drones simultaneously with trajectory prediction, city zones and radars
    map_with_image.animate_multiple_drones(
        num_drones=5,
        spawn_position='right', 
        interval=0.05,
        show_trajectory=True, 
        show_direction=True,
        show_prediction=True,
        speed=25,
        barrier_x=900,
        city_manager=city_manager,
        radar_manager=radar_manager  # Pass radar manager to animation
    )