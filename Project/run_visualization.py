from map import Map

if __name__ == "__main__":
    image_path = "/Users/sansrozpierducha/hackathon/Project/polish_map.png" 
    
    # Initialize map with 1920x1080 resolution
    map_with_image = Map(
        image_path=image_path, 
        rotation=180, 
        flip_horizontal=True, 
        flip_vertical=True,
        target_width=1920, 
        target_height=1080
    )
    
    # Animate 5 drones simultaneously with trajectory prediction
    map_with_image.animate_multiple_drones(
        num_drones=1,
        spawn_position='right', 
        interval=0.05,
        show_trajectory=True, 
        show_direction=True,
        show_prediction=True,
        speed=25,
        barrier_x=750,
        vertical_line_x=1380
    )