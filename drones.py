import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Drone:
    def __init__(self, x: float, y: float):
        self.drone_color: str = "blue"
        self.coordinate: list[float] = [x, y]

    def move(self, x: float, y: float):
        self.move_coordinate_x = x
        self.move_coordinate_y = y
        self.coordinate = [self.coordinate[0] + self.move_coordinate_x, self.coordinate[1] + self.move_coordinate_y]


class Map:
    def __init__(self, width: int = None, height: int = None, image_path: str = None):
        """
        Inicjalizacja mapy z opcjonalnym obrazem
        :param width: szerokość mapy w jednostkach (opcjonalne, domyślnie rozdzielczość obrazu)
        :param height: wysokość mapy w jednostkach (opcjonalne, domyślnie rozdzielczość obrazu)
        :param image_path: ścieżka do pliku obrazu
        """
        if image_path:
            self.load_image(image_path)
            # Użyj rozdzielczości obrazu jako domyślnych wymiarów
            self.width = width if width is not None else float(self.image.shape[1])
            self.height = height if height is not None else float(self.image.shape[0])
        else:
            self.width = width
            self.height = height
            self.grid = np.zeros((height, width))
            self.image = None
    
    def load_image(self, image_path: str):
        """Wczytuje obraz i konwertuje do skali szarości"""
        img = Image.open(image_path)
        self.image = np.array(img.convert('L'))
        self.grid = self.image
        
    def pixel_to_coordinates(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        """
        Konwertuje współrzędne pikselowe na współrzędne kartezjańskie
        :param pixel_x: pozycja X w pikselach
        :param pixel_y: pozycja Y w pikselach (od góry)
        :return: (x, y) w układzie kartezjańskim
        """
        image_height, image_width = self.grid.shape
    
        # Przeliczenie pikseli na współrzędne float
        x_coord = (pixel_x / image_width) * self.width
        # Teraz y_coord odpowiada bezpośrednio pikselom (0 u góry)
        y_coord = (pixel_y / image_height) * self.height
    
        return (x_coord, y_coord)
    
    def coordinates_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """
        Konwertuje współrzędne kartezjańskie na współrzędne pikselowe
        :param x: pozycja X w układzie kartezjańskim
        :param y: pozycja Y w układzie kartezjańskim
        :return: (pixel_x, pixel_y)
        """
        image_height, image_width = self.grid.shape
    
        pixel_x = int((x / self.width) * image_width)
        pixel_y = int((y / self.height) * image_height)
        
        return (pixel_x, pixel_y)
        
    def display(self, drones: list = None):
        """
        Wyświetla mapę z opcjonalnymi dronami
        :param drones: lista obiektów Drone do wyświetlenia
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.grid, cmap='gray', origin='upper', extent=[0, self.width, self.height, 0])
        
        # Dodaj drony jeśli podane
        if drones:
            for drone in drones:
                plt.plot(drone.coordinate[0], drone.coordinate[1], 
                        'o', color=drone.drone_color, markersize=10, 
                        markeredgecolor='white', markeredgewidth=2)
        
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.title("Map - Układ Kartezjański")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # Przykład użycia z obrazem - wymiary automatycznie dopasowane do rozdzielczości
    map_with_image = Map(image_path="/Users/sansrozpierducha/hackathon/Zrzut ekranu 2025-10-17 o 20.32.16.png")
    # Konwersja współrzędnych
    x, y = map_with_image.pixel_to_coordinates(100, 150)
    print(f"Piksel (100, 150) -> Współrzędne: ({x:.2f}, {y:.2f})")
    print(f"Rozmiar mapy: {map_with_image.width} x {map_with_image.height}")
    
    # Utworzenie drona w centrum mapy
    drone1 = Drone(map_with_image.width / 2, map_with_image.height / 2)
    
    # Wyświetlenie mapy z dronem
    map_with_image.display(drones=[drone1])

