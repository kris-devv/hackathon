import numpy as np
from numpy import poly1d

class Prediction:
    def __init__(self):
        self.prediction_cache = {}
    
    def predict(self, drone_data, steps_ahead=3):
        """
        Predict future drone positions based on trajectory history using polynomial fitting.
        
        Args:
            drone_data: pandas DataFrame with 'x' and 'y' columns
            steps_ahead: number of future points to predict (default 3)
            
        Returns:
            dict: Dictionary with 'points' key containing list of predicted positions
        """
        # Use last 10 points for better polynomial fitting
        last_points = drone_data[['x', 'y']].tail(10).values.tolist()
        
        if len(last_points) < 2:
            return None
        
        # Create cache key based on input data
        cache_key = str(tuple(map(tuple, last_points)))
        
        # Check if result is already in cache
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        try:
            # Extract x and y coordinates
            x_coords = [p[0] for p in last_points]
            y_coords = [p[1] for p in last_points]
            
            # Determine polynomial degree based on number of points
            if len(last_points) == 2:
                # Linear fit for 2 points
                coeffs = np.polyfit(x_coords, y_coords, 1)
            else:
                # Quadratic fit for 3+ points
                coeffs = np.polyfit(x_coords, y_coords, 2)
            
            # Create polynomial function
            poly_func = poly1d(coeffs)
            
            # Get last position and calculate step size
            last_x = x_coords[-1]
            if len(x_coords) > 1:
                # Calculate average x step
                x_diffs = [x_coords[i] - x_coords[i-1] for i in range(1, len(x_coords))]
                avg_x_step = sum(x_diffs) / len(x_diffs)
            else:
                avg_x_step = -10  # Default step (moving left)
            
            # Generate predicted points
            predicted_points = []
            for i in range(1, steps_ahead + 1):
                pred_x = last_x + (avg_x_step * i)
                pred_y = float(poly_func(pred_x))
                predicted_points.append({"x": round(pred_x, 2), "y": round(pred_y, 2)})
            
            prediction = {"points": predicted_points, "coefficients": coeffs.tolist()}
            
            # Cache the result
            self.prediction_cache[cache_key] = prediction
            return prediction
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None