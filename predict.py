class Prediction:
    def __init__(self):
        self.prediction_cache = {}
    
    def predict(self, drone_data):
        """
        Predict future drone positions based on trajectory history.
        
        Args:
            drone_data: pandas DataFrame with 'x' and 'y' columns
            
        Returns:
            dict: Dictionary with 'points' key containing list of predicted positions
        """
        # Use only last 5 points for prediction
        last_points = drone_data[['x', 'y']].tail(5).values.tolist()
        
        # Create cache key based on input data
        cache_key = str(tuple(map(tuple, last_points)))
        
        # Check if result is already in cache
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # If not in cache, use simplified prediction method
        try:
            diffs = []
            for i in range(1, len(last_points)):
                dx = last_points[i][0] - last_points[i-1][0]
                dy = last_points[i][1] - last_points[i-1][1]
                diffs.append((dx, dy))
            
            if diffs:
                avg_dx = sum(d[0] for d in diffs) / len(diffs)
                avg_dy = sum(d[1] for d in diffs) / len(diffs)
                
                last_point = last_points[-1]
                predicted_points = []
                
                for i in range(1, 4): 
                    new_x = last_point[0] + (avg_dx * i)
                    new_y = last_point[1] + (avg_dy * i)
                    predicted_points.append({"x": round(new_x, 2), "y": round(new_y, 2)})
                
                prediction = {"points": predicted_points}
                
                self.prediction_cache[cache_key] = prediction
                return prediction
        except Exception as e:
            return None
        
        return None