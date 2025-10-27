"""
Inverse Distance Weighted (IDW) Interpolation for Weather Prediction
====================================================================

This baseline model predicts weather at arbitrary (lat, lon, alt) points
by interpolating from nearby aerodrome measurements.

Author: [Your Name]
Date: October 2025
Course: [Your Course Code]
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DISTANCE CALCULATIONS (3D SPACE)
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Earth.
    
    This is the "horizontal" distance ignoring altitude.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in kilometers
        
    Example:
        >>> haversine_distance(43.6777, -79.6248, 45.4706, -73.7408)
        503.89  # Toronto to Montreal ~ 504 km
    """
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    R = 6371.0
    
    return R * c


def distance_3d(point1: Tuple[float, float, float], 
                point2: Tuple[float, float, float]) -> float:
    """
    Calculate 3D Euclidean distance between two points.
    
    Combines horizontal (lat/lon) and vertical (altitude) distances.
    
    Args:
        point1: (latitude, longitude, altitude_feet) tuple
        point2: (latitude, longitude, altitude_feet) tuple
    
    Returns:
        3D distance in kilometers
        
    Why 3D?
        Weather at 30,000 feet is VERY different from ground level!
        A nearby airport at sea level shouldn't dominate predictions
        for an airplane at cruising altitude.
    """
    lat1, lon1, alt1 = point1
    lat2, lon2, alt2 = point2
    
    # Horizontal distance (km)
    horizontal_dist = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Vertical distance (convert feet to km)
    vertical_dist = abs(alt2 - alt1) * 0.0003048  # 1 foot = 0.0003048 km
    
    # 3D Euclidean distance
    distance = np.sqrt(horizontal_dist**2 + vertical_dist**2)
    
    return distance


# ============================================================================
# PART 2: IDW INTERPOLATOR CLASS
# ============================================================================

class IDWWeatherInterpolator:
    """
    Inverse Distance Weighted interpolation for multi-variable weather prediction.
    
    This is your BASELINE 1 model.
    
    Key Features:
        - Handles multiple weather variables simultaneously
        - Uses 3D distance (lat, lon, altitude)
        - Configurable power parameter
        - Stores aerodrome locations for repeated queries
    """
    
    def __init__(self, power: float = 2.0, min_neighbors: int = 3):
        """
        Initialize IDW interpolator.
        
        Args:
            power: Exponent for distance weighting (default=2)
                   - power=1: Linear decay
                   - power=2: Quadratic decay (standard)
                   - power=3: Faster decay (more local influence)
            
            min_neighbors: Minimum number of aerodromes to use (default=3)
        """
        self.power = power
        self.min_neighbors = min_neighbors
        
        # These will be set during fit()
        self.aerodrome_coords = None
        self.aerodrome_weather = None
        self.aerodrome_names = None
        self.weather_variables = None
        
    
    def fit(self, 
            aerodrome_coords: np.ndarray, 
            weather_data: np.ndarray,
            aerodrome_names: Optional[List[str]] = None,
            variable_names: Optional[List[str]] = None):
        """
        Store aerodrome locations and weather measurements.
        
        Args:
            aerodrome_coords: (N, 3) array of [lat, lon, altitude_feet]
            weather_data: (N, M) array of weather measurements
                         Each row = one aerodrome
                         Each column = one weather variable
            aerodrome_names: List of aerodrome codes (e.g., ['CYYZ', 'CYUL'])
            variable_names: List of variable names (e.g., ['temp', 'pressure'])
        
        Example:
            coords = np.array([
                [43.6777, -79.6248, 569],    # CYYZ (Toronto)
                [45.4706, -73.7408, 118]     # CYUL (Montreal)
            ])
            
            weather = np.array([
                [15.0, 1013.25, 12.5, 80, 10, 5000],  # temp, pressure, wind_spd, wind_dir, vis, ceiling
                [12.0, 1015.00, 10.2, 90, 15, 6000]
            ])
            
            idw.fit(coords, weather)
        """
        self.aerodrome_coords = np.array(aerodrome_coords)
        self.aerodrome_weather = np.array(weather_data)
        
        # Store metadata
        if aerodrome_names is None:
            self.aerodrome_names = [f"Aerodrome_{i}" for i in range(len(aerodrome_coords))]
        else:
            self.aerodrome_names = aerodrome_names
        
        if variable_names is None:
            self.weather_variables = [f"Variable_{i}" for i in range(weather_data.shape[1])]
        else:
            self.weather_variables = variable_names
        
        print(f"✓ Fitted IDW model with {len(self.aerodrome_names)} aerodromes")
        print(f"  Weather variables: {', '.join(self.weather_variables)}")
    
    
    def predict(self, test_points: np.ndarray, return_weights: bool = False) -> np.ndarray:
        """
        Predict weather at test points using IDW interpolation.
        
        Args:
            test_points: (K, 3) array of [lat, lon, altitude_feet]
            return_weights: If True, also return the weights used
        
        Returns:
            predictions: (K, M) array of predicted weather variables
            weights: (K, N) array of weights (if return_weights=True)
        
        Algorithm:
            For each test point:
                1. Calculate distance to all aerodromes
                2. Compute weights = 1 / distance^power
                3. Normalize weights to sum to 1
                4. Weighted average of weather values
        """
        if self.aerodrome_coords is None:
            raise ValueError("Model not fitted! Call fit() first.")
        
        test_points = np.array(test_points)
        if test_points.ndim == 1:
            test_points = test_points.reshape(1, -1)
        
        predictions = []
        all_weights = []
        
        for point in test_points:
            # Calculate distances to all aerodromes
            distances = np.array([
                distance_3d(point, aerodrome_coord)
                for aerodrome_coord in self.aerodrome_coords
            ])
            
            # Handle exact match (distance = 0)
            if np.min(distances) < 1e-6:
                # Return exact measurement from that aerodrome
                idx = np.argmin(distances)
                predictions.append(self.aerodrome_weather[idx])
                
                # Create one-hot weights
                weights = np.zeros(len(distances))
                weights[idx] = 1.0
                all_weights.append(weights)
                continue
            
            # Calculate IDW weights
            weights = 1.0 / (distances ** self.power)
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Weighted average
            prediction = np.sum(weights[:, np.newaxis] * self.aerodrome_weather, axis=0)
            
            predictions.append(prediction)
            all_weights.append(weights)
        
        predictions = np.array(predictions)
        
        if return_weights:
            return predictions, np.array(all_weights)
        else:
            return predictions
    
    
    def predict_single(self, lat: float, lon: float, altitude: float) -> Dict[str, float]:
        """
        Convenience method to predict at a single point.
        
        Returns:
            Dictionary mapping variable names to predicted values
        
        Example:
            >>> result = idw.predict_single(44.5, -76.5, 35000)
            >>> print(f"Temperature: {result['temperature']}°C")
        """
        point = np.array([[lat, lon, altitude]])
        prediction = self.predict(point)[0]
        
        return dict(zip(self.weather_variables, prediction))
    
    
    def predict_along_route(self, route_waypoints: List[Tuple]) -> pd.DataFrame:
        """
        Predict weather along an entire flight route.
        
        Args:
            route_waypoints: List of (lat, lon, altitude, ground_speed, layover) tuples
        
        Returns:
            DataFrame with predictions at each waypoint
        
        Example:
            route = [
                (43.6777, -79.6248, 5000, 250, 0),   # Toronto departure
                (44.5, -76.5, 35000, 450, 0),        # Cruising
                (45.4706, -73.7408, 3000, 180, 60)   # Montreal arrival
            ]
            
            predictions_df = idw.predict_along_route(route)
        """
        # Extract coordinates
        coords = np.array([(wp[0], wp[1], wp[2]) for wp in route_waypoints])
        
        # Predict
        predictions = self.predict(coords)
        
        # Create DataFrame
        df = pd.DataFrame(predictions, columns=self.weather_variables)
        df['latitude'] = coords[:, 0]
        df['longitude'] = coords[:, 1]
        df['altitude_ft'] = coords[:, 2]
        
        if len(route_waypoints[0]) >= 4:
            df['ground_speed_kts'] = [wp[3] for wp in route_waypoints]
        if len(route_waypoints[0]) >= 5:
            df['layover_min'] = [wp[4] for wp in route_waypoints]
        
        return df


# ============================================================================
# PART 3: EXAMPLE USAGE & TESTING
# ============================================================================

def create_sample_aerodrome_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sample aerodrome data for testing.
    
    Returns:
        coords: (N, 3) array
        weather: (N, 6) array [temp, pressure, wind_speed, wind_dir, visibility, ceiling]
        names: List of aerodrome codes
    """
    # Major Canadian airports
    aerodromes = {
        'CYYZ': {'lat': 43.6777, 'lon': -79.6248, 'alt': 569},   # Toronto
        'CYUL': {'lat': 45.4706, 'lon': -73.7408, 'alt': 118},   # Montreal
        'CYVR': {'lat': 49.1939, 'lon': -123.1844, 'alt': 14},   # Vancouver
        'CYYC': {'lat': 51.1139, 'lon': -114.0203, 'alt': 3557}, # Calgary
        'CYOW': {'lat': 45.3225, 'lon': -75.6692, 'alt': 374},   # Ottawa
    }
    
    # Simulated weather (in reality, you'd fetch METAR/HRDPS data)
    weather_data = {
        'CYYZ': [15.0, 1013.25, 12.5, 80, 10.0, 5000],
        'CYUL': [12.0, 1015.00, 10.2, 90, 15.0, 6000],
        'CYVR': [18.0, 1012.50, 8.5, 270, 20.0, 8000],
        'CYYC': [10.0, 1010.00, 15.0, 310, 12.0, 4000],
        'CYOW': [13.5, 1014.50, 11.8, 85, 18.0, 7000],
    }
    
    coords = []
    weather = []
    names = []
    
    for code, location in aerodromes.items():
        coords.append([location['lat'], location['lon'], location['alt']])
        weather.append(weather_data[code])
        names.append(code)
    
    return np.array(coords), np.array(weather), names


def example_basic_usage():
    """Example 1: Basic IDW prediction"""
    print("=" * 70)
    print("EXAMPLE 1: Basic IDW Usage")
    print("=" * 70)
    
    # Create sample data
    coords, weather, names = create_sample_aerodrome_data()
    
    variable_names = ['temperature', 'pressure', 'wind_speed', 
                     'wind_direction', 'visibility', 'cloud_ceiling']
    
    # Initialize and fit model
    idw = IDWWeatherInterpolator(power=2.0)
    idw.fit(coords, weather, names, variable_names)
    
    # Predict at a test point (somewhere between Toronto and Montreal)
    test_point = [44.5, -76.5, 15000]  # Mid-flight altitude
    
    print(f"\n📍 Test Point: Lat={test_point[0]}, Lon={test_point[1]}, Alt={test_point[2]} ft")
    print("-" * 70)
    
    prediction = idw.predict_single(test_point[0], test_point[1], test_point[2])
    
    for var, value in prediction.items():
        print(f"{var:20s}: {value:8.2f}")


def example_route_prediction():
    """Example 2: Predict along a flight route"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Flight Route Prediction")
    print("=" * 70)
    
    # Create sample data
    coords, weather, names = create_sample_aerodrome_data()
    variable_names = ['temperature', 'pressure', 'wind_speed', 
                     'wind_direction', 'visibility', 'cloud_ceiling']
    
    # Fit model
    idw = IDWWeatherInterpolator(power=2.0)
    idw.fit(coords, weather, names, variable_names)
    
    # Define a simple route: Toronto → Montreal
    route = [
        (43.6777, -79.6248, 1000, 180, 0),   # Takeoff
        (44.0, -78.0, 15000, 350, 0),        # Climb
        (44.5, -76.5, 35000, 450, 0),        # Cruise
        (45.0, -75.0, 20000, 300, 0),        # Descent
        (45.4706, -73.7408, 500, 150, 30)    # Landing
    ]
    
    # Predict along route
    predictions_df = idw.predict_along_route(route)
    
    print("\n✈️  Flight Route Predictions:")
    print(predictions_df.to_string(index=False))


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_route_prediction()
    
    print("\n" + "=" * 70)
    print("✓ IDW Baseline Implementation Complete!")
    print("=" * 70)