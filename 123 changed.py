"""
Spatio-Temporal (4D) IDW Interpolation for Weather Prediction
============================================================

This baseline model predicts weather at arbitrary (time, lat, lon, alt) points
by interpolating from the 'k' nearest aerodrome measurements in 4D space.

This is a 'k-Nearest Neighbors IDW' approach.

Author: [Your Name]
Date: October 2025
Course: [Your Course Code]
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DISTANCE CALCULATIONS (3D and 4D)
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
# PART 2: k-NN IDW INTERPOLATOR CLASS
# ============================================================================

class SpatioTemporalInterpolator:
    """
    Spatio-Temporal k-Nearest Neighbors IDW Interpolator.
    
    This model interpolates weather by:
    1. Finding the 'k' nearest data points in a 4D (time, space)
    2. Performing Inverse Distance Weighting (IDW) on just those 'k' neighbors.
    
    This addresses the "interpolate from 3-4 nearest points" requirement,
    which is a common and robust alternative to true barycentric interpolation,
    especially when query points may be outside the convex hull of the data
    (e.g., at high altitude).
    """
    
    def __init__(self, 
                 k_neighbors: int = 4,
                 power: float = 2.0, 
                 time_scale_km_per_hour: float = 50.0):
        """
        Initialize the k-NN IDW interpolator.
        
        Args:
            k_neighbors: Number of nearest neighbors to use (e.g., 3, 4, or 5)
            
            power: Exponent for distance weighting (default=2)
            
            time_scale_km_per_hour: A critical hyperparameter!
                This defines the "exchange rate" between time and space.
                e.g., 50.0 means "1 hour of time difference is equivalent
                to 50km of spatial distance" when finding neighbors.
                This value should represent how fast weather systems move.
        """
        self.k_neighbors = k_neighbors
        self.power = power
        self.time_scale_km_per_hour = time_scale_km_per_hour
        
        if self.k_neighbors < 2:
            raise ValueError("k_neighbors must be at least 2 for interpolation")
        
        # These will be set during fit()
        self.aerodrome_coords = None
        self.aerodrome_weather = None
        self.aerodrome_names = None
        self.weather_variables = None
    
    
    def fit(self, 
            spatiotemporal_coords: np.ndarray, 
            weather_data: np.ndarray,
            aerodrome_names: Optional[List[str]] = None,
            variable_names: Optional[List[str]] = None):
        """
        Store aerodrome locations/times and weather measurements.
        
        Args:
            spatiotemporal_coords: (N, 4) array of
                [posix_timestamp, lat, lon, altitude_feet]
            
            weather_data: (N, M) array of weather measurements
                Each row = one aerodrome at one time
                Each column = one weather variable
            
            aerodrome_names: List of aerodrome codes (e.g., ['CYYZ', 'CYUL'])
            variable_names: List of variable names (e.g., ['temp', 'pressure'])
        
        Example:
            ts1 = datetime.now().timestamp()
            ts2 = ts1 + 3600  # 1 hour later
            
            coords = np.array([
                [ts1, 43.6777, -79.6248, 569],   # CYYZ at T=0
                [ts1, 45.4706, -73.7408, 118],   # CYUL at T=0
                [ts2, 43.6777, -79.6248, 569],   # CYYZ at T=1
            ])
            
            weather = np.array([
                [15.0, 1013.25],  # Weather for CYYZ at T=0
                [12.0, 1015.00],  # Weather for CYUL at T=0
                [16.0, 1012.00]   # Weather for CYYZ at T=1
            ])
            
            idw.fit(coords, weather)
        """
        self.aerodrome_coords = np.array(spatiotemporal_coords)
        self.aerodrome_weather = np.array(weather_data)
        
        if self.aerodrome_coords.shape[0] < self.k_neighbors:
            raise ValueError(
                f"Number of data points ({self.aerodrome_coords.shape[0]}) "
                f"is less than k_neighbors ({self.k_neighbors})"
            )
        
        if self.aerodrome_coords.shape[1] != 4:
            raise ValueError(
                f"spatiotemporal_coords must be (N, 4), "
                f"got {self.aerodrome_coords.shape}"
            )
        
        # Store metadata
        if aerodrome_names is None:
            self.aerodrome_names = [f"DataPoint_{i}" for i in range(len(spatiotemporal_coords))]
        else:
            self.aerodrome_names = aerodrome_names
        
        if variable_names is None:
            self.weather_variables = [f"Variable_{i}" for i in range(weather_data.shape[1])]
        else:
            self.weather_variables = variable_names
        
        print(f"✓ Fitted k-NN IDW model with {len(self.aerodrome_names)} data points")
        print(f"  k_neighbors = {self.k_neighbors}")
        print(f"  time_scale_km_per_hour = {self.time_scale_km_per_hour}")
        print(f"  Weather variables: {', '.join(self.weather_variables)}")
    
    
    def predict(self, test_points: np.ndarray, return_weights: bool = False) -> np.ndarray:
        """
        Predict weather at test points using k-NN IDW interpolation.
        
        Args:
            test_points: (K, 4) array of 
                [posix_timestamp, lat, lon, altitude_feet]
            return_weights: If True, also return the weights used
        
        Returns:
            predictions: (K, M) array of predicted weather variables
            weights: (K, k) array of weights (if return_weights=True)
        """
        if self.aerodrome_coords is None:
            raise ValueError("Model not fitted! Call fit() first.")
        
        test_points = np.array(test_points)
        if test_points.ndim == 1:
            test_points = test_points.reshape(1, -1)
        
        if test_points.shape[1] != 4:
            raise ValueError(
                f"test_points must be (K, 4), got {test_points.shape}"
            )
        
        predictions = []
        all_k_weights = []
        
        # Pre-calculate for vectorization
        all_spatial_coords = self.aerodrome_coords[:, 1:]
        all_timestamps = self.aerodrome_coords[:, 0]
        
        for point in test_points:
            # Unpack query point
            query_timestamp = point[0]
            query_spatial_point = point[1:]
            
            # --- 1. Calculate 4D Distances ---
            
            # Calculate all spatial (3D) distances
            spatial_distances = np.array([
                distance_3d(query_spatial_point, aero_coord)
                for aero_coord in all_spatial_coords
            ])
            
            # Calculate all temporal distances (in hours)
            time_distances_hr = np.abs(query_timestamp - all_timestamps) / 3600.0
            
            # Scale time distance to be "km-equivalent"
            time_distances_scaled = time_distances_hr * self.time_scale_km_per_hour
            
            # Combine into a single 4D distance
            distances = np.sqrt(spatial_distances**2 + time_distances_scaled**2)
            
            # --- 2. Find k-Nearest Neighbors ---
            k_indices = np.argsort(distances)[:self.k_neighbors]
            
            # Filter all data to just the k-nearest
            k_distances = distances[k_indices]
            k_weather = self.aerodrome_weather[k_indices]
            
            # --- 3. Perform IDW on k-Nearest Neighbors ---
            
            # Handle exact match (distance = 0)
            if np.min(k_distances) < 1e-6:
                idx_in_k = np.argmin(k_distances)
                predictions.append(k_weather[idx_in_k])
                
                weights = np.zeros(self.k_neighbors)
                weights[idx_in_k] = 1.0
                all_k_weights.append(weights)
                continue
            
            # Calculate IDW weights
            weights = 1.0 / (k_distances ** self.power)
            
            # Normalize weights to sum to 1
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                # Should not happen if exact match is caught, but as a safeguard
                weights = np.ones(self.k_neighbors) / self.k_neighbors
            else:
                weights = weights / weights_sum
            
            # Weighted average
            prediction = np.sum(weights[:, np.newaxis] * k_weather, axis=0)
            
            predictions.append(prediction)
            all_k_weights.append(weights)
        
        predictions = np.array(predictions)
        
        if return_weights:
            return predictions, np.array(all_k_weights)
        else:
            return predictions
    
    
    def predict_single(self, 
                       timestamp: float, 
                       lat: float, 
                       lon: float, 
                       altitude: float) -> Dict[str, float]:
        """
        Convenience method to predict at a single point.
        
        Args:
            timestamp: POSIX timestamp (e.g., datetime.now().timestamp())
            lat: Latitude
            lon: Longitude
            altitude: Altitude in feet
            
        Returns:
            Dictionary mapping variable names to predicted values
        
        Example:
            >>> ts = datetime.now().timestamp()
            >>> result = idw.predict_single(ts, 44.5, -76.5, 35000)
            >>> print(f"Temperature: {result['temperature']}°C")
        """
        point = np.array([[timestamp, lat, lon, altitude]])
        prediction = self.predict(point)[0]
        
        return dict(zip(self.weather_variables, prediction))
    
    
    def predict_along_route(self, route_waypoints: List[Tuple]) -> pd.DataFrame:
        """
        Predict weather along an entire flight route.
        
        Args:
            route_waypoints: List of (timestamp, lat, lon, altitude, ...) tuples.
                             Additional items (e.g., ground_speed) are ignored
                             by predict() but re-added to the final DataFrame.
        
        Returns:
            DataFrame with predictions at each waypoint
        
        Example:
            ts = datetime.now().timestamp()
            route = [
                (ts, 43.6777, -79.6248, 5000, 250, 0),    # Toronto departure
                (ts+1800, 44.5, -76.5, 35000, 450, 0),  # Cruise (30m later)
                (ts+3600, 45.4706, -73.7408, 3000, 180, 60) # Montreal (60m later)
            ]
            
            predictions_df = idw.predict_along_route(route)
        """
        # Extract coordinates
        coords = np.array([(wp[0], wp[1], wp[2], wp[3]) for wp in route_waypoints])
        
        # Predict
        predictions = self.predict(coords)
        
        # Create DataFrame
        df = pd.DataFrame(predictions, columns=self.weather_variables)
        df['timestamp'] = coords[:, 0]
        df['datetime_utc'] = [datetime.utcfromtimestamp(ts) for ts in coords[:, 0]]
        df['latitude'] = coords[:, 1]
        df['longitude'] = coords[:, 2]
        df['altitude_ft'] = coords[:, 3]
        
        if len(route_waypoints[0]) >= 5:
            df['ground_speed_kts'] = [wp[4] for wp in route_waypoints]
        if len(route_waypoints[0]) >= 6:
            df['layover_min'] = [wp[5] for wp in route_waypoints]
        
        return df


# ============================================================================
# PART 3: EXAMPLE USAGE & TESTING
# ============================================================================

def create_sample_aerodrome_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sample aerodrome data for testing.
    
    Returns:
        coords: (N, 4) array [timestamp, lat, lon, alt]
        weather: (N, 6) array [temp, pressure, wind_speed, wind_dir, visibility, ceiling]
        names: List of aerodrome codes
    """
    # Major Canadian airports
    aerodromes = {
        'CYYZ': {'lat': 43.6777, 'lon': -79.6248, 'alt': 569},   # Toronto
        'CYUL': {'lat': 45.4706, 'lon': -73.7408, 'alt': 118},   # Montreal
        'CYVR': {'lat': 49.1939, 'lon': -123.1844, 'alt': 14},  # Vancouver
        'CYYC': {'lat': 51.1139, 'lon': -114.0203, 'alt': 3557}, # Calgary
        'CYOW': {'lat': 45.3225, 'lon': -75.6692, 'alt': 374},   # Ottawa
    }
    
    # Simulated weather (in reality, you'd fetch METAR/HRDPS data)
    # We will create two time-steps (now, and 1 hour from now)
    
    time_now = datetime.now().timestamp()
    time_plus_1hr = (datetime.now() + timedelta(hours=1)).timestamp()
    
    # --- Time 1: NOW ---
    weather_data_t1 = {
        'CYYZ': [15.0, 1013.25, 12.5, 80, 10.0, 5000],
        'CYUL': [12.0, 1015.00, 10.2, 90, 15.0, 6000],
        'CYVR': [18.0, 1012.50, 8.5, 270, 20.0, 8000],
        'CYYC': [10.0, 1010.00, 15.0, 310, 12.0, 4000],
        'CYOW': [13.5, 1014.50, 11.8, 85, 18.0, 7000],
    }
    
    # --- Time 2: +1 HOUR ---
    weather_data_t2 = {
        'CYYZ': [15.5, 1013.00, 13.0, 85, 10.0, 5500], # Slightly warmer, pressure drop
        'CYUL': [12.2, 1014.80, 11.0, 95, 15.0, 6000],
        'CYVR': [18.0, 1012.00, 9.0, 265, 20.0, 8000],
        'CYYC': [10.5, 1009.50, 14.5, 315, 12.0, 4200],
        'CYOW': [14.0, 1014.00, 12.0, 90, 18.0, 7000],
    }
    
    coords = []
    weather = []
    names = []
    
    # Add data for T=NOW
    for code, location in aerodromes.items():
        coords.append([time_now, location['lat'], location['lon'], location['alt']])
        weather.append(weather_data_t1[code])
        names.append(f"{code}_T0")
        
    # Add data for T=+1HR
    for code, location in aerodromes.items():
        coords.append([time_plus_1hr, location['lat'], location['lon'], location['alt']])
        weather.append(weather_data_t2[code])
        names.append(f"{code}_T1")
    
    # Our data now has N = 5 * 2 = 10 data points
    return np.array(coords), np.array(weather), names


def example_basic_usage():
    """Example 1: Basic k-NN IDW prediction"""
    print("=" * 70)
    print("EXAMPLE 1: Basic k-NN IDW Usage (4D)")
    print("=" * 70)
    
    # Create sample data
    coords, weather, names = create_sample_aerodrome_data()
    
    variable_names = ['temperature', 'pressure', 'wind_speed', 
                      'wind_direction', 'visibility', 'cloud_ceiling']
    
    # Initialize and fit model
    # Use k=4 nearest neighbors, time scale = 50 km/hr
    idw = SpatioTemporalInterpolator(k_neighbors=4, power=2.0, time_scale_km_per_hour=50)
    idw.fit(coords, weather, names, variable_names)
    
    # Predict at a test point
    # Time: 30 minutes in the future (between T0 and T1)
    # Location: between Toronto and Montreal
    # Altitude: 15,000 ft
    
    test_time = (datetime.now() + timedelta(minutes=30)).timestamp()
    test_point_4d = [test_time, 44.5, -76.5, 15000]
    
    print(f"\n📍 Test Point: Time={datetime.utcfromtimestamp(test_point_4d[0])} (UTC)")
    print(f"               Lat={test_point_4d[1]}, Lon={test_point_4d[2]}, Alt={test_point_4d[3]} ft")
    print("-" * 70)
    
    prediction = idw.predict_single(test_point_4d[0], test_point_4d[1], test_point_4d[2], test_point_4d[3])
    
    for var, value in prediction.items():
        print(f"{var:20s}: {value:8.2f}")


def example_route_prediction():
    """Example 2: Predict along a flight route"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Flight Route Prediction (4D)")
    print("=" * 70)
    
    # Create sample data
    coords, weather, names = create_sample_aerodrome_data()
    variable_names = ['temperature', 'pressure', 'wind_speed', 
                      'wind_direction', 'visibility', 'cloud_ceiling']
    
    # Fit model
    idw = SpatioTemporalInterpolator(k_neighbors=4, power=2.0, time_scale_km_per_hour=50)
    idw.fit(coords, weather, names, variable_names)
    
    # Define a simple route: Toronto -> Montreal
    # Tuples are now (timestamp, lat, lon, alt, speed, layover)
    
    start_time = datetime.now().timestamp()
    
    route = [
        (start_time + 0,    43.6777, -79.6248, 1000, 180, 0),  # Takeoff (T=0m)
        (start_time + 900,  44.0, -78.0, 15000, 350, 0),      # Climb (T=15m)
        (start_time + 2700, 44.5, -76.5, 35000, 450, 0),      # Cruise (T=45m)
        (start_time + 4500, 45.0, -75.0, 20000, 300, 0),      # Descent (T=75m)
        (start_time + 5400, 45.4706, -73.7408, 500, 150, 30) # Landing (T=90m)
    ]
    
    # Predict along route
    predictions_df = idw.predict_along_route(route)
    
    print("\n✈️  Flight Route Predictions:")
    # We remove some columns for cleaner printing
    cols_to_print = [
        'datetime_utc', 'latitude', 'longitude', 'altitude_ft', 
        'temperature', 'pressure', 'wind_speed'
    ]
    print(predictions_df[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_route_prediction()
    
    print("\n" + "=" * 70)
    print("✓ 4D k-NN IDW Implementation Complete!")
    print("=" * 70)