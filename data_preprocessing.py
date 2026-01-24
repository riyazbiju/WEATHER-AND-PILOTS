"""
Data preprocessing and normalization utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import torch


class DataPreprocessor:
    """Handles loading, cleaning, and normalizing weather data"""
    
    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.coord_scaler = MinMaxScaler()
        self.time_scaler = MinMaxScaler()
        
        self.feature_cols = config.FEATURE_COLS
        self.coord_cols = config.COORD_COLS
        self.circular_vars = config.CIRCULAR_VARS
        
    def load_and_preprocess(self, data_path, verbose=True):
        """Load and preprocess the dataset"""
        if verbose:
            print(f"Loading data from {data_path}...")
        
        # Load data
        df = pd.read_csv(data_path, low_memory=False)
        
        if verbose:
            print(f"Raw data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by station and time
        df = df.sort_values(['station', 'datetime'])
        
        # Create time_hours from datetime (hours since start)
        min_time = df['datetime'].min()
        df['time_hours'] = (df['datetime'] - min_time).dt.total_seconds() / 3600.0
        
        # Handle missing values
        if verbose:
            print(f"\nMissing values before cleaning:")
            missing_counts = df[self.feature_cols + self.coord_cols + ['time_hours']].isnull().sum()
            print(missing_counts)
            total_missing = missing_counts.sum()
            print(f"Total missing values: {total_missing}")
        
        # Ensure all feature columns are numeric
        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Forward fill missing values within each station (temporal interpolation)
        for col in self.feature_cols:
            df[col] = df.groupby('station')[col].ffill()
            df[col] = df.groupby('station')[col].bfill()
        
        # Fill remaining missing values with median
        for col in self.feature_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                if verbose:
                    print(f"Filled {col} missing values with median: {median_val:.2f}")
        
        # Drop rows with missing coordinates (critical)
        df = df.dropna(subset=self.coord_cols)
        
        if verbose:
            print(f"\nData shape after cleaning: {df.shape}")
            print(f"Unique stations: {df['station'].nunique()}")
            print(f"Time range: {df['time_hours'].min():.2f} to {df['time_hours'].max():.2f} hours")
            print(f"Altitude range: {df['altitude'].min():.2f} to {df['altitude'].max():.2f}")
            print(f"Remaining missing values: {df[self.feature_cols].isnull().sum().sum()}")
        
        return df
    
    def fit_scalers(self, df):
        """Fit scalers on training data"""
        df_copy = df.copy()
        
        # Convert circular variables to sin/cos before fitting
        for var in self.circular_vars:
            if var in self.feature_cols:
                # Convert to numeric first (handle any string values)
                df_copy[var] = pd.to_numeric(df_copy[var], errors='coerce')
                # Fill any NaN from conversion with median
                df_copy[var] = df_copy[var].fillna(df_copy[var].median())
                # Convert degrees to radians and compute sin/cos
                rad = np.deg2rad(df_copy[var])
                df_copy[f'{var}_sin'] = np.sin(rad)
                df_copy[f'{var}_cos'] = np.cos(rad)
        
        # Fit feature scaler (exclude circular variables)
        non_circular_cols = [col for col in self.feature_cols if col not in self.circular_vars]
        if non_circular_cols:
            self.feature_scaler.fit(df_copy[non_circular_cols])
        
        # Fit coordinate scaler
        self.coord_scaler.fit(df[self.coord_cols])
        
        # Fit time scaler
        self.time_scaler.fit(df[['time_hours']])
        
        print("Scalers fitted successfully!")
        if non_circular_cols:
            print(f"Feature means: {self.feature_scaler.mean_}")
            print(f"Feature stds: {self.feature_scaler.scale_}")
        print(f"Circular variables (sin/cos encoded): {self.circular_vars}")
        
    def transform(self, df):
        """Transform features and coordinates using fitted scalers"""
        df_norm = df.copy()
        
        # Handle circular variables: convert to sin/cos
        circular_features = []
        for var in self.circular_vars:
            if var in self.feature_cols:
                # Convert to numeric first (handle any string values)
                df_norm[var] = pd.to_numeric(df_norm[var], errors='coerce')
                # Fill any NaN from conversion with median
                df_norm[var] = df_norm[var].fillna(df_norm[var].median())
                # Convert degrees to radians and compute sin/cos
                rad = np.deg2rad(df_norm[var])
                df_norm[f'{var}_sin'] = np.sin(rad)
                df_norm[f'{var}_cos'] = np.cos(rad)
                circular_features.extend([f'{var}_sin', f'{var}_cos'])
        
        # Normalize non-circular features
        non_circular_cols = [col for col in self.feature_cols if col not in self.circular_vars]
        if non_circular_cols:
            df_norm[non_circular_cols] = self.feature_scaler.transform(df[non_circular_cols])
        
        # Update feature columns to include sin/cos versions
        df_norm['_original_feature_cols'] = ','.join(self.feature_cols)
        
        # Normalize coordinates
        df_norm[self.coord_cols] = self.coord_scaler.transform(df[self.coord_cols])
        
        # Normalize time
        df_norm['time_hours_norm'] = self.time_scaler.transform(df[['time_hours']]).flatten()
        
        return df_norm
    
    def inverse_transform_features(self, features_norm, circular_indices=None):
        """Inverse transform normalized features to original scale"""
        # If we have circular variables encoded as sin/cos, we need to handle them
        if circular_indices is not None and len(self.circular_vars) > 0:
            # Reconstruct from sin/cos
            result = np.zeros((features_norm.shape[0], len(self.feature_cols)))
            
            non_circular_cols = [col for col in self.feature_cols if col not in self.circular_vars]
            non_circ_idx = 0
            
            for i, col in enumerate(self.feature_cols):
                if col in self.circular_vars:
                    # Get sin/cos indices
                    sin_idx = circular_indices[col]['sin']
                    cos_idx = circular_indices[col]['cos']
                    
                    # Reconstruct angle from sin/cos
                    angle_rad = np.arctan2(features_norm[:, sin_idx], features_norm[:, cos_idx])
                    angle_deg = np.rad2deg(angle_rad)
                    # Ensure 0-360 range
                    angle_deg = (angle_deg + 360) % 360
                    result[:, i] = angle_deg
                else:
                    # Regular inverse transform for non-circular
                    result[:, i] = self.feature_scaler.inverse_transform(
                        features_norm[:, non_circ_idx].reshape(-1, 1)
                    ).flatten()
                    non_circ_idx += 1
            
            return result
        else:
            # Simple case: no circular variables or already in correct format
            return self.feature_scaler.inverse_transform(features_norm)
    
    def save_scalers(self, path='scalers.pkl'):
        """Save fitted scalers"""
        scalers = {
            'feature_scaler': self.feature_scaler,
            'coord_scaler': self.coord_scaler,
            'time_scaler': self.time_scaler
        }
        with open(path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {path}")
    
    def load_scalers(self, path='scalers.pkl'):
        """Load fitted scalers"""
        with open(path, 'rb') as f:
            scalers = pickle.load(f)
        self.feature_scaler = scalers['feature_scaler']
        self.coord_scaler = scalers['coord_scaler']
        self.time_scaler = scalers['time_scaler']
        print(f"Scalers loaded from {path}")


class DataSplitter:
    """Handles different splitting strategies"""
    
    @staticmethod
    def random_split(df, train_ratio=0.7, val_ratio=0.15, seed=42):
        """Random split of observations"""
        np.random.seed(seed)
        indices = np.random.permutation(len(df))
        
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        return (df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
                df.iloc[test_idx].reset_index(drop=True))
    
    @staticmethod
    def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
        """Temporal split: train on early times, test on later times"""
        df = df.sort_values('time_hours')
        
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
        
        print(f"Temporal split:")
        print(f"  Train time range: {train_df['time_hours'].min():.2f} - {train_df['time_hours'].max():.2f}")
        print(f"  Val time range: {val_df['time_hours'].min():.2f} - {val_df['time_hours'].max():.2f}")
        print(f"  Test time range: {test_df['time_hours'].min():.2f} - {test_df['time_hours'].max():.2f}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def station_split(df, n_test_stations=30, n_val_stations=20, seed=42):
        """Station split: test on completely unseen stations"""
        np.random.seed(seed)
        
        stations = df['station'].unique()
        np.random.shuffle(stations)
        
        test_stations = stations[:n_test_stations]
        val_stations = stations[n_test_stations:n_test_stations + n_val_stations]
        train_stations = stations[n_test_stations + n_val_stations:]
        
        train_df = df[df['station'].isin(train_stations)].reset_index(drop=True)
        val_df = df[df['station'].isin(val_stations)].reset_index(drop=True)
        test_df = df[df['station'].isin(test_stations)].reset_index(drop=True)
        
        print(f"Station split:")
        print(f"  Train stations: {len(train_stations)}")
        print(f"  Val stations: {len(val_stations)}")
        print(f"  Test stations: {len(test_stations)}")
        print(f"  Train observations: {len(train_df)}")
        print(f"  Val observations: {len(val_df)}")
        print(f"  Test observations: {len(test_df)}")
        
        return train_df, val_df, test_df


def prepare_data(config, split_type='random'):
    """Complete data preparation pipeline"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load and preprocess
    df = preprocessor.load_and_preprocess(config.DATA_PATH)
    
    # Split data
    if split_type == 'random':
        train_df, val_df, test_df = DataSplitter.random_split(
            df, config.TRAIN_RATIO, config.VAL_RATIO, config.SEED
        )
    elif split_type == 'temporal':
        train_df, val_df, test_df = DataSplitter.temporal_split(
            df, config.TRAIN_RATIO, config.VAL_RATIO
        )
    elif split_type == 'station':
        train_df, val_df, test_df = DataSplitter.station_split(
            df, config.STATION_HOLDOUT, seed=config.SEED
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    # Fit scalers on training data
    preprocessor.fit_scalers(train_df)
    
    # Transform all splits
    train_df_norm = preprocessor.transform(train_df)
    val_df_norm = preprocessor.transform(val_df)
    test_df_norm = preprocessor.transform(test_df)
    
    # Save scalers
    preprocessor.save_scalers()
    
    return {
        'train': train_df_norm,
        'val': val_df_norm,
        'test': test_df_norm,
        'train_raw': train_df,
        'val_raw': val_df,
        'test_raw': test_df,
        'preprocessor': preprocessor
    }
