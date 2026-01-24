"""
Context selection module: K-Nearest Neighbors in spacetime
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional


class SpacetimeContextSelector:
    """
    Selects K nearest neighbors in spacetime for each query point.
    Supports temporal causality enforcement for temporal split.
    """
    
    def __init__(self, config, enforce_causality=False):
        """
        Args:
            config: Configuration object
            enforce_causality: If True, only use observations from t < query_time
        """
        self.config = config
        self.k = config.K_NEIGHBORS
        self.time_weight = config.TIME_WEIGHT
        self.space_weight = config.SPACE_WEIGHT
        self.max_time_diff = config.MAX_TIME_DIFF
        self.enforce_causality = enforce_causality
        
        self.tree = None
        self.observations = None
        
    def build_index(self, df, verbose=True):
        """
        Build KD-tree for fast nearest neighbor search.
        
        Args:
            df: DataFrame with normalized coordinates and time
        """
        # Extract spacetime coordinates
        # Combine [time, lat, lon, alt] with appropriate weighting
        coords = np.column_stack([
            df['time_hours_norm'].values * self.time_weight,
            df['latitude'].values * self.space_weight,
            df['longitude'].values * self.space_weight,
            df['altitude'].values * self.space_weight
        ])
        
        # Build KD-tree
        self.tree = cKDTree(coords)
        self.observations = df
        
        if verbose:
            print(f"Built KD-tree with {len(df)} observations")
            print(f"  Time weight: {self.time_weight}")
            print(f"  Space weight: {self.space_weight}")
            print(f"  K neighbors: {self.k}")
            print(f"  Enforce causality: {self.enforce_causality}")
    
    def get_context(self, query_idx: int, k: Optional[int] = None) -> np.ndarray:
        """
        Get K nearest neighbors for a query point.
        
        Args:
            query_idx: Index of query point in observations
            k: Number of neighbors (default: self.k)
            
        Returns:
            indices: Array of context observation indices
        """
        if k is None:
            k = self.k
            
        # Get query point
        query_row = self.observations.iloc[query_idx]
        query_time = query_row['time_hours_norm']
        query_time_raw = query_row['time_hours']
        
        # Build query coordinate
        query_coord = np.array([
            query_time * self.time_weight,
            query_row['latitude'] * self.space_weight,
            query_row['longitude'] * self.space_weight,
            query_row['altitude'] * self.space_weight
        ])
        
        if self.enforce_causality:
            # Filter to only past observations
            valid_mask = self.observations['time_hours'].values < query_time_raw
            
            if valid_mask.sum() < k:
                # Not enough past observations, use what we have
                valid_indices = np.where(valid_mask)[0]
                return valid_indices
            
            # Build temporary tree with only valid observations
            valid_coords = np.column_stack([
                self.observations.loc[valid_mask, 'time_hours_norm'].values * self.time_weight,
                self.observations.loc[valid_mask, 'latitude'].values * self.space_weight,
                self.observations.loc[valid_mask, 'longitude'].values * self.space_weight,
                self.observations.loc[valid_mask, 'altitude'].values * self.space_weight
            ])
            
            temp_tree = cKDTree(valid_coords)
            distances, temp_indices = temp_tree.query(query_coord, k=k + 1)  # +1 to exclude self
            
            # Map back to original indices
            valid_indices = np.where(valid_mask)[0]
            context_indices = valid_indices[temp_indices]
            
            # Remove query point itself if present
            context_indices = context_indices[context_indices != query_idx][:k]
            
        else:
            # Query K+1 neighbors to exclude the query point itself
            distances, indices = self.tree.query(query_coord, k=k + 1)
            
            # Remove query point itself
            context_indices = indices[indices != query_idx][:k]
        
        return context_indices
    
    def get_context_batch(self, query_indices: List[int], k: Optional[int] = None) -> List[np.ndarray]:
        """
        Get contexts for a batch of query points.
        
        Args:
            query_indices: List of query point indices
            k: Number of neighbors
            
        Returns:
            List of context index arrays
        """
        return [self.get_context(idx, k) for idx in query_indices]


class ContextDataCollator:
    """
    Collates observations and contexts into batch tensors.
    Handles variable-length contexts via padding/masking.
    Handles circular variables encoded as sin/cos.
    """
    
    def __init__(self, config, observations):
        """
        Args:
            config: Configuration object
            observations: DataFrame with all observations
        """
        self.config = config
        self.observations = observations
        self.feature_cols = config.FEATURE_COLS
        self.coord_cols = config.COORD_COLS
        self.circular_vars = config.CIRCULAR_VARS
        
        # Determine actual feature columns (with sin/cos for circular vars)
        self.actual_feature_cols = []
        for col in self.feature_cols:
            if col in self.circular_vars:
                self.actual_feature_cols.append(f'{col}_sin')
                self.actual_feature_cols.append(f'{col}_cos')
            else:
                self.actual_feature_cols.append(col)
        
    def collate(self, batch_data: List[Tuple[int, np.ndarray]]) -> dict:
        """
        Collate a batch of (query_idx, context_indices) pairs.
        
        Args:
            batch_data: List of (query_idx, context_indices) tuples
            
        Returns:
            Dictionary with tensors:
                - query_coords: [B, 4] (time, lat, lon, alt)
                - query_features: [B, F] (may include sin/cos for circular vars)
                - context_coords: [B, K, 4]
                - context_features: [B, K, F]
                - context_mask: [B, K] (1 for valid, 0 for padding)
        """
        batch_size = len(batch_data)
        max_k = max(len(context_idx) for _, context_idx in batch_data)
        n_features = len(self.actual_feature_cols)
        
        # Initialize tensors
        query_coords = torch.zeros(batch_size, 4)
        query_features = torch.zeros(batch_size, n_features)
        context_coords = torch.zeros(batch_size, max_k, 4)
        context_features = torch.zeros(batch_size, max_k, n_features)
        context_mask = torch.zeros(batch_size, max_k)
        
        for i, (query_idx, context_indices) in enumerate(batch_data):
            # Get query observation
            query_obs = self.observations.iloc[query_idx]
            
            query_coords[i] = torch.tensor([
                query_obs['time_hours_norm'],
                query_obs['latitude'],
                query_obs['longitude'],
                query_obs['altitude']
            ])
            
            query_features[i] = torch.tensor(
                query_obs[self.actual_feature_cols].values.astype(np.float32)
            )
            
            # Get context observations
            context_obs = self.observations.iloc[context_indices]
            k_actual = len(context_indices)
            
            context_coords[i, :k_actual] = torch.tensor(
                context_obs[['time_hours_norm'] + self.coord_cols].values.astype(np.float32)
            )
            
            context_features[i, :k_actual] = torch.tensor(
                context_obs[self.actual_feature_cols].values.astype(np.float32)
            )
            
            context_mask[i, :k_actual] = 1.0
        
        return {
            'query_coords': query_coords,
            'query_features': query_features,
            'context_coords': context_coords,
            'context_features': context_features,
            'context_mask': context_mask
        }


def compute_spacetime_distance(
    time1: np.ndarray,
    coords1: np.ndarray,
    time2: np.ndarray,
    coords2: np.ndarray,
    time_weight: float = 1.0,
    space_weight: float = 1.0
) -> np.ndarray:
    """
    Compute spacetime distance between two sets of points.
    
    Args:
        time1, time2: Time coordinates [N] and [M]
        coords1, coords2: Spatial coordinates [N, 3] and [M, 3] (lat, lon, alt)
        time_weight, space_weight: Weighting factors
        
    Returns:
        distances: [N, M] distance matrix
    """
    # Temporal distance
    time_dist = np.abs(time1[:, None] - time2[None, :]) * time_weight
    
    # Spatial distance (Euclidean in normalized space)
    space_dist = np.sqrt(
        np.sum((coords1[:, None, :] - coords2[None, :, :]) ** 2, axis=2)
    ) * space_weight
    
    # Combined spacetime distance
    total_dist = np.sqrt(time_dist ** 2 + space_dist ** 2)
    
    return total_dist
