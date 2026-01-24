"""
Helper functions for handling circular variables and denormalization
"""

import numpy as np
import torch


def circular_mean_loss(pred_sin, pred_cos, true_sin, true_cos):
    """
    Compute loss for circular variables using sin/cos encoding.
    This properly handles the circular nature (0° = 360°).
    
    Args:
        pred_sin, pred_cos: Predicted sin/cos values
        true_sin, true_cos: True sin/cos values
        
    Returns:
        loss: Angular distance loss
    """
    # Angular distance: 1 - cos(θ_pred - θ_true)
    # Using identity: cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
    cos_diff = pred_sin * true_sin + pred_cos * true_cos
    
    # Loss: 1 - cos(diff) = angular distance normalized to [0, 2]
    loss = 1.0 - cos_diff
    
    return loss.mean()


def sincos_to_degrees(sin_vals, cos_vals):
    """
    Convert sin/cos encoding back to degrees [0, 360)
    
    Args:
        sin_vals: Sine values
        cos_vals: Cosine values
        
    Returns:
        degrees: Angles in degrees [0, 360)
    """
    if isinstance(sin_vals, torch.Tensor):
        angles = torch.atan2(sin_vals, cos_vals)
        degrees = torch.rad2deg(angles)
        degrees = (degrees + 360) % 360
    else:
        angles = np.arctan2(sin_vals, cos_vals)
        degrees = np.rad2deg(angles)
        degrees = (degrees + 360) % 360
    
    return degrees


def degrees_to_sincos(degrees):
    """
    Convert degrees to sin/cos encoding
    
    Args:
        degrees: Angles in degrees
        
    Returns:
        sin_vals, cos_vals: Sin and cos components
    """
    if isinstance(degrees, torch.Tensor):
        radians = torch.deg2rad(degrees)
        return torch.sin(radians), torch.cos(radians)
    else:
        radians = np.deg2rad(degrees)
        return np.sin(radians), np.cos(radians)


def circular_mae(pred_degrees, true_degrees):
    """
    Compute MAE for circular variable (handles wrap-around)
    
    Args:
        pred_degrees: Predicted angles in degrees
        true_degrees: True angles in degrees
        
    Returns:
        mae: Circular mean absolute error
    """
    # Compute angular difference
    diff = pred_degrees - true_degrees
    
    # Wrap to [-180, 180]
    if isinstance(diff, torch.Tensor):
        diff = ((diff + 180) % 360) - 180
        mae = torch.abs(diff).mean()
    else:
        diff = ((diff + 180) % 360) - 180
        mae = np.abs(diff).mean()
    
    return mae


def denormalize_predictions(predictions, targets, preprocessor, config):
    """
    Denormalize predictions and targets, handling circular variables.
    
    Args:
        predictions: [N, F] predictions (may include sin/cos)
        targets: [N, F] targets (may include sin/cos)
        preprocessor: DataPreprocessor instance
        config: Config instance
        
    Returns:
        pred_denorm: [N, n_vars] denormalized predictions
        target_denorm: [N, n_vars] denormalized targets
        feature_names: List of feature names
    """
    # Build mapping of feature indices
    feature_idx = 0
    pred_denorm = np.zeros((predictions.shape[0], len(config.FEATURE_COLS)))
    target_denorm = np.zeros((targets.shape[0], len(config.FEATURE_COLS)))
    
    non_circular_features = []
    non_circular_indices = []
    
    for i, feature in enumerate(config.FEATURE_COLS):
        if feature in config.CIRCULAR_VARS:
            # Circular variable: convert sin/cos back to degrees
            pred_sin = predictions[:, feature_idx]
            pred_cos = predictions[:, feature_idx + 1]
            true_sin = targets[:, feature_idx]
            true_cos = targets[:, feature_idx + 1]
            
            pred_denorm[:, i] = sincos_to_degrees(pred_sin, pred_cos)
            target_denorm[:, i] = sincos_to_degrees(true_sin, true_cos)
            
            feature_idx += 2  # Skip both sin and cos
        else:
            # Non-circular variable: collect for inverse transform
            non_circular_features.append(feature)
            non_circular_indices.append((i, feature_idx))
            feature_idx += 1
    
    # Inverse transform non-circular features
    if non_circular_features:
        non_circ_pred = predictions[:, [idx for _, idx in non_circular_indices]]
        non_circ_target = targets[:, [idx for _, idx in non_circular_indices]]
        
        pred_denorm_nc = preprocessor.feature_scaler.inverse_transform(non_circ_pred)
        target_denorm_nc = preprocessor.feature_scaler.inverse_transform(non_circ_target)
        
        for j, (orig_idx, _) in enumerate(non_circular_indices):
            pred_denorm[:, orig_idx] = pred_denorm_nc[:, j]
            target_denorm[:, orig_idx] = target_denorm_nc[:, j]
    
    return pred_denorm, target_denorm, config.FEATURE_COLS


def compute_mae_with_units(predictions, targets, feature_names, units):
    """
    Compute MAE for each feature with proper units.
    Handles circular variables specially.
    
    Args:
        predictions: [N, F] denormalized predictions
        targets: [N, F] denormalized targets
        feature_names: List of feature names
        units: Dictionary of units per feature
        
    Returns:
        mae_dict: Dictionary with MAE and units per feature
    """
    mae_dict = {}
    
    for i, feature in enumerate(feature_names):
        pred = predictions[:, i]
        true = targets[:, i]
        
        # Check if circular
        if 'direction' in feature.lower():
            # Circular MAE
            mae_val = circular_mae(pred, true)
        else:
            # Regular MAE
            if isinstance(pred, torch.Tensor):
                mae_val = torch.abs(pred - true).mean().item()
            else:
                mae_val = np.abs(pred - true).mean()
        
        unit = units.get(feature, '')
        
        mae_dict[feature] = {
            'MAE': mae_val,
            'unit': unit
        }
    
    return mae_dict
