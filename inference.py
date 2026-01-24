"""
Inference script for making predictions with trained Latent ODE model

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --split test
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from data_preprocessing import DataPreprocessor
from models import LatentODEModel
from ode_solver import ODESolver, SimpleODESolver
from context_selection import SpacetimeContextSelector, ContextDataCollator


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint['config']
    
    # Create model
    model = LatentODEModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, config


def predict_single_observation(
    model,
    query_coords,
    context_selector,
    collator,
    query_idx,
    ode_solver,
    device
):
    """
    Make prediction for a single observation
    
    Args:
        model: Trained LatentODEModel
        query_coords: Query coordinates (time, lat, lon, alt)
        context_selector: Context selector with built index
        collator: Data collator
        query_idx: Index of query in observations
        ode_solver: ODE solver
        device: torch device
        
    Returns:
        prediction: [F] predicted features
    """
    # Get context
    context_indices = context_selector.get_context(query_idx)
    
    # Collate batch
    batch = collator.collate([(query_idx, context_indices)])
    
    # Move to device
    query_coords_t = batch['query_coords'].to(device)
    context_coords = batch['context_coords'].to(device)
    context_features = batch['context_features'].to(device)
    context_mask = batch['context_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        prediction, _, _ = model(
            context_features,
            context_coords,
            context_mask,
            query_coords_t,
            ode_solver=ode_solver
        )
    
    return prediction.cpu().numpy()[0]


def predict_batch(
    model,
    test_df,
    config,
    ode_solver,
    preprocessor,
    device='cpu',
    n_samples=None
):
    """
    Make predictions for a batch of test observations
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        config: Configuration
        ode_solver: ODE solver
        preprocessor: Data preprocessor
        device: torch device
        n_samples: Number of samples to predict (None = all)
        
    Returns:
        DataFrame with predictions and targets
    """
    # Build context selector
    context_selector = SpacetimeContextSelector(config, enforce_causality=False)
    context_selector.build_index(test_df, verbose=False)
    
    collator = ContextDataCollator(config, test_df)
    
    # Sample indices
    if n_samples is not None:
        indices = np.random.choice(len(test_df), min(n_samples, len(test_df)), replace=False)
    else:
        indices = np.arange(len(test_df))
    
    # Collect predictions
    predictions = []
    targets = []
    coords_list = []
    
    print(f"Making predictions for {len(indices)} observations...")
    
    for idx in indices:
        # Get query
        query = test_df.iloc[idx]
        
        # Predict
        pred = predict_single_observation(
            model,
            query[['time_hours_norm', 'latitude', 'longitude', 'altitude']].values,
            context_selector,
            collator,
            idx,
            ode_solver,
            device
        )
        
        # Store
        predictions.append(pred)
        targets.append(query[config.FEATURE_COLS].values)
        coords_list.append({
            'time_hours': query['time_hours'],
            'latitude': query['latitude'],
            'longitude': query['longitude'],
            'altitude': query['altitude'],
            'station': query['station']
        })
    
    # Convert to arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Inverse transform
    predictions_orig = preprocessor.inverse_transform_features(predictions)
    targets_orig = preprocessor.inverse_transform_features(targets)
    
    # Create results DataFrame
    results = pd.DataFrame(coords_list)
    
    for i, feature in enumerate(config.FEATURE_COLS):
        results[f'{feature}_true'] = targets_orig[:, i]
        results[f'{feature}_pred'] = predictions_orig[:, i]
        results[f'{feature}_error'] = predictions_orig[:, i] - targets_orig[:, i]
    
    return results


def main(args):
    """Main inference pipeline"""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model, config = load_model(args.checkpoint, device=device)
    
    # Load preprocessor
    preprocessor = DataPreprocessor(config)
    preprocessor.load_scalers('scalers.pkl')
    
    # Load test data
    print(f"\nLoading test data...")
    df = preprocessor.load_and_preprocess(config.DATA_PATH, verbose=False)
    
    # Apply same split as training
    from data_preprocessing import DataSplitter
    
    if config.SPLIT_TYPE == 'random':
        _, _, test_df = DataSplitter.random_split(df, config.TRAIN_RATIO, config.VAL_RATIO, config.SEED)
    elif config.SPLIT_TYPE == 'temporal':
        _, _, test_df = DataSplitter.temporal_split(df, config.TRAIN_RATIO, config.VAL_RATIO)
    elif config.SPLIT_TYPE == 'station':
        _, _, test_df = DataSplitter.station_split(df, config.STATION_HOLDOUT, seed=config.SEED)
    
    # Normalize
    test_df_norm = preprocessor.transform(test_df)
    
    print(f"Test set size: {len(test_df_norm)}")
    
    # Create ODE solver
    try:
        ode_solver = ODESolver(method=config.ODE_SOLVER, rtol=config.ODE_RTOL, atol=config.ODE_ATOL)
    except ImportError:
        ode_solver = SimpleODESolver(method='rk4', n_steps=10)
    
    # Make predictions
    results = predict_batch(
        model=model,
        test_df=test_df_norm,
        config=config,
        ode_solver=ode_solver,
        preprocessor=preprocessor,
        device=device,
        n_samples=args.n_samples
    )
    
    # Save results
    output_path = args.output if args.output else 'predictions.csv'
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    
    for feature in config.FEATURE_COLS:
        error = results[f'{feature}_error']
        print(f"\n{feature.upper()}:")
        print(f"  Mean error: {error.mean():.4f}")
        print(f"  Std error: {error.std():.4f}")
        print(f"  RMSE: {np.sqrt((error**2).mean()):.4f}")
        print(f"  MAE: {np.abs(error).mean():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions with trained Latent ODE model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file for predictions')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to predict (default: all)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    
    main(args)
