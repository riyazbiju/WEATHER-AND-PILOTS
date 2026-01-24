"""
Evaluation module for Latent ODE model
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pandas as pd
from circular_utils import (
    denormalize_predictions, 
    compute_mae_with_units, 
    circular_mae,
    circular_mean_loss
)


class Evaluator:
    """Evaluates model performance on test data"""
    
    def __init__(self, model, test_loader, config, ode_solver, preprocessor):
        """
        Args:
            model: Trained LatentODEModel
            test_loader: Test dataloader
            config: Configuration object
            ode_solver: ODESolver instance
            preprocessor: DataPreprocessor for inverse transform
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.ode_solver = ode_solver
        self.preprocessor = preprocessor
        self.device = config.DEVICE
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, return_predictions=False):
        """
        Evaluate model on test set
        
        Args:
            return_predictions: If True, return all predictions and targets
            
        Returns:
            metrics: Dictionary with evaluation metrics
            predictions: (optional) Array of predictions
            targets: (optional) Array of targets
        """
        all_predictions = []
        all_targets = []
        all_losses = []
        
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            # Move to device
            query_coords = batch['query_coords'].to(self.device)
            query_features = batch['query_features'].to(self.device)
            context_coords = batch['context_coords'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            
            # Forward pass
            predictions, z0_mean, z0_logvar = self.model(
                context_features,
                context_coords,
                context_mask,
                query_coords,
                ode_solver=self.ode_solver
            )
            
            # Compute loss
            loss, recon_loss, kl_loss = self.model.compute_loss(
                predictions,
                query_features,
                z0_mean,
                z0_logvar,
                kl_weight=1.0
            )
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(query_features.cpu().numpy())
            all_losses.append(recon_loss.item())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Denormalize predictions properly (handles circular variables)
        predictions_orig, targets_orig, feature_names = denormalize_predictions(
            predictions, targets, self.preprocessor, self.config
        )
        
        # Compute metrics in original scale with proper units
        metrics_orig = self.compute_metrics_with_units(predictions_orig, targets_orig)
        
        # Combine metrics
        metrics = {
            'original': metrics_orig,
            'avg_loss': np.mean(all_losses)
        }
        
        # Print summary
        self.print_metrics_with_units(metrics)
        
        if return_predictions:
            return metrics, predictions_orig, targets_orig
        else:
            return metrics
    
    def compute_metrics_with_units(self, predictions, targets):
        """
        Compute evaluation metrics per feature with proper units
        Handles circular variables correctly
        
        Args:
            predictions: [N, F] predicted features (denormalized)
            targets: [N, F] ground truth features (denormalized)
            
        Returns:
            Dictionary with metrics per feature including units
        """
        feature_names = self.config.FEATURE_COLS
        units = self.config.FEATURE_UNITS
        
        metrics = {}
        
        for i, feature_name in enumerate(feature_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            # Check if circular variable
            is_circular = feature_name in self.config.CIRCULAR_VARS
            
            if is_circular:
                # Use circular MAE for wind direction
                mae = circular_mae(pred_i, target_i)
                
                # Circular RMSE (angular distance)
                diff = pred_i - target_i
                diff = ((diff + 180) % 360) - 180  # Wrap to [-180, 180]
                rmse = np.sqrt((diff ** 2).mean())
                
                # For R², treat as regular (not perfect but reasonable)
                r2 = r2_score(target_i, pred_i)
            else:
                # Regular metrics
                mse = mean_squared_error(target_i, pred_i)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(target_i, pred_i)
                r2 = r2_score(target_i, pred_i)
            
            # MAPE (avoid division by zero)
            mask = np.abs(target_i) > 1e-8
            mape = np.mean(np.abs((target_i[mask] - pred_i[mask]) / target_i[mask])) * 100 if mask.sum() > 0 else np.nan
            
            metrics[feature_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape if not is_circular else np.nan,
                'unit': units.get(feature_name, ''),
                'is_circular': is_circular
            }
        
        # Overall metrics (averaged across features, excluding circular from MAPE)
        non_circular_features = [f for f in feature_names if f not in self.config.CIRCULAR_VARS]
        
        metrics['overall'] = {
            'MAE': np.mean([metrics[f]['MAE'] for f in feature_names]),
            'RMSE': np.mean([metrics[f]['RMSE'] for f in feature_names]),
            'R2': np.mean([metrics[f]['R2'] for f in feature_names]),
        }
        
        return metrics
    
    def print_metrics_with_units(self, metrics):
        """Print formatted metrics with proper units"""
        print("\n" + "="*80)
        print("EVALUATION METRICS (DENORMALIZED WITH UNITS)")
        print("="*80)
        
        metrics_orig = metrics['original']
        
        # Per-feature metrics
        print("\n" + "-"*80)
        print("PER-FEATURE PERFORMANCE:")
        print("-"*80)
        
        for feature_name in self.config.FEATURE_COLS:
            if feature_name in metrics_orig:
                m = metrics_orig[feature_name]
                unit = m['unit']
                is_circular = m.get('is_circular', False)
                
                print(f"\n{feature_name.upper()} [{unit}]:")
                if is_circular:
                    print(f"  MAE (Circular):  {m['MAE']:.4f} {unit}")
                    print(f"  RMSE (Circular): {m['RMSE']:.4f} {unit}")
                else:
                    print(f"  MAE:   {m['MAE']:.4f} {unit}")
                    print(f"  RMSE:  {m['RMSE']:.4f} {unit}")
                    if not np.isnan(m['MAPE']):
                        print(f"  MAPE:  {m['MAPE']:.2f}%")
                print(f"  R²:    {m['R2']:.4f}")
        
        # Overall metrics
        print("\n" + "-"*80)
        print("OVERALL (averaged across all features):")
        print("-"*80)
        m_overall = metrics_orig['overall']
        print(f"  Average MAE:   {m_overall['MAE']:.4f}")
        print(f"  Average RMSE:  {m_overall['RMSE']:.4f}")
        print(f"  Average R²:    {m_overall['R2']:.4f}")
        
        print("\n" + "="*80)
    
    def evaluate_by_altitude(self, n_altitude_bins=5):
        """Evaluate performance stratified by altitude"""
        print("\nEvaluating performance by altitude...")
        
        all_predictions = []
        all_targets = []
        all_altitudes = []
        
        for batch in tqdm(self.test_loader, desc="Collecting predictions"):
            # Move to device
            query_coords = batch['query_coords'].to(self.device)
            query_features = batch['query_features'].to(self.device)
            context_coords = batch['context_coords'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            
            # Forward pass
            predictions, _, _ = self.model(
                context_features,
                context_coords,
                context_mask,
                query_coords,
                ode_solver=self.ode_solver
            )
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(query_features.cpu().numpy())
            all_altitudes.append(query_coords[:, 3].cpu().numpy())  # Normalized altitude
        
        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        altitudes = np.concatenate(all_altitudes, axis=0)
        
        # Create altitude bins
        altitude_bins = np.percentile(altitudes, np.linspace(0, 100, n_altitude_bins + 1))
        
        # Evaluate per bin
        results = []
        for i in range(n_altitude_bins):
            mask = (altitudes >= altitude_bins[i]) & (altitudes < altitude_bins[i + 1])
            
            if mask.sum() > 0:
                bin_predictions = predictions[mask]
                bin_targets = targets[mask]
                
                # Denormalize properly
                bin_predictions_orig, bin_targets_orig, _ = denormalize_predictions(
                    bin_predictions, bin_targets, self.preprocessor, self.config
                )
                
                # Compute metrics
                bin_metrics = self.compute_metrics_with_units(bin_predictions_orig, bin_targets_orig)
                
                results.append({
                    'altitude_range': f"{altitude_bins[i]:.3f} - {altitude_bins[i+1]:.3f}",
                    'n_samples': mask.sum(),
                    'rmse': bin_metrics['overall']['RMSE'],
                    'mae': bin_metrics['overall']['MAE'],
                    'r2': bin_metrics['overall']['R2']
                })
        
        # Print results
        print("\nPerformance by Altitude:")
        print("-"*80)
        for result in results:
            print(f"Altitude {result['altitude_range']} (n={result['n_samples']}):")
            print(f"  RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}, R²: {result['r2']:.4f}")
        
        return results
    
    def evaluate_by_time(self, n_time_bins=5):
        """Evaluate performance stratified by time"""
        print("\nEvaluating performance by time...")
        
        all_predictions = []
        all_targets = []
        all_times = []
        
        for batch in tqdm(self.test_loader, desc="Collecting predictions"):
            # Move to device
            query_coords = batch['query_coords'].to(self.device)
            query_features = batch['query_features'].to(self.device)
            context_coords = batch['context_coords'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            
            # Forward pass
            predictions, _, _ = self.model(
                context_features,
                context_coords,
                context_mask,
                query_coords,
                ode_solver=self.ode_solver
            )
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(query_features.cpu().numpy())
            all_times.append(query_coords[:, 0].cpu().numpy())  # Normalized time
        
        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        times = np.concatenate(all_times, axis=0)
        
        # Create time bins
        time_bins = np.percentile(times, np.linspace(0, 100, n_time_bins + 1))
        
        # Evaluate per bin
        results = []
        for i in range(n_time_bins):
            mask = (times >= time_bins[i]) & (times < time_bins[i + 1])
            
            if mask.sum() > 0:
                bin_predictions = predictions[mask]
                bin_targets = targets[mask]
                
                # Denormalize properly
                bin_predictions_orig, bin_targets_orig, _ = denormalize_predictions(
                    bin_predictions, bin_targets, self.preprocessor, self.config
                )
                
                # Compute metrics
                bin_metrics = self.compute_metrics_with_units(bin_predictions_orig, bin_targets_orig)
                
                results.append({
                    'time_range': f"{time_bins[i]:.3f} - {time_bins[i+1]:.3f}",
                    'n_samples': mask.sum(),
                    'rmse': bin_metrics['overall']['RMSE'],
                    'mae': bin_metrics['overall']['MAE'],
                    'r2': bin_metrics['overall']['R2']
                })
        
        # Print results
        print("\nPerformance by Time:")
        print("-"*80)
        for result in results:
            print(f"Time {result['time_range']} (n={result['n_samples']}):")
            print(f"  RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}, R²: {result['r2']:.4f}")
        
        return results
