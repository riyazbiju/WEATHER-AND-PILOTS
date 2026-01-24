"""
Visualization utilities for Latent ODE model
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(epochs, [x['loss'] for x in train_losses], label='Train', marker='o')
    axes[0].plot(epochs, [x['loss'] for x in val_losses], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss (ELBO)')
    axes[0].set_title('Training Curves - Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, [x['recon_loss'] for x in train_losses], label='Train', marker='o')
    axes[1].plot(epochs, [x['recon_loss'] for x in val_losses], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss (MSE)')
    axes[1].set_title('Training Curves - Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(epochs, [x['kl_loss'] for x in train_losses], label='Train', marker='o')
    axes[2].plot(epochs, [x['kl_loss'] for x in val_losses], label='Val', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('Training Curves - KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_predictions_vs_targets(predictions, targets, feature_names, save_path=None):
    """
    Scatter plots of predictions vs targets for each feature
    
    Args:
        predictions: [N, F] predicted values
        targets: [N, F] target values
        feature_names: List of feature names
        save_path: Path to save figure
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        pred = predictions[:, i]
        targ = targets[:, i]
        
        # Scatter plot
        ax.scatter(targ, pred, alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val = min(targ.min(), pred.min())
        max_val = max(targ.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # Compute R²
        from sklearn.metrics import r2_score
        r2 = r2_score(targ, pred)
        
        ax.set_xlabel(f'True {feature_name}')
        ax.set_ylabel(f'Predicted {feature_name}')
        ax.set_title(f'{feature_name} (R² = {r2:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plots saved to {save_path}")
    
    plt.show()


def plot_residuals(predictions, targets, feature_names, save_path=None):
    """
    Plot residual distributions for each feature
    
    Args:
        predictions: [N, F] predicted values
        targets: [N, F] target values
        feature_names: List of feature names
        save_path: Path to save figure
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        residuals = predictions[:, i] - targets[:, i]
        
        # Histogram
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero residual')
        ax.axvline(residuals.mean(), color='g', linestyle='--', lw=2, label=f'Mean: {residuals.mean():.3f}')
        
        ax.set_xlabel('Residual (Predicted - True)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature_name} Residuals (Std: {residuals.std():.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plots saved to {save_path}")
    
    plt.show()


def plot_metrics_by_altitude(results, save_path=None):
    """
    Plot performance metrics stratified by altitude
    
    Args:
        results: List of dictionaries with altitude-stratified results
        save_path: Path to save figure
    """
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = range(len(df))
    
    # RMSE
    axes[0].bar(x, df['rmse'])
    axes[0].set_xlabel('Altitude Bin')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Altitude')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Bin {i+1}" for i in x], rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].bar(x, df['mae'])
    axes[1].set_xlabel('Altitude Bin')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE by Altitude')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Bin {i+1}" for i in x], rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].bar(x, df['r2'])
    axes[2].set_xlabel('Altitude Bin')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² by Altitude')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Bin {i+1}" for i in x], rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Altitude stratified plots saved to {save_path}")
    
    plt.show()


def plot_metrics_by_time(results, save_path=None):
    """
    Plot performance metrics stratified by time
    
    Args:
        results: List of dictionaries with time-stratified results
        save_path: Path to save figure
    """
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = range(len(df))
    
    # RMSE
    axes[0].plot(x, df['rmse'], marker='o', linewidth=2)
    axes[0].set_xlabel('Time Bin')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Time')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Bin {i+1}" for i in x])
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(x, df['mae'], marker='o', linewidth=2)
    axes[1].set_xlabel('Time Bin')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE by Time')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Bin {i+1}" for i in x])
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].plot(x, df['r2'], marker='o', linewidth=2)
    axes[2].set_xlabel('Time Bin')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² by Time')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Bin {i+1}" for i in x])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time stratified plots saved to {save_path}")
    
    plt.show()


def plot_feature_comparison(predictions, targets, feature_names, n_samples=100, save_path=None):
    """
    Plot time series comparison for random samples
    
    Args:
        predictions: [N, F] predicted values
        targets: [N, F] target values
        feature_names: List of feature names
        n_samples: Number of samples to plot
        save_path: Path to save figure
    """
    n_features = len(feature_names)
    
    # Select random samples
    indices = np.random.choice(len(predictions), min(n_samples, len(predictions)), replace=False)
    indices = np.sort(indices)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        ax.plot(indices, targets[indices, i], 'o-', label='True', alpha=0.7, markersize=4)
        ax.plot(indices, predictions[indices, i], 's-', label='Predicted', alpha=0.7, markersize=4)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(feature_name)
        ax.set_title(f'{feature_name} - True vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature comparison saved to {save_path}")
    
    plt.show()


def create_evaluation_report(metrics, save_dir='results'):
    """
    Create comprehensive evaluation report with all visualizations
    
    Args:
        metrics: Dictionary with evaluation metrics
        save_dir: Directory to save all visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING EVALUATION REPORT")
    print("="*80)
    
    # Save metrics to file
    import json
    metrics_path = save_dir / 'metrics.json'
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_path}")
    print(f"All visualizations saved to {save_dir}/")
    print("="*80 + "\n")
