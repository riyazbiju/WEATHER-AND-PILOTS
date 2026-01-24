"""
Model analysis utilities: inspect latent space, visualize ODE trajectories
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class LatentSpaceAnalyzer:
    """Analyze and visualize latent space of trained model"""
    
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def encode_batch(self, context_features, context_coords, context_mask):
        """Encode batch to latent space"""
        z0_mean, z0_logvar = self.model.encode(
            context_features.to(self.device),
            context_coords.to(self.device),
            context_mask.to(self.device)
        )
        return z0_mean.cpu().numpy(), z0_logvar.cpu().numpy()
    
    @torch.no_grad()
    def trace_ode_trajectory(self, z0, t_start, t_end, n_steps=50):
        """
        Trace ODE trajectory from z0 at t_start to t_end
        
        Returns:
            times: [n_steps] time points
            states: [n_steps, latent_dim] latent states
        """
        times = np.linspace(t_start, t_end, n_steps)
        states = []
        
        z = z0.to(self.device)
        
        for i in range(len(times) - 1):
            states.append(z.cpu().numpy())
            
            # Compute derivative
            t = torch.tensor(times[i], device=self.device, dtype=z.dtype)
            dz = self.model.ode_func(t, z.unsqueeze(0)).squeeze(0)
            
            # Euler step
            dt = times[i+1] - times[i]
            z = z + dt * dz
        
        states.append(z.cpu().numpy())
        states = np.array(states)
        
        return times, states
    
    def visualize_latent_trajectories(self, dataloader, n_samples=10, save_path=None):
        """Visualize ODE trajectories in latent space (2D projection)"""
        
        # Collect some samples
        trajectories = []
        
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            
            # Encode to latent
            z0_mean, _ = self.encode_batch(
                batch['context_features'],
                batch['context_coords'],
                batch['context_mask']
            )
            
            # Get times
            query_time = batch['query_coords'][0, 0].item()
            context_time = batch['context_coords'][0, :, 0].mean().item()
            
            # Trace trajectory
            if abs(query_time - context_time) > 0.01:
                times, states = self.trace_ode_trajectory(
                    torch.tensor(z0_mean[0]),
                    context_time,
                    query_time,
                    n_steps=20
                )
                trajectories.append((times, states))
        
        # Plot (first 2 latent dimensions)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, (times, states) in enumerate(trajectories):
            # Plot trajectory
            ax.plot(states[:, 0], states[:, 1], '-', alpha=0.6, linewidth=2)
            
            # Mark start and end
            ax.scatter(states[0, 0], states[0, 1], c='green', s=100, marker='o', zorder=10)
            ax.scatter(states[-1, 0], states[-1, 1], c='red', s=100, marker='s', zorder=10)
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title('ODE Trajectories in Latent Space')
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='End')
        ]
        ax.legend(handles=legend_elements)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to {save_path}")
        
        plt.show()
    
    def analyze_latent_statistics(self, dataloader, n_batches=50):
        """Compute statistics of latent representations"""
        
        all_means = []
        all_logvars = []
        
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            
            z0_mean, z0_logvar = self.encode_batch(
                batch['context_features'],
                batch['context_coords'],
                batch['context_mask']
            )
            
            all_means.append(z0_mean)
            all_logvars.append(z0_logvar)
        
        all_means = np.concatenate(all_means, axis=0)
        all_logvars = np.concatenate(all_logvars, axis=0)
        
        print("\n" + "="*80)
        print("LATENT SPACE STATISTICS")
        print("="*80)
        
        print(f"\nLatent dimension: {all_means.shape[1]}")
        print(f"Number of samples: {all_means.shape[0]}")
        
        print(f"\nMean statistics:")
        print(f"  Mean of means: {all_means.mean():.4f}")
        print(f"  Std of means: {all_means.std():.4f}")
        print(f"  Min: {all_means.min():.4f}, Max: {all_means.max():.4f}")
        
        print(f"\nLog-variance statistics:")
        print(f"  Mean: {all_logvars.mean():.4f}")
        print(f"  Std: {all_logvars.std():.4f}")
        
        # Per-dimension statistics
        print(f"\nPer-dimension mean magnitude:")
        dim_means = np.abs(all_means).mean(axis=0)
        print(f"  Top 5 dimensions: {np.argsort(dim_means)[-5:][::-1]}")
        print(f"  Values: {dim_means[np.argsort(dim_means)[-5:][::-1]]}")
        
        return {
            'means': all_means,
            'logvars': all_logvars,
            'mean_mean': all_means.mean(),
            'mean_std': all_means.std()
        }
    
    def visualize_attention_weights(self, batch, save_path=None):
        """Visualize attention weights from encoder (if using attention aggregation)"""
        
        if self.config.AGGREGATION != 'attention':
            print("Model does not use attention aggregation")
            return
        
        # Forward pass through encoder
        context_features = batch['context_features'].to(self.device)
        context_coords = batch['context_coords'].to(self.device)
        context_mask = batch['context_mask'].to(self.device)
        
        # Get attention weights (need to modify model to return these)
        # This is a simplified version
        print("Attention weight visualization requires model modification")
        print("Add return_attention=True option to SetEncoder.forward()")


def load_and_analyze(checkpoint_path, dataloader):
    """Convenience function to load model and analyze"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    from models import LatentODEModel
    model = LatentODEModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create analyzer
    analyzer = LatentSpaceAnalyzer(model, config, device=config.DEVICE)
    
    # Run analyses
    print(f"\nLoaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Latent statistics
    stats = analyzer.analyze_latent_statistics(dataloader, n_batches=50)
    
    # Visualize trajectories
    analyzer.visualize_latent_trajectories(
        dataloader,
        n_samples=10,
        save_path='latent_trajectories.png'
    )
    
    return analyzer, stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze latent space of trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which data split to analyze')
    
    args = parser.parse_args()
    
    # Load data
    from config import Config
    from data_preprocessing import prepare_data
    from dataset import create_dataloaders
    
    config = Config()
    data_dict = prepare_data(config, split_type='random')
    loader_dict = create_dataloaders(data_dict, config, split_type='random')
    
    # Analyze
    dataloader = loader_dict[args.split]
    analyzer, stats = load_and_analyze(args.checkpoint, dataloader)
    
    print("\nAnalysis complete!")
