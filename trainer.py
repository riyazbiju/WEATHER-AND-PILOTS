"""
Training loop for Latent ODE model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time


class Trainer:
    """Handles training and validation of Latent ODE model"""
    
    def __init__(self, model, train_loader, val_loader, config, ode_solver):
        """
        Args:
            model: LatentODEModel instance
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration object
            ode_solver: ODESolver instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.ode_solver = ode_solver
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Device
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Mixed precision training (AMP)
        self.use_amp = config.USE_AMP and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("Mixed precision training (AMP) enabled")
        
        # Checkpointing
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def get_kl_weight(self, epoch):
        """KL annealing schedule"""
        if epoch < self.config.KL_ANNEALING_EPOCHS:
            return epoch / self.config.KL_ANNEALING_EPOCHS
        return 1.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n_batches = 0
        
        kl_weight = self.get_kl_weight(self.epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            query_coords = batch['query_coords'].to(self.device)
            query_features = batch['query_features'].to(self.device)
            context_coords = batch['context_coords'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=self.use_amp):
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
                    kl_weight=kl_weight
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRAD_CLIP
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'kl_w': kl_weight
            })
        
        # Average metrics
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl,
            'kl_weight': kl_weight
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n_batches = 0
        
        kl_weight = 1.0  # Always use full KL weight for validation
        
        for batch in self.val_loader:
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
                kl_weight=kl_weight
            )
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1
        
        # Average metrics
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {self.epoch}")
    
    def train(self, num_epochs=None):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train (default: from config)
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            # Validate (only every VAL_FREQUENCY epochs)
            should_validate = (epoch + 1) % self.config.VAL_FREQUENCY == 0 or epoch == num_epochs - 1
            
            if should_validate:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics)
                
                # Update learning rate
                self.scheduler.step(val_metrics['loss'])
                
                # Log with validation
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Recon: {train_metrics['recon_loss']:.4f}, "
                      f"KL: {train_metrics['kl_loss']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Recon: {val_metrics['recon_loss']:.4f}, "
                      f"KL: {val_metrics['kl_loss']:.4f}")
                
                # Save checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    print(f"  New best validation loss: {self.best_val_loss:.4f}")
                
                if (epoch + 1) % self.config.SAVE_INTERVAL == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            else:
                # Log training only
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Recon: {train_metrics['recon_loss']:.4f}, "
                      f"KL: {train_metrics['kl_loss']:.4f}")
        
        # Training complete
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total time: {elapsed / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Save final training history
        self.save_training_history()
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'num_epochs': self.epoch + 1
        }
        
        path = self.checkpoint_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {path}")
