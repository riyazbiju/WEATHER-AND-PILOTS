"""
Example: Training a minimal Latent ODE model (Quick Start)
This script trains a small model for demonstration purposes
"""

import torch
from config import Config
from data_preprocessing import prepare_data
from dataset import create_dataloaders
from models import LatentODEModel
from ode_solver import SimpleODESolver
from trainer import Trainer

print("\n" + "="*80)
print("QUICK START: Training a Latent ODE Model")
print("="*80 + "\n")

# 1. Create a minimal config
config = Config()
config.NUM_EPOCHS = 5  # Just 5 epochs for quick demo
config.BATCH_SIZE = 64
config.LATENT_DIM = 32  # Smaller model
config.HIDDEN_DIM = 64
config.K_NEIGHBORS = 20
config.SPLIT_TYPE = 'random'

print("Configuration:")
print(f"  Epochs: {config.NUM_EPOCHS}")
print(f"  Batch size: {config.BATCH_SIZE}")
print(f"  Latent dim: {config.LATENT_DIM}")
print(f"  K neighbors: {config.K_NEIGHBORS}")
print(f"  Device: {config.DEVICE}")

# 2. Prepare data (this will take a moment)
print("\nPreparing data...")
data_dict = prepare_data(config, split_type='random')

print(f"\nData split:")
print(f"  Train: {len(data_dict['train'])} observations")
print(f"  Val: {len(data_dict['val'])} observations")
print(f"  Test: {len(data_dict['test'])} observations")

# 3. Create dataloaders
print("\nCreating dataloaders...")
loader_dict = create_dataloaders(data_dict, config, split_type='random')

train_loader = loader_dict['train']
val_loader = loader_dict['val']

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# 4. Create model
print("\nCreating model...")
model = LatentODEModel(config)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {n_params:,}")

# 5. Create ODE solver (use simple solver for speed)
ode_solver = SimpleODESolver(method='rk4', n_steps=5)
print(f"  ODE solver: RK4 with 5 steps")

# 6. Create trainer
print("\nCreating trainer...")
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    ode_solver=ode_solver
)

# 7. Train!
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

trainer.train(num_epochs=config.NUM_EPOCHS)

print("\n" + "="*80)
print("QUICK START COMPLETE!")
print("="*80)
print(f"\nModel saved to: {config.CHECKPOINT_DIR}/best_model.pt")
print("\nTo train a full model, run:")
print("  python main_train.py --split random --epochs 50")
print("\n")
