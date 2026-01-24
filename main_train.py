"""
Main training script for Set-Based Latent ODE model

Usage:
    python main_train.py --split random --epochs 50
    python main_train.py --split temporal --epochs 100
    python main_train.py --split station --epochs 100
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import json

# Import all modules
from config import Config
from data_preprocessing import prepare_data
from dataset import create_dataloaders
from models import LatentODEModel
from ode_solver import ODESolver, SimpleODESolver
from trainer import Trainer
from evaluator import Evaluator
from visualization import (
    plot_training_curves,
    plot_predictions_vs_targets,
    plot_residuals,
    create_evaluation_report
)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """Main training pipeline"""
    
    # Initialize config
    config = Config()
    config.SPLIT_TYPE = args.split
    config.NUM_EPOCHS = args.epochs
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.k_neighbors:
        config.K_NEIGHBORS = args.k_neighbors
    if args.latent_dim:
        config.LATENT_DIM = args.latent_dim
    
    # Set seed
    set_seed(config.SEED)
    
    print("\n" + "="*80)
    print("SET-BASED LATENT ODE FOR WEATHER PREDICTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Split type: {config.SPLIT_TYPE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  K neighbors: {config.K_NEIGHBORS}")
    print(f"  Latent dim: {config.LATENT_DIM}")
    print(f"  Device: {config.DEVICE}")
    print(f"  ODE solver: {config.ODE_SOLVER}")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # 1. DATA PREPARATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80 + "\n")
    
    data_dict = prepare_data(config, split_type=config.SPLIT_TYPE)
    
    # -------------------------------------------------------------------------
    # 2. CREATE DATALOADERS
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: CREATE DATALOADERS")
    print("="*80 + "\n")
    
    loader_dict = create_dataloaders(data_dict, config, split_type=config.SPLIT_TYPE)
    
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # -------------------------------------------------------------------------
    # 3. CREATE MODEL
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: CREATE MODEL")
    print("="*80 + "\n")
    
    model = LatentODEModel(config)
    
    print(f"Model created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # -------------------------------------------------------------------------
    # 4. CREATE ODE SOLVER
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: CREATE ODE SOLVER")
    print("="*80 + "\n")
    
    try:
        # Try to use torchdiffeq
        ode_solver = ODESolver(
            method=config.ODE_SOLVER,
            rtol=config.ODE_RTOL,
            atol=config.ODE_ATOL
        )
        print(f"Using torchdiffeq ODE solver: {config.ODE_SOLVER}")
    except ImportError:
        # Fallback to simple solver
        print("torchdiffeq not found, using SimpleODESolver")
        ode_solver = SimpleODESolver(method='rk4', n_steps=10)
    
    # -------------------------------------------------------------------------
    # 5. TRAINING
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: TRAINING")
    print("="*80 + "\n")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        ode_solver=ode_solver
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=config.NUM_EPOCHS)
    
    # Plot training curves
    history_path = config.CHECKPOINT_DIR + '/training_history.json'
    if Path(history_path).exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        plot_training_curves(
            history,
            save_path=f"{config.RESULTS_DIR}/training_curves.png"
        )
    
    # -------------------------------------------------------------------------
    # 6. EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: EVALUATION")
    print("="*80 + "\n")
    
    # Load best model
    best_model_path = Path(config.CHECKPOINT_DIR) / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        ode_solver=ode_solver,
        preprocessor=data_dict['preprocessor']
    )
    
    # Evaluate
    metrics, predictions, targets = evaluator.evaluate(return_predictions=True)
    
    # Stratified evaluation
    print("\n" + "-"*80)
    altitude_results = evaluator.evaluate_by_altitude(n_altitude_bins=5)
    
    print("\n" + "-"*80)
    time_results = evaluator.evaluate_by_time(n_time_bins=5)
    
    # -------------------------------------------------------------------------
    # 7. VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 7: VISUALIZATION")
    print("="*80 + "\n")
    
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    # Prediction vs target plots
    plot_predictions_vs_targets(
        predictions,
        targets,
        config.FEATURE_COLS,
        save_path=results_dir / 'predictions_vs_targets.png'
    )
    
    # Residual plots
    plot_residuals(
        predictions,
        targets,
        config.FEATURE_COLS,
        save_path=results_dir / 'residuals.png'
    )
    
    # Create comprehensive report
    create_evaluation_report(metrics, save_dir=results_dir)
    
    print("\n" + "="*80)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Set-Based Latent ODE model')
    
    # Training arguments
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'temporal', 'station'],
                        help='Data split strategy')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    
    # Model arguments
    parser.add_argument('--k_neighbors', type=int, default=None,
                        help='Number of context neighbors')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent ODE dimension')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    main(args)
