"""
Quick evaluation script to test the trained model
"""
import torch
from evaluator import Evaluator
from models import LatentODEModel
from data_preprocessing import prepare_data
from dataset import create_dataloaders
from config import Config

def main():
    config = Config()
    
    # Load data
    print("Loading data...")
    train_df, val_df, test_df, normalizer = prepare_data(
        config.DATA_PATH,
        split_type='random',
        test_ratio=0.15,
        val_ratio=0.15
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=config.BATCH_SIZE,
        k_neighbors=20,  # Same as quick start
        time_weight=config.TIME_WEIGHT,
        space_weight=config.SPACE_WEIGHT,
        enforce_causality=False
    )
    
    # Create model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use quick start config values
    model = LatentODEModel(
        latent_dim=32,  # Quick start value
        n_features=6,  # 5 features + 1 from sin/cos expansion
        hidden_dim=128,
        ode_solver_method='rk4',
        n_ode_steps=5
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    model.eval()
    
    # Create evaluator
    evaluator = Evaluator(model, normalizer, device)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80 + "\n")
    
    metrics = evaluator.evaluate(test_loader, split='Test')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return metrics

if __name__ == "__main__":
    main()
