"""
Quick test script to verify installation and data loading
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("LATENT ODE - INSTALLATION CHECK")
print("="*80)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check package versions
packages = {
    'torch': torch.__version__,
    'numpy': np.__version__,
    'pandas': pd.__version__,
}

print("\nPackage versions:")
for pkg, version in packages.items():
    print(f"  {pkg}: {version}")

# Check CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Check torchdiffeq
try:
    import torchdiffeq
    print(f"\ntorchdiffeq: Available")
except ImportError:
    print(f"\ntorchdiffeq: NOT FOUND (will use fallback ODE solver)")

# Check data file
data_path = Path("data_5_with_pressure.csv")
if data_path.exists():
    print(f"\nData file: FOUND")
    
    # Load first few rows
    df = pd.read_csv(data_path, nrows=10)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample shape: {df.shape}")
else:
    print(f"\nData file: NOT FOUND")
    print(f"  Expected path: {data_path.absolute()}")

# Test basic imports
print("\nTesting module imports...")
try:
    from config import Config
    print("  ✓ config")
    from data_preprocessing import prepare_data
    print("  ✓ data_preprocessing")
    from models import LatentODEModel
    print("  ✓ models")
    from ode_solver import ODESolver
    print("  ✓ ode_solver")
    from context_selection import SpacetimeContextSelector
    print("  ✓ context_selection")
    from trainer import Trainer
    print("  ✓ trainer")
    from evaluator import Evaluator
    print("  ✓ evaluator")
    
    print("\nAll imports successful! ✓")
    
    # Test model creation
    print("\nTesting model creation...")
    config = Config()
    model = LatentODEModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created with {n_params:,} parameters ✓")
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Check complete!")
print("="*80)
print("\nTo start training, run:")
print("  python main_train.py --split random --epochs 10")
print("\n")
