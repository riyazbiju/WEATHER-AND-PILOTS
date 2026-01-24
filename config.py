"""
Configuration file for Set-Based Latent ODE model
"""

import torch

class Config:
    """Model and training configuration"""
    
    # Data parameters
    DATA_PATH = "data_5_with_pressure.csv"
    FEATURE_COLS = ['temperature', 'windspeed', 'winddirection', 'visibility', 'pressure']
    COORD_COLS = ['latitude', 'longitude', 'altitude']
    TIME_COL = 'datetime'
    STATION_COL = 'station'
    
    # Circular variables (for special handling)
    CIRCULAR_VARS = ['winddirection']  # Variables with circular nature (0° = 360°)
    
    # Units for each feature (for reporting)
    FEATURE_UNITS = {
        'temperature': '°C',
        'windspeed': 'knots',
        'winddirection': 'degrees',
        'visibility': 'SM',  # Statute Miles
        'pressure': 'hPa'
    }
    
    # Context selection parameters
    K_NEIGHBORS = 20  # Number of nearest neighbors for context
    TIME_WEIGHT = 1.0  # Weight for temporal distance
    SPACE_WEIGHT = 1.0  # Weight for spatial distance
    MAX_TIME_DIFF = 24.0  # Maximum time difference in hours for context
    
    # Model architecture
    LATENT_DIM = 64  # Dimension of latent ODE state
    HIDDEN_DIM = 128  # Hidden layer dimension
    ENCODER_LAYERS = 3  # Number of layers in encoder MLPs
    DECODER_LAYERS = 3  # Number of layers in decoder
    ODE_HIDDEN_DIM = 64  # Hidden dimension for ODE function
    
    # Aggregation method
    AGGREGATION = 'attention'  # 'mean', 'max', 'attention'
    ATTENTION_HEADS = 4  # Number of attention heads
    
    # Training parameters
    BATCH_SIZE = 256  # Reduced for faster iterations on limited hardware
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    KL_ANNEALING_EPOCHS = 20  # Epochs to anneal KL term from 0 to 1
    GRAD_CLIP = 1.0  # Gradient clipping
    
    # Performance optimizations
    USE_AMP = True  # Mixed precision training (FP16) - 50-100% speedup
    NUM_WORKERS = 0  # DataLoader workers (0 for Windows, 4+ for Linux)
    VAL_FREQUENCY = 5  # Validate every N epochs (1=every epoch, 5=every 5 epochs)
    
    # ODE solver parameters
    ODE_SOLVER = 'rk4'  # 'dopri5', 'rk4', 'euler' (rk4 is faster)
    N_ODE_STEPS = 3  # Integration steps for RK4 (3 for speed, 5 for accuracy)
    ODE_RTOL = 1e-3
    ODE_ATOL = 1e-4
    
    # Split strategies
    SPLIT_TYPE = 'random'  # 'random', 'temporal', 'station'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    TEMPORAL_CUTOFF = 0.7  # For temporal split: fraction of time for training
    STATION_HOLDOUT = 30  # Number of stations to hold out for testing
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    SEED = 42
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save model every N epochs
    CHECKPOINT_DIR = 'checkpoints'
    RESULTS_DIR = 'results'
    
    # Evaluation
    EVAL_K_NEIGHBORS = 30  # Use more neighbors for evaluation
    EVAL_BATCH_SIZE = 256
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_')]
        return f"Config({', '.join(attrs)})"
