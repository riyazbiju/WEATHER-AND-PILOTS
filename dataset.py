"""
PyTorch Dataset for observation-based Latent ODE training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class WeatherObservationDataset(Dataset):
    """
    Dataset that returns individual observations with their context.
    Each sample = (query_observation, context_observations)
    """
    
    def __init__(self, df, context_selector, config, mode='train'):
        """
        Args:
            df: DataFrame with normalized observations
            context_selector: SpacetimeContextSelector instance
            config: Configuration object
            mode: 'train', 'val', or 'test'
        """
        self.df = df
        self.context_selector = context_selector
        self.config = config
        self.mode = mode
        
        # Build index for context selection
        self.context_selector.build_index(df, verbose=(mode == 'train'))
        
        # Store observations DataFrame
        self.observations = df
        
        print(f"{mode.upper()} dataset: {len(df)} observations")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            (query_idx, context_indices) tuple
        """
        # Get context for this query
        context_indices = self.context_selector.get_context(idx)
        
        return idx, context_indices


def create_dataloaders(data_dict, config, split_type='random'):
    """
    Create train/val/test dataloaders with context selection.
    
    Args:
        data_dict: Dictionary with 'train', 'val', 'test' DataFrames
        config: Configuration object
        split_type: 'random', 'temporal', or 'station'
        
    Returns:
        Dictionary with dataloaders and context selectors
    """
    from context_selection import SpacetimeContextSelector, ContextDataCollator
    
    # Determine if we need causality enforcement
    enforce_causality = (split_type == 'temporal')
    
    # Create context selectors for each split
    train_selector = SpacetimeContextSelector(config, enforce_causality=False)
    val_selector = SpacetimeContextSelector(config, enforce_causality=enforce_causality)
    test_selector = SpacetimeContextSelector(config, enforce_causality=enforce_causality)
    
    # Create datasets
    train_dataset = WeatherObservationDataset(
        data_dict['train'], train_selector, config, mode='train'
    )
    val_dataset = WeatherObservationDataset(
        data_dict['val'], val_selector, config, mode='val'
    )
    test_dataset = WeatherObservationDataset(
        data_dict['test'], test_selector, config, mode='test'
    )
    
    # Create collators
    train_collator = ContextDataCollator(config, data_dict['train'])
    val_collator = ContextDataCollator(config, data_dict['val'])
    test_collator = ContextDataCollator(config, data_dict['test'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collator.collate,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=val_collator.collate,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=test_collator.collate,
        num_workers=0,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_selector': train_selector,
        'val_selector': val_selector,
        'test_selector': test_selector
    }
