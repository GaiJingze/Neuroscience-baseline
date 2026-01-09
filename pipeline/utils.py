"""
Utility functions for the clustering/hashing pipeline.
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def set_seed(seed: int = 0):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary containing results
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to results file
    
    Returns:
        Dictionary containing results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_spike_sparsity(spike_counts: np.ndarray, n_neurons: int, n_timesteps: int) -> float:
    """
    Compute spike sparsity (1 - firing_rate).
    
    Args:
        spike_counts: Total number of spikes
        n_neurons: Number of neurons in the network
        n_timesteps: Number of simulation timesteps
    
    Returns:
        Sparsity value (0 = dense, 1 = completely sparse)
    """
    max_possible_spikes = n_neurons * n_timesteps
    if isinstance(spike_counts, np.ndarray):
        spike_counts = np.sum(spike_counts)
    sparsity = 1.0 - (spike_counts / max_possible_spikes)
    return float(sparsity)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


class Logger:
    """Simple logger for experiment tracking."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            ensure_dir(os.path.dirname(log_file))
    
    def log(self, message: str):
        """Log a message to console and file."""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
    
    def log_dict(self, data: Dict[str, Any], prefix: str = ""):
        """Log a dictionary in a formatted way."""
        for key, value in data.items():
            self.log(f"{prefix}{key}: {value}")
