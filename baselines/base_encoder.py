"""
Abstract base class for all baseline encoders.
"""

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Optional


class BaseEncoder(ABC):
    """
    Abstract base class for all baseline encoders.
    Ensures consistent API across different methods.
    """
    
    def __init__(self, config: dict):
        """
        Initialize encoder with configuration.
        
        Args:
            config: Dictionary with encoder-specific parameters
        """
        self.config = config
        self.is_trained = False
        self.name = self.__class__.__name__
    
    @abstractmethod
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train the encoder (unsupervised, labels only for analysis).
        
        Args:
            train_data: (n_samples, input_dim) numpy array
            train_labels: (n_samples,) optional, for analysis only (not used in training)
        """
        pass
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data into representations.
        
        Args:
            data: (n_samples, input_dim) numpy array
        
        Returns:
            dict with keys:
                - 'pre_code': (n_samples, code_dim) continuous representation before binarization
                - 'code': (n_samples, code_dim) sparse/binary code after binarization
        """
        pass
    
    def save(self, path: str):
        """
        Save trained model to disk.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': self.config,
            'is_trained': self.is_trained,
            'name': self.name,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load trained model from disk.
        
        Args:
            path: Path to saved file
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.is_trained = state['is_trained']
        self.name = state['name']
        
        print(f"Model loaded from {path}")
    
    def get_info(self) -> Dict[str, any]:
        """
        Get encoder information.
        
        Returns:
            Dictionary with encoder info
        """
        return {
            'name': self.name,
            'config': self.config,
            'is_trained': self.is_trained,
        }
    
    def __repr__(self) -> str:
        return f"{self.name}(trained={self.is_trained})"


class DummyEncoder(BaseEncoder):
    """
    Dummy encoder for testing (random projection + binarization).
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.input_dim = config.get('input_dim', 784)
        self.code_dim = config.get('code_dim', 128)
        self.projection_matrix = None
    
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """Initialize random projection matrix."""
        self.projection_matrix = np.random.randn(self.input_dim, self.code_dim)
        self.is_trained = True
        print(f"Dummy encoder fitted: {self.input_dim} -> {self.code_dim}")
    
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Random projection + sign binarization."""
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")
        
        # Project
        pre_code = np.dot(data, self.projection_matrix)
        
        # Binarize (sign)
        code = (pre_code > 0).astype(np.float32)
        
        return {
            'pre_code': pre_code,
            'code': code
        }


if __name__ == '__main__':
    # Test base encoder with dummy implementation
    print("Testing DummyEncoder...")
    
    # Generate synthetic data
    np.random.seed(0)
    train_data = np.random.randn(100, 784)
    test_data = np.random.randn(20, 784)
    
    # Create encoder
    config = {'input_dim': 784, 'code_dim': 128}
    encoder = DummyEncoder(config)
    
    print(f"Encoder: {encoder}")
    
    # Fit
    encoder.fit(train_data)
    print(f"After fitting: {encoder}")
    
    # Encode
    result = encoder.encode(test_data)
    print(f"Pre-code shape: {result['pre_code'].shape}")
    print(f"Code shape: {result['code'].shape}")
    print(f"Code sparsity: {1 - np.mean(result['code']):.3f}")
    
    # Save/load
    encoder.save('/tmp/test_encoder.pkl')
    encoder2 = DummyEncoder(config)
    encoder2.load('/tmp/test_encoder.pkl')
    print(f"Loaded encoder: {encoder2}")
