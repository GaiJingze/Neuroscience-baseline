"""
FlyHash encoder implementation.
Based on fruit fly olfactory circuit.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder
from typing import Optional, Dict


class FlyHashEncoder(BaseEncoder):
    """
    FlyHash encoder using random projection + Winner-Take-All.
    
    Based on: "A neural algorithm for a fundamental computing problem"
    Dasgupta et al., Science 2017
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.input_dim = config['input_dim']
        self.projection_dim = config.get('projection_dim', self.input_dim * 20)  # Expansion
        self.hash_length = config.get('hash_length', int(self.projection_dim * 0.05))  # 5% sparsity
        self.sampling_ratio = config.get('sampling_ratio', 0.1)  # Connection probability
        
        self.projection_matrix = None
        self.use_package = config.get('use_package', False)  # Use PyPI FlyHash package
    
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Initialize random projection matrix (FlyHash is non-parametric).
        """
        if self.use_package:
            try:
                from flyhash import FlyHash
                self.flyhash = FlyHash(
                    input_dim=self.input_dim,
                    projection_dim=self.projection_dim,
                    hash_length=self.hash_length,
                    sampling_ratio=self.sampling_ratio
                )
                print(f"FlyHash package initialized: {self.input_dim} -> {self.projection_dim} (k={self.hash_length})")
            except ImportError:
                print("Warning: FlyHash package not installed, using manual implementation")
                self.use_package = False
        
        if not self.use_package:
            # Manual implementation: sparse random projection
            self.projection_matrix = self._create_sparse_projection()
            print(f"FlyHash manual init: {self.input_dim} -> {self.projection_dim} (k={self.hash_length})")
        
        self.is_trained = True
    
    def _create_sparse_projection(self) -> np.ndarray:
        """Create sparse binary projection matrix.

        Matches the fruit fly olfactory circuit: each Kenyon cell (KC) connects
        to ~sampling_ratio of projection neurons (PNs) with binary weights (0/1).
        The KC activation is a simple sum of connected PN firing rates.

        Ref: Dasgupta, Stevens & Navlakha, Science 2017, Fig. 1 & Eq. 1.
        """
        # Each output neuron (KC) connects to ~sampling_ratio of input neurons (PNs)
        n_connections_per_neuron = max(1, int(self.input_dim * self.sampling_ratio))

        projection = np.zeros((self.input_dim, self.projection_dim))

        for j in range(self.projection_dim):
            # Randomly select input neurons to connect
            selected_inputs = np.random.choice(
                self.input_dim,
                size=n_connections_per_neuron,
                replace=False
            )
            # Binary weights: connected = 1, disconnected = 0
            projection[selected_inputs, j] = 1.0

        return projection
    
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data using FlyHash.
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")
        
        if self.use_package:
            # Use FlyHash package
            pre_code = self.flyhash.project(data)
            code = self.flyhash.hash(data)
        else:
            # Manual implementation
            # Step 1: Random projection (expansion)
            pre_code = np.dot(data, self.projection_matrix)
            
            # Step 2: Winner-Take-All (top-k)
            code = self._wta_binarization(pre_code, self.hash_length)
        
        return {
            'pre_code': pre_code,
            'code': code
        }
    
    def _wta_binarization(self, features: np.ndarray, k: int) -> np.ndarray:
        """Winner-Take-All: keep top k activations."""
        binary_codes = np.zeros_like(features)
        
        # For each sample, find top k indices
        top_k_indices = np.argsort(features, axis=1)[:, -k:]
        
        # Set top k values to 1
        rows = np.arange(len(features))[:, None]
        binary_codes[rows, top_k_indices] = 1
        
        return binary_codes
    
    def save(self, path: str):
        """Save model (projection matrix)."""
        super().save(path)
        if not self.use_package and self.projection_matrix is not None:
            np.save(path.replace('.pkl', '_projection.npy'), self.projection_matrix)
    
    def load(self, path: str):
        """Load model (projection matrix)."""
        super().load(path)
        if not self.use_package:
            proj_path = path.replace('.pkl', '_projection.npy')
            if Path(proj_path).exists():
                self.projection_matrix = np.load(proj_path)


if __name__ == '__main__':
    # Test FlyHash encoder
    print("Testing FlyHash encoder...")
    
    np.random.seed(0)
    train_data = np.random.rand(100, 128)  # SIFT-like dimension
    test_data = np.random.rand(20, 128)
    
    # Create encoder
    config = {
        'input_dim': 128,
        'projection_dim': 320,  # 2.5x expansion
        'hash_length': 16,  # 5% of 320
        'sampling_ratio': 0.1,
        'use_package': False  # Test manual implementation
    }
    
    encoder = FlyHashEncoder(config)
    encoder.fit(train_data)
    
    # Encode
    result = encoder.encode(test_data)
    print(f"Pre-code shape: {result['pre_code'].shape}")
    print(f"Code shape: {result['code'].shape}")
    print(f"Code ones per sample: {np.sum(result['code'], axis=1)[:5]}")
    print(f"Expected: {config['hash_length']} ones per sample")
