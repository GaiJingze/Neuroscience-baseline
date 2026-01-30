"""
Krotov-Hopfield Biological Learning Encoder.

Implementation of "Unsupervised Learning by Competing Hidden Units"
by Dmitry Krotov and John Hopfield (PNAS, 2019)

Key features:
- k-Winner-Take-All (k-WTA) competition
- Anti-Hebbian learning (winner strengthened, k-th weakened)
- Lebesgue p-norm weight regularization
- Minibatch gradient descent
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder


class KrotovEncoder(BaseEncoder):
    """
    Krotov-Hopfield encoder using competing hidden units.
    
    This implements unsupervised learning through k-WTA competition
    and anti-Hebbian plasticity. The algorithm:
    
    1. Computes activation: Q = sign(W) * |W|^(p-1) @ X
    2. Selects winners: top-1 gets +1, top-k gets -delta
    3. Updates weights: W += lr * (winner * X - winner * Q * W)
    
    References:
        Krotov & Hopfield (2019). "Unsupervised Learning by Competing Hidden Units"
        PNAS. https://doi.org/10.1073/pnas.1820458116
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Architecture
        self.input_dim = config['input_dim']
        self.n_neurons = config.get('n_neurons', 100)
        
        # Training hyperparameters
        self.n_epochs = config.get('n_epochs', 200)
        self.batch_size = config.get('batch_size', 100)
        self.lr = config.get('lr', 0.02)  # Initial learning rate
        
        # Krotov-specific parameters
        self.delta = config.get('delta', 0.4)  # Anti-Hebbian strength
        self.p = config.get('p', 2.0)          # Lebesgue norm
        self.k = config.get('k', 2)            # Ranking parameter (must be >= 2)
        
        # Initialization parameters
        self.init_mu = config.get('init_mu', 0.0)
        self.init_sigma = config.get('init_sigma', 1.0)
        
        # Numerical precision threshold
        self.prec = config.get('prec', 1e-30)
        
        # Weight matrix (will be initialized in fit)
        self.W = None
        
        # Verbose output
        self.verbose = config.get('verbose', True)
    
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train the Krotov-Hopfield network using competing hidden units.
        
        Args:
            train_data: Training images [n_samples, input_dim]
            train_labels: Optional labels (not used in unsupervised training)
        """
        n_samples = len(train_data)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("Training Krotov-Hopfield Network")
            print(f"{'='*70}")
            print(f"Architecture: {self.input_dim} -> {self.n_neurons} neurons")
            print(f"Training samples: {n_samples}")
            print(f"Epochs: {self.n_epochs}")
            print(f"Batch size: {self.batch_size}")
            print(f"Learning rate: {self.lr} (linearly annealed)")
            print(f"Anti-Hebbian strength (delta): {self.delta}")
            print(f"Lebesgue norm (p): {self.p}")
            print(f"Ranking parameter (k): {self.k}")
            print(f"{'='*70}\n")
        
        # Initialize weights: Gaussian N(mu, sigma)
        self.W = np.random.normal(
            self.init_mu, 
            self.init_sigma, 
            (self.n_neurons, self.input_dim)
        )
        
        # Training loop
        n_batches = n_samples // self.batch_size
        
        for epoch in range(self.n_epochs):
            # Linearly anneal learning rate
            eps = self.lr * (1 - epoch / self.n_epochs)
            
            # Shuffle data at the start of each epoch
            shuffled_indices = np.random.permutation(n_samples)
            train_data_shuffled = train_data[shuffled_indices]
            
            # Process minibatches
            for batch_idx in range(n_batches):
                # Get minibatch
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch = train_data_shuffled[start_idx:end_idx]  # [batch_size, input_dim]
                
                # Transpose for computation: [input_dim, batch_size]
                inputs = batch.T
                
                # Compute activation with power-law weights
                # Q = sign(W) * |W|^(p-1) @ X
                sig = np.sign(self.W)  # [n_neurons, input_dim]
                weights_powered = sig * np.abs(self.W) ** (self.p - 1)
                tot_input = np.dot(weights_powered, inputs)  # [n_neurons, batch_size]
                
                # k-WTA selection: sort activations and select winners
                # y[i, j] = index of i-th ranked neuron for sample j
                y = np.argsort(tot_input, axis=0)  # [n_neurons, batch_size]
                
                # Create winner signals: +1 for winner, -delta for k-th
                yl = np.zeros((self.n_neurons, self.batch_size))
                
                # Winner (highest activation) gets +1
                winner_indices = y[-1, :]  # [batch_size]
                yl[winner_indices, np.arange(self.batch_size)] = 1.0
                
                # k-th ranked gets -delta (anti-Hebbian)
                kth_indices = y[-self.k, :]  # [batch_size]
                yl[kth_indices, np.arange(self.batch_size)] = -self.delta
                
                # Compute weight updates (Eq. 3 from paper)
                # ΔW = η * [yl @ X.T - (yl * Q).sum(1) * W]
                
                # Hebbian term: yl @ X.T
                hebbian_term = np.dot(yl, inputs.T)  # [n_neurons, input_dim]
                
                # Anti-Hebbian term: (yl * Q).sum(1) * W
                xx = np.sum(yl * tot_input, axis=1)  # [n_neurons]
                anti_hebbian_term = np.outer(xx, np.ones(self.input_dim)) * self.W
                
                # Total update
                ds = hebbian_term - anti_hebbian_term
                
                # Normalize update to control step size
                nc = np.max(np.abs(ds))
                if nc < self.prec:
                    nc = self.prec
                
                # Update weights
                self.W += eps * ds / nc
            
            # Progress reporting
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} complete (lr={eps:.4f})")
        
        self.is_trained = True
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("✅ Krotov-Hopfield training complete!")
            print(f"{'='*70}\n")
    
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data using trained Krotov-Hopfield network.
        
        Args:
            data: Input images [n_samples, input_dim]
            
        Returns:
            Dictionary with 'pre_code' (activations) and 'code' (binarized)
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")
        
        if self.verbose:
            print(f"Encoding {len(data)} samples...")
        
        # Compute activations: Q = sign(W) * |W|^(p-1) @ X
        sig = np.sign(self.W)
        weights_powered = sig * np.abs(self.W) ** (self.p - 1)
        
        # Transpose data for matrix multiplication
        inputs = data.T  # [input_dim, n_samples]
        tot_input = np.dot(weights_powered, inputs)  # [n_neurons, n_samples]
        
        # Transpose back to [n_samples, n_neurons]
        pre_code = tot_input.T
        
        # Binarize using top-k selection (k winners per sample)
        # We use k=20 (5% of 400 neurons) for consistency with other baselines
        k_active = max(int(self.n_neurons * 0.05), 1)
        code = self._top_k_binarization(pre_code, k_active)
        
        if self.verbose:
            print("✅ Encoding complete!")
        
        return {
            'pre_code': pre_code,
            'code': code
        }
    
    def _top_k_binarization(self, activations: np.ndarray, k: int) -> np.ndarray:
        """
        Binarize activations by keeping only top-k values per sample.
        
        Args:
            activations: Activation matrix [n_samples, n_neurons]
            k: Number of active neurons per sample
            
        Returns:
            Binary code [n_samples, n_neurons]
        """
        n_samples, n_neurons = activations.shape
        code = np.zeros_like(activations)
        
        for i in range(n_samples):
            # Get indices of top-k activations
            top_k_indices = np.argpartition(activations[i], -k)[-k:]
            code[i, top_k_indices] = 1.0
        
        return code
    
    def save(self, path: str):
        """Save trained model weights."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        
        np.save(path, self.W)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model weights."""
        self.W = np.load(path)
        self.is_trained = True
        
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def __repr__(self):
        return f"KrotovEncoder(neurons={self.n_neurons}, trained={self.is_trained})"
