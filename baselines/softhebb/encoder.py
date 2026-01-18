"""
SoftHebb encoder implementation.

Based on: "SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks"
Kozachkov et al., Neural Computation and Engineering (2022), ICLR (2023)

Paper: https://iopscience.iop.org/article/10.1088/2634-4386/ac98a9
Official repo: https://github.com/NeuromorphicComputing/SoftHebb
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder


class SoftHebbLayer(nn.Module):
    """
    A single SoftHebb layer implementing probabilistic Hebbian learning.
    
    Key components:
    - Soft Winner-Take-All (soft-WTA) activation
    - Hebbian weight updates
    - Lateral inhibition through soft competition
    """
    
    def __init__(self, input_dim: int, output_dim: int, k: int, 
                 beta: float = 5.0, eta: float = 0.01):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Number of output neurons
            k: Top-k for soft WTA (sparsity parameter)
            beta: Temperature parameter for soft-WTA (higher = more sparse)
            eta: Learning rate for Hebbian updates
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.beta = beta
        self.eta = eta
        
        # Weight matrix: forward connections (normalized initialization)
        W_init = torch.randn(output_dim, input_dim)
        W_init = F.normalize(W_init, p=2, dim=1)  # Normalize each row
        self.W = nn.Parameter(W_init * 0.1)
        
        # Lateral inhibition matrix (optional, for stronger WTA)
        # In soft-WTA, this is implicit through the competition mechanism
        
    def soft_wta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft Winner-Take-All activation.
        
        Instead of hard top-k selection, uses a differentiable approximation:
        - Compute activations
        - Apply soft competition (e.g., softmax with temperature)
        - Keep top-k winners (soft)
        
        Args:
            x: Input activations [batch_size, output_dim]
            
        Returns:
            Soft-WTA activations [batch_size, output_dim]
        """
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x, k=min(self.k, x.size(1)), dim=1)
        
        # Create mask for top-k
        mask = torch.zeros_like(x)
        mask.scatter_(1, topk_indices, 1.0)
        
        # Apply mask to keep only top-k
        # Use the original activation values (not softmax) to preserve diversity
        output = x * mask
        
        # Optional: Apply temperature scaling to top-k values for soft competition
        # This makes the distribution softer while maintaining diversity
        if self.beta > 1.0:
            # Scale the masked values
            output = output * self.beta
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with soft-WTA.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output activations [batch_size, output_dim]
        """
        # Linear projection
        activations = F.linear(x, self.W)
        
        # Apply ReLU (common in Hebbian networks)
        activations = F.relu(activations)
        
        # Apply soft-WTA
        output = self.soft_wta(activations)
        
        return output
    
    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Hebbian weight update: ΔW = η * y * x^T
        
        This implements the classic Hebbian rule: "neurons that fire together, wire together"
        
        Args:
            x: Input [batch_size, input_dim]
            y: Output [batch_size, output_dim]
        """
        with torch.no_grad():
            # Compute outer product: y^T * x
            # Average over batch
            delta_W = torch.matmul(y.T, x) / x.size(0)
            
            # Update weights
            self.W.data += self.eta * delta_W
            
            # Normalize weights to prevent unbounded growth and maintain stability
            self.W.data = F.normalize(self.W.data, p=2, dim=1)


class SoftHebbNetwork(nn.Module):
    """
    Multi-layer SoftHebb network.
    
    Can be single-layer or multi-layer for hierarchical feature learning.
    """
    
    def __init__(self, layer_dims: list, k_values: list, 
                 beta: float = 5.0, eta: float = 0.01):
        """
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            k_values: Top-k for each layer
            beta: Temperature for soft-WTA
            eta: Hebbian learning rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            layer = SoftHebbLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i+1],
                k=k_values[i] if isinstance(k_values, list) else k_values,
                beta=beta,
                eta=eta
            )
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Forward pass through all layers.
        
        Args:
            x: Input [batch_size, input_dim]
            return_all: If True, return all layer activations
            
        Returns:
            Output of final layer (or dict of all layers if return_all=True)
        """
        activations = {'input': x}
        h = x
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            activations[f'layer_{i}'] = h
        
        if return_all:
            return activations
        else:
            return h
    
    def train_step(self, x: torch.Tensor):
        """
        Single training step with Hebbian updates.
        
        Args:
            x: Input batch [batch_size, input_dim]
        """
        # Forward pass
        h = x
        inputs = [x]
        outputs = []
        
        for layer in self.layers:
            h = layer(h)
            outputs.append(h)
            inputs.append(h)
        
        # Backward Hebbian updates (layer by layer)
        for i, layer in enumerate(self.layers):
            layer.hebbian_update(inputs[i], outputs[i])


class SoftHebbEncoder(BaseEncoder):
    """
    SoftHebb encoder for clustering/hashing pipeline.
    
    Implements probabilistic Hebbian learning with soft Winner-Take-All.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.input_dim = config['input_dim']
        self.hidden_dims = config.get('hidden_dims', [1000, 500])  # Multi-layer
        self.output_dim = config.get('output_dim', 400)
        
        # Sparsity parameters
        self.k_values = config.get('k_values', [50, 20])  # Top-k for each layer
        if isinstance(self.k_values, int):
            self.k_values = [self.k_values] * len(self.hidden_dims)
        
        # SoftHebb hyperparameters
        self.beta = config.get('beta', 5.0)  # Temperature for soft-WTA
        self.eta = config.get('eta', 0.01)  # Hebbian learning rate
        
        # Training parameters
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 128)
        
        # Device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build network
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        k_values_full = self.k_values + [int(self.output_dim * 0.05)]  # 5% for final layer
        
        self.network = SoftHebbNetwork(
            layer_dims=layer_dims,
            k_values=k_values_full,
            beta=self.beta,
            eta=self.eta
        ).to(self.device)
        
        self.is_trained = False
    
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train SoftHebb network using Hebbian learning.
        
        Args:
            train_data: Training data [n_samples, input_dim]
            train_labels: Optional labels (not used in unsupervised training)
        """
        print(f"\n{'='*60}")
        print("Training SoftHebb Network")
        print(f"{'='*60}")
        print(f"Architecture: {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.output_dim}")
        print(f"K-values (sparsity): {self.k_values + [int(self.output_dim * 0.05)]}")
        print(f"Beta (temperature): {self.beta}")
        print(f"Eta (learning rate): {self.eta}")
        print(f"Training samples: {len(train_data)}")
        print(f"Epochs: {self.n_epochs}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Convert to torch tensor
        train_data = torch.from_numpy(train_data).float()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(train_data)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Training loop
        self.network.train()
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            n_batches = 0
            
            for batch_idx, (batch_x,) in enumerate(loader):
                batch_x = batch_x.to(self.device)
                
                # Hebbian training step
                self.network.train_step(batch_x)
                
                n_batches += 1
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, Batch {batch_idx+1}/{len(loader)}")
            
            print(f"✅ Epoch {epoch+1}/{self.n_epochs} complete")
        
        self.is_trained = True
        print(f"\n{'='*60}")
        print("✅ SoftHebb Training Complete!")
        print(f"{'='*60}\n")
    
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data using trained SoftHebb network.
        
        Args:
            data: Input data [n_samples, input_dim]
            
        Returns:
            Dictionary with 'pre_code' (continuous) and 'code' (binary/sparse)
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")
        
        self.network.eval()
        
        # Convert to torch
        data_tensor = torch.from_numpy(data).float().to(self.device)
        
        # Encode in batches
        all_outputs = []
        
        with torch.no_grad():
            for i in range(0, len(data_tensor), self.batch_size):
                batch = data_tensor[i:i+self.batch_size]
                output = self.network(batch)
                all_outputs.append(output.cpu())
        
        # Concatenate
        pre_code = torch.cat(all_outputs, dim=0).numpy()
        
        # Binarization: top-k
        k = int(self.output_dim * 0.05)  # 5% sparsity
        code = self._top_k_binarization(pre_code, k)
        
        return {
            'pre_code': pre_code,
            'code': code
        }
    
    def _top_k_binarization(self, features: np.ndarray, k: int) -> np.ndarray:
        """Top-k binarization."""
        binary_codes = np.zeros_like(features)
        top_k_indices = np.argsort(features, axis=1)[:, -k:]
        rows = np.arange(len(features))[:, None]
        binary_codes[rows, top_k_indices] = 1
        return binary_codes
    
    def save(self, path: str):
        """Save trained model."""
        super().save(path)
        
        # Save network weights
        model_path = path.replace('.pkl', '_model.pt')
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim,
                'k_values': self.k_values,
                'beta': self.beta,
                'eta': self.eta,
            }
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load(self, path: str):
        """Load trained model."""
        super().load(path)
        
        # Load network weights
        model_path = path.replace('.pkl', '_model.pt')
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            print(f"Model loaded from {model_path}")


if __name__ == '__main__':
    print("Testing SoftHebb encoder...")
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Test data
    train_data = np.random.rand(1000, 784).astype(np.float32)
    test_data = np.random.rand(100, 784).astype(np.float32)
    
    # Configuration
    config = {
        'input_dim': 784,
        'hidden_dims': [1000, 500],
        'output_dim': 400,
        'k_values': [50, 20],
        'beta': 5.0,
        'eta': 0.01,
        'n_epochs': 3,
        'batch_size': 128,
        'device': 'cpu'
    }
    
    # Create and train encoder
    encoder = SoftHebbEncoder(config)
    encoder.fit(train_data)
    
    # Encode test data
    result = encoder.encode(test_data)
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Pre-code shape: {result['pre_code'].shape}")
    print(f"Code shape: {result['code'].shape}")
    print(f"Code sparsity: {1 - np.mean(result['code']):.3f}")
    print(f"Average ones per sample: {np.mean(np.sum(result['code'], axis=1)):.1f}")
    print(f"{'='*60}")
