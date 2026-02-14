"""
SoftHebb encoder implementation.

Based on: "SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks"
Moraitis et al., Neural Computation and Engineering (2022), ICLR (2023)

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

    Matches the official algorithm from:
    Moraitis et al., "SoftHebb: Bayesian inference in unsupervised Hebbian
    soft winner-take-all networks", NCE 2022 / ICLR 2023.
    Ref: https://github.com/NeuromorphicComputing/SoftHebb

    Key components:
    - Temperature-scaled softmax as soft-WTA (NOT hard top-k)
    - Self-normalizing Hebbian update: ΔW = η * y * (x - u * w)
    - All neurons participate in competition via softmax probabilities
    """

    def __init__(self, input_dim: int, output_dim: int,
                 t_invert: float = 5.0, eta: float = 0.01):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Number of output neurons
            t_invert: Inverse temperature for softmax (higher = sharper)
            eta: Learning rate for Hebbian updates
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t_invert = t_invert
        self.eta = eta

        # Weight matrix [output_dim, input_dim], L2-normalized rows
        W_init = torch.randn(output_dim, input_dim)
        W_init = F.normalize(W_init, p=2, dim=1)
        self.W = nn.Parameter(W_init)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with temperature-scaled softmax soft-WTA.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            (wta, pre_activation):
                wta: Softmax probabilities [batch_size, output_dim]
                pre_activation: Linear pre-activation u [batch_size, output_dim]
        """
        # Linear projection: u = W @ x
        pre_act = F.linear(x, self.W)  # [batch, output_dim]

        # Soft-WTA via temperature-scaled softmax (Eq. 7 in paper)
        # y_k = softmax(u / τ) = softmax(t_invert * u)
        wta = F.softmax(self.t_invert * pre_act, dim=1)

        return wta, pre_act

    def hebbian_update(self, x: torch.Tensor, wta: torch.Tensor,
                       pre_act: torch.Tensor):
        """
        Self-normalizing Hebbian update (Eq. 8 in paper):
            ΔW_ik = η * y_k * (x_i − u_k * W_ik)

        In matrix form:
            ΔW = η * (y^T @ x − diag(y^T @ u) * W)

        Args:
            x: Input [batch_size, input_dim]
            wta: Softmax probabilities [batch_size, output_dim]
            pre_act: Linear pre-activation [batch_size, output_dim]
        """
        with torch.no_grad():
            batch_size = x.size(0)

            # Hebbian correlation: y^T @ x  →  [output_dim, input_dim]
            yx = torch.matmul(wta.T, x) / batch_size

            # Self-normalizing term: sum_batch(y_k * u_k) per neuron
            yu = torch.sum(wta * pre_act, dim=0) / batch_size  # [output_dim]

            # ΔW = y*x − y*u*W  (Eq. 8)
            delta_W = yx - yu.unsqueeze(1) * self.W

            self.W.data += self.eta * delta_W

            # L2 normalize rows (maintains unit-norm constraint)
            self.W.data = F.normalize(self.W.data, p=2, dim=1)


class SoftHebbNetwork(nn.Module):
    """
    Multi-layer SoftHebb network.

    Each layer: linear projection → softmax soft-WTA → Hebbian update.
    The softmax probabilities are passed as input to the next layer.
    """

    def __init__(self, layer_dims: list,
                 t_invert: float = 5.0, eta: float = 0.01):
        """
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, ..., output_dim]
            t_invert: Inverse temperature for softmax soft-WTA
            eta: Hebbian learning rate
        """
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            layer = SoftHebbLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i+1],
                t_invert=t_invert,
                eta=eta
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through all layers.

        Args:
            x: Input [batch_size, input_dim]

        Returns:
            Softmax probabilities from the final layer [batch_size, output_dim]
        """
        h = x
        for layer in self.layers:
            wta, _ = layer(h)
            h = wta
        return h

    def train_step(self, x: torch.Tensor):
        """
        Single training step with layer-wise Hebbian updates.

        Args:
            x: Input batch [batch_size, input_dim]
        """
        h = x
        for layer in self.layers:
            wta, pre_act = layer(h)
            layer.hebbian_update(h, wta, pre_act)
            h = wta  # Next layer receives softmax output


class SoftHebbEncoder(BaseEncoder):
    """
    SoftHebb encoder for clustering/hashing pipeline.

    Implements probabilistic Hebbian learning with softmax soft-WTA.
    Ref: Moraitis et al., NCE 2022 / ICLR 2023.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']
        self.hidden_dims = config.get('hidden_dims', [1000, 500])  # Multi-layer
        self.output_dim = config.get('output_dim', 400)

        # SoftHebb hyperparameters
        # t_invert = inverse temperature (1/τ); higher = sharper competition
        # Accepts 'beta' for backward-compatibility with existing configs
        self.t_invert = config.get('t_invert', config.get('beta', 5.0))
        self.eta = config.get('eta', 0.01)  # Hebbian learning rate

        # Training parameters
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 128)

        # Device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Build network
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        self.network = SoftHebbNetwork(
            layer_dims=layer_dims,
            t_invert=self.t_invert,
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
        print(f"Inverse temperature (1/τ): {self.t_invert}")
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
                't_invert': self.t_invert,
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
        't_invert': 5.0,
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
