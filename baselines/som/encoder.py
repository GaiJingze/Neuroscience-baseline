"""
Self-Organizing Map (SOM / Kohonen Network) encoder implementation.

Based on: Kohonen, T. "Self-organized formation of topologically correct
feature maps." Biological Cybernetics, 43(1), 59-69, 1982.

Key bio-inspired principles:
1. Competitive learning: neurons compete, only the winner (BMU) gets updated
2. Lateral interaction: neighborhood function ensures topological preservation
3. Self-organization: the map organizes to reflect the statistical structure of input
4. Adaptation: learning rate and neighborhood radius decay over time

The SOM produces a topologically ordered mapping of input space to a 2D grid
of neurons. For hashing, we encode inputs as sparse activations over the grid.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder


class SOMEncoder(BaseEncoder):
    """
    Self-Organizing Map encoder for clustering/hashing.

    The map is a 2D grid of neurons. Each neuron has a weight vector.
    Training uses competitive learning with neighborhood function.
    Encoding produces distance-to-neuron features, binarized via top-k.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']
        self.map_height = config.get('map_height', 20)
        self.map_width = config.get('map_width', 20)
        self.n_neurons = self.map_height * self.map_width
        self.n_epochs = config.get('n_epochs', 10)
        self.lr_init = config.get('lr_init', 0.5)
        self.lr_final = config.get('lr_final', 0.01)
        self.sigma_init = config.get('sigma_init', None)
        self.sigma_final = config.get('sigma_final', 1.0)
        self.batch_size = config.get('batch_size', 1)  # SOM typically uses batch_size=1
        self.k_active = config.get('k_active', max(1, int(self.n_neurons * 0.05)))
        self.verbose = config.get('verbose', True)

        self.weights = None  # [n_neurons, input_dim]
        self.positions = None  # [n_neurons, 2] grid positions

    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train SOM using competitive learning with neighborhood function.
        """
        n_samples = len(train_data)

        if self.verbose:
            print(f"\n{'='*60}")
            print("Training Self-Organizing Map")
            print(f"{'='*60}")
            print(f"Input dim: {self.input_dim}")
            print(f"Map size: {self.map_height} x {self.map_width} = {self.n_neurons} neurons")
            print(f"Epochs: {self.n_epochs}")
            print(f"LR: {self.lr_init} -> {self.lr_final}")
            print(f"{'='*60}\n")

        # Initialize neuron grid positions
        self.positions = np.array(
            [[i, j] for i in range(self.map_height) for j in range(self.map_width)],
            dtype=np.float32
        )

        # Precompute pairwise grid distances
        pos_diff = self.positions[:, None, :] - self.positions[None, :, :]  # [N, N, 2]
        self._grid_dist_sq = np.sum(pos_diff ** 2, axis=2)  # [N, N]

        # Initialize weights from random data samples + small noise
        init_indices = np.random.choice(n_samples, self.n_neurons, replace=True)
        self.weights = train_data[init_indices].copy().astype(np.float32)
        self.weights += np.random.randn(*self.weights.shape).astype(np.float32) * 0.01

        if self.sigma_init is None:
            self.sigma_init = max(self.map_height, self.map_width) / 2.0

        # Training loop
        total_steps = self.n_epochs * n_samples
        step = 0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch = train_data[batch_idx]

                progress = step / total_steps

                # Exponential decay for learning rate and sigma
                lr = self.lr_init * (self.lr_final / self.lr_init) ** progress
                sigma = self.sigma_init * (self.sigma_final / self.sigma_init) ** progress
                sigma_sq = sigma ** 2

                for x in batch:
                    # Find Best Matching Unit (BMU)
                    diffs = self.weights - x  # [n_neurons, input_dim]
                    dists = np.sum(diffs ** 2, axis=1)  # [n_neurons]
                    bmu_idx = np.argmin(dists)

                    # Compute neighborhood function (Gaussian)
                    h = np.exp(-self._grid_dist_sq[bmu_idx] / (2 * sigma_sq))  # [n_neurons]

                    # Update weights
                    self.weights += lr * h[:, np.newaxis] * (x - self.weights)

                    step += 1

            if self.verbose:
                print(f"  Epoch {epoch+1}/{self.n_epochs} (lr={lr:.4f}, sigma={sigma:.2f})")

        self.is_trained = True
        if self.verbose:
            print(f"\n✅ SOM training complete!\n")

    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data using trained SOM.

        Each input is mapped to negative distances to all neurons.
        Binary code: top-k nearest neurons are active (1).
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")

        if self.verbose:
            print(f"Encoding {len(data)} samples with SOM...")

        n_samples = len(data)
        pre_code = np.zeros((n_samples, self.n_neurons), dtype=np.float32)

        # Process in batches for memory efficiency
        batch_size = 512
        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            # Compute distances to all neurons
            # [batch, 1, input_dim] - [1, n_neurons, input_dim]
            diffs = batch[:, np.newaxis, :] - self.weights[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)  # [batch, n_neurons]
            pre_code[i:i + len(batch)] = -dists  # Negative distance = similarity

        # Top-k binarization
        code = self._top_k_binarization(pre_code, self.k_active)

        if self.verbose:
            print("✅ SOM encoding complete!")

        return {'pre_code': pre_code, 'code': code}

    def _top_k_binarization(self, features: np.ndarray, k: int) -> np.ndarray:
        """Top-k binarization."""
        binary_codes = np.zeros_like(features)
        top_k_indices = np.argsort(features, axis=1)[:, -k:]
        rows = np.arange(len(features))[:, None]
        binary_codes[rows, top_k_indices] = 1
        return binary_codes

    def save(self, path: str):
        """Save model."""
        super().save(path)
        if self.weights is not None:
            np.save(path.replace('.pkl', '_weights.npy'), self.weights)

    def load(self, path: str):
        """Load model."""
        super().load(path)
        w_path = path.replace('.pkl', '_weights.npy')
        if Path(w_path).exists():
            self.weights = np.load(w_path)
            # Rebuild grid
            self.positions = np.array(
                [[i, j] for i in range(self.map_height) for j in range(self.map_width)],
                dtype=np.float32
            )
