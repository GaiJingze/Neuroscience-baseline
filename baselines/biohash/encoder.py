"""
BioHash encoder implementation.

Bio-inspired hashing with Hebbian-learned sparse projections and WTA binarization.

Combines key principles from the fruit fly olfactory circuit:
1. Sparse random connectivity (like fly projection neurons → Kenyon cells)
2. Hebbian learning to adapt projections to data statistics
3. Winner-Take-All competition for sparse binary codes

Unlike FlyHash (purely random), BioHash learns data-adaptive projections
through Hebbian plasticity while maintaining biological plausibility.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder


class BioHashEncoder(BaseEncoder):
    """
    BioHash encoder using Hebbian-learned sparse projections + WTA.

    Key features:
    - Sparse random initial connectivity (bio-inspired)
    - Hebbian weight updates: ΔW = η * y * x^T (neurons that fire together wire together)
    - Weight normalization to prevent unbounded growth
    - Winner-Take-All binarization for sparse binary output
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']
        self.hash_dim = config.get('hash_dim', 256)
        self.sparse_ratio = config.get('sparse_ratio', 0.1)  # Connection sparsity
        self.k_winners = config.get('k_winners', max(1, int(self.hash_dim * 0.05)))
        self.n_epochs = config.get('n_epochs', 5)
        self.batch_size = config.get('batch_size', 128)
        self.lr = config.get('lr', 0.01)
        self.verbose = config.get('verbose', True)

        self.W = None  # Projection matrix [input_dim, hash_dim]

    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train BioHash using Hebbian learning on sparse projections.
        """
        n_samples = len(train_data)

        if self.verbose:
            print(f"\n{'='*60}")
            print("Training BioHash Encoder")
            print(f"{'='*60}")
            print(f"Architecture: {self.input_dim} -> {self.hash_dim} (k={self.k_winners})")
            print(f"Sparse ratio: {self.sparse_ratio}")
            print(f"Epochs: {self.n_epochs}, LR: {self.lr}")
            print(f"{'='*60}\n")

        # Initialize sparse projection matrix
        n_connections = int(self.input_dim * self.sparse_ratio)
        self.W = np.zeros((self.input_dim, self.hash_dim), dtype=np.float32)
        self._connection_mask = np.zeros_like(self.W, dtype=bool)

        for j in range(self.hash_dim):
            indices = np.random.choice(self.input_dim, n_connections, replace=False)
            self.W[indices, j] = np.random.randn(n_connections).astype(np.float32)
            self._connection_mask[indices, j] = True

        # L2 normalize columns
        norms = np.linalg.norm(self.W, axis=0, keepdims=True)
        norms[norms == 0] = 1
        self.W /= norms

        # Hebbian learning loop
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            n_batches = n_samples // self.batch_size

            for b in range(n_batches):
                batch = train_data[indices[b * self.batch_size:(b + 1) * self.batch_size]]

                # Forward: project
                activations = batch @ self.W  # [batch, hash_dim]

                # WTA: top-k winners
                topk_idx = np.argsort(activations, axis=1)[:, -self.k_winners:]
                mask = np.zeros_like(activations)
                rows = np.arange(len(batch))[:, None]
                mask[rows, topk_idx] = 1.0
                winners = activations * mask

                # Hebbian update: ΔW = η * x^T @ y (only for active connections)
                delta_W = (batch.T @ winners) / len(batch)
                delta_W[~self._connection_mask] = 0  # Maintain sparsity

                lr = self.lr * (1 - epoch / self.n_epochs)  # Anneal LR
                self.W += lr * delta_W

                # Normalize columns
                norms = np.linalg.norm(self.W, axis=0, keepdims=True)
                norms[norms == 0] = 1
                self.W /= norms

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs} complete (lr={lr:.4f})")

        self.is_trained = True
        if self.verbose:
            print(f"\n✅ BioHash training complete!\n")

    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Encode data using learned sparse projection + WTA."""
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")

        pre_code = data @ self.W
        code = self._wta_binarization(pre_code, self.k_winners)

        return {'pre_code': pre_code, 'code': code}

    def _wta_binarization(self, features: np.ndarray, k: int) -> np.ndarray:
        """Winner-Take-All: keep top k activations."""
        binary_codes = np.zeros_like(features)
        top_k_indices = np.argsort(features, axis=1)[:, -k:]
        rows = np.arange(len(features))[:, None]
        binary_codes[rows, top_k_indices] = 1
        return binary_codes

    def save(self, path: str):
        """Save model."""
        super().save(path)
        if self.W is not None:
            np.save(path.replace('.pkl', '_weights.npy'), self.W)
            np.save(path.replace('.pkl', '_mask.npy'), self._connection_mask)

    def load(self, path: str):
        """Load model."""
        super().load(path)
        w_path = path.replace('.pkl', '_weights.npy')
        m_path = path.replace('.pkl', '_mask.npy')
        if Path(w_path).exists():
            self.W = np.load(w_path)
            self._connection_mask = np.load(m_path)
