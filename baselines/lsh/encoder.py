"""
LSH/SimHash encoder implementation.

SimHash (Locality-Sensitive Hashing using random hyperplanes).
Based on: Charikar, M. "Similarity estimation techniques from rounding algorithms."
STOC 2002.

This is a foundational hashing method that preserves cosine similarity.
While not directly bio-inspired in origin, it shares key principles with
neural population coding:
1. Random projections (analogous to random synaptic connections)
2. Sign/threshold activation (analogous to neuronal firing threshold)
3. Distributed binary representation (analogous to population codes)

Serves as an important non-learning baseline for comparison.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder


class SimHashEncoder(BaseEncoder):
    """
    SimHash (LSH) encoder using random hyperplane projections.

    hash(x) = sign(R @ x)

    where R is a random Gaussian matrix. This preserves cosine similarity:
    Pr[h(x) = h(y)] = 1 - angle(x, y) / π

    Non-parametric: no training required.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']
        self.hash_dim = config.get('hash_dim', 128)
        self.verbose = config.get('verbose', True)

        self.random_vectors = None  # [input_dim, hash_dim]

    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Initialize random hyperplanes (non-parametric, no training).
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Initializing SimHash (LSH) Encoder")
            print(f"{'='*60}")
            print(f"Input dim: {self.input_dim}")
            print(f"Hash dim: {self.hash_dim}")
            print(f"{'='*60}\n")

        # Random Gaussian projection matrix
        self.random_vectors = np.random.randn(
            self.input_dim, self.hash_dim
        ).astype(np.float32)

        # L2 normalize each hyperplane
        norms = np.linalg.norm(self.random_vectors, axis=0, keepdims=True)
        norms[norms == 0] = 1
        self.random_vectors /= norms

        self.is_trained = True
        if self.verbose:
            print("✅ SimHash initialization complete!\n")

    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data using SimHash: sign(data @ R).
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")

        # Random projection
        pre_code = data @ self.random_vectors  # [n_samples, hash_dim]

        # Sign binarization (SimHash core)
        code = (pre_code > 0).astype(np.float32)

        return {'pre_code': pre_code, 'code': code}

    def save(self, path: str):
        """Save model."""
        super().save(path)
        if self.random_vectors is not None:
            np.save(path.replace('.pkl', '_vectors.npy'), self.random_vectors)

    def load(self, path: str):
        """Load model."""
        super().load(path)
        v_path = path.replace('.pkl', '_vectors.npy')
        if Path(v_path).exists():
            self.random_vectors = np.load(v_path)
