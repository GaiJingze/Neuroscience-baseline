"""
WTA Hash (Winner-Take-All Hashing) encoder implementation.

Based on competitive selection principle found in biological neural circuits.
Inspired by: "The Power of Comparative Reasoning" (Yagnik et al., ICCV 2011)

Key bio-inspired principles:
1. Random feature grouping (analogous to random dendritic connections)
2. Local competition within groups (Winner-Take-All, like lateral inhibition)
3. Only the winner in each group produces output (sparse coding)

The hash is constructed by:
- Randomly grouping input features into windows
- Within each window, only the position of the maximum fires (WTA)
- This produces a one-hot code per window, concatenated into a sparse binary code
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder


class WTAHashEncoder(BaseEncoder):
    """
    Winner-Take-All Hash encoder.

    For each of n_hashes random windows of input features,
    the position of the maximum value is set to 1 (one-hot).
    Output dimension = n_hashes * window_size.
    Sparsity = 1 / window_size per window.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']
        self.n_hashes = config.get('n_hashes', 64)          # Number of WTA windows
        self.window_size = config.get('window_size', 8)       # Features per window
        self.output_dim = self.n_hashes * self.window_size
        self.verbose = config.get('verbose', True)

        self.permutations = None  # Random feature selections for each window

    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Initialize random permutations (non-parametric, no training needed).
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Initializing WTA Hash Encoder")
            print(f"{'='*60}")
            print(f"Input dim: {self.input_dim}")
            print(f"Windows: {self.n_hashes}, Window size: {self.window_size}")
            print(f"Output dim: {self.output_dim}")
            print(f"Sparsity: {1 - 1.0/self.window_size:.3f}")
            print(f"{'='*60}\n")

        # Generate random feature selections for each window
        self.permutations = np.zeros((self.n_hashes, self.window_size), dtype=np.int64)
        for h in range(self.n_hashes):
            self.permutations[h] = np.random.choice(
                self.input_dim, size=self.window_size, replace=False
            )

        self.is_trained = True
        if self.verbose:
            print("✅ WTA Hash initialization complete!\n")

    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data using WTA hashing.

        For each window, select the argmax position (one-hot).
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")

        n_samples = len(data)
        pre_code = np.zeros((n_samples, self.output_dim), dtype=np.float32)
        code = np.zeros((n_samples, self.output_dim), dtype=np.float32)

        for h in range(self.n_hashes):
            perm = self.permutations[h]
            windowed = data[:, perm]  # [n_samples, window_size]

            offset = h * self.window_size
            pre_code[:, offset:offset + self.window_size] = windowed

            # Winner-Take-All within window
            max_idx = np.argmax(windowed, axis=1)  # [n_samples]
            code[np.arange(n_samples), offset + max_idx] = 1.0

        return {'pre_code': pre_code, 'code': code}

    def save(self, path: str):
        """Save model."""
        super().save(path)
        if self.permutations is not None:
            np.save(path.replace('.pkl', '_permutations.npy'), self.permutations)

    def load(self, path: str):
        """Load model."""
        super().load(path)
        p_path = path.replace('.pkl', '_permutations.npy')
        if Path(p_path).exists():
            self.permutations = np.load(p_path)
