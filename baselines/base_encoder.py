"""
Abstract base class for all baseline encoders.

Encoder Interface Specification (v1.0)
======================================

All baseline encoders must inherit from ``BaseEncoder`` and follow the
unified input/output contract described below.  This ensures that any
encoder can be swapped into the evaluation pipeline (clustering,
retrieval, etc.) without modification.

1. Configuration (``__init__``)
-------------------------------
Each encoder is initialised with a single ``config: dict``.
Two keys are **required** across all encoders:

    config['input_dim']   — int, dimensionality of a single input sample
                            (e.g. 784 for flattened 28x28 MNIST).
    config['output_dim']  — int, dimensionality of the output code
                            (both pre_code and code must have this width).

All other keys are encoder-specific (learning rate, number of neurons,
spiking parameters, etc.) and should have sensible defaults.

2. Training (``fit``)
---------------------
::

    encoder.fit(train_data, train_labels=None)

    Parameters
    ----------
    train_data   : np.ndarray, shape (n_samples, input_dim), dtype float32
                   Training samples.  Values should be in [0, 1] (if the
                   raw data uses a different range, the encoder normalises
                   internally).
    train_labels : np.ndarray, shape (n_samples,), optional
                   Ground-truth labels.  **Not used for training** (all
                   encoders are unsupervised).  May be used for logging /
                   analysis only.

    Side effects
    ------------
    Sets ``self.is_trained = True`` upon successful completion.

3. Encoding (``encode``)
------------------------
::

    result = encoder.encode(data)

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, input_dim), dtype float32

    Returns
    -------
    result : Dict[str, np.ndarray]
        'pre_code' — np.ndarray, shape (n_samples, output_dim), dtype float32
                     Continuous-valued representation **before** binarisation
                     (e.g. spike counts, raw activations, projections).
        'code'     — np.ndarray, shape (n_samples, output_dim), dtype float32
                     Binary / sparse code **after** binarisation.
                     Values in {0, 1}.

    Notes
    -----
    - ``pre_code`` and ``code`` must have the **same** shape.
    - ``code`` is what downstream tasks (clustering, hashing retrieval)
      consume; ``pre_code`` is kept for analysis and alternative metrics.

4. Persistence (``save`` / ``load``)
------------------------------------
::

    encoder.save(path)   # path: str, e.g. 'outputs/model.pkl'
    encoder.load(path)

Encoders that keep framework-specific state (e.g. PyTorch weights)
should override ``save`` / ``load`` and store additional files alongside
the base pickle.

5. Minimal Example
------------------
::

    import numpy as np
    from baselines.flyhash.encoder import FlyHashEncoder

    config = {'input_dim': 784, 'output_dim': 256}
    encoder = FlyHashEncoder(config)

    train_data = np.random.rand(1000, 784).astype(np.float32)
    encoder.fit(train_data)

    test_data = np.random.rand(200, 784).astype(np.float32)
    result = encoder.encode(test_data)

    assert result['pre_code'].shape == (200, 256)
    assert result['code'].shape     == (200, 256)
    assert set(np.unique(result['code'])).issubset({0.0, 1.0})
"""

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Optional


class BaseEncoder(ABC):
    """
    Abstract base class for all baseline encoders.

    Ensures a consistent API across different methods so that any encoder
    can be used interchangeably in the evaluation pipeline.  See the
    module-level docstring for the full interface specification.

    Subclasses **must** implement ``fit()`` and ``encode()``.
    Subclasses **may** override ``save()`` / ``load()`` to persist
    framework-specific state (e.g. PyTorch model weights).
    """

    def __init__(self, config: dict):
        """
        Initialize encoder with configuration.

        Args:
            config: Dictionary with encoder-specific parameters.
                    Required keys: ``input_dim`` (int), ``output_dim`` (int).
                    All other keys are encoder-specific.
        """
        self.config = config
        self.is_trained = False
        self.name = self.__class__.__name__
    
    @abstractmethod
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train the encoder (unsupervised).

        Args:
            train_data:   np.ndarray, shape (n_samples, input_dim), dtype float32.
            train_labels: np.ndarray, shape (n_samples,), optional.
                          Ground-truth labels — **not used for training**.
                          May be used for logging / analysis only.

        Side effects:
            Sets ``self.is_trained = True`` upon successful completion.
        """
        pass

    @abstractmethod
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data into representations.

        Args:
            data: np.ndarray, shape (n_samples, input_dim), dtype float32.

        Returns:
            Dict[str, np.ndarray] with exactly two keys:
                'pre_code': shape (n_samples, output_dim), float32.
                            Continuous representation before binarisation.
                'code':     shape (n_samples, output_dim), float32.
                            Binary code after binarisation, values in {0, 1}.

        Raises:
            RuntimeError: If the encoder has not been fitted yet.
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
