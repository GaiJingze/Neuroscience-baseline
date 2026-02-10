"""
Dataset loading and preprocessing utilities.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict, Optional, Tuple
import struct


class MNISTDataset:
    """MNIST dataset loader."""
    
    def __init__(self, root: str = './data', download: bool = True):
        self.root = root
        
        train_dataset = datasets.MNIST(
            root=root, train=True, download=download,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.MNIST(
            root=root, train=False, download=download,
            transform=transforms.ToTensor()
        )
        
        self.train_data = train_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.train_labels = train_dataset.targets.numpy()
        self.test_data = test_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.test_labels = test_dataset.targets.numpy()
        
        print(f"MNIST loaded: {len(self.train_data)} train, {len(self.test_data)} test")
    
    def get_splits(self) -> Dict[str, np.ndarray]:
        """Return train/test splits with labels for clustering + retrieval."""
        return {
            'train_data': self.train_data.astype(np.float32),
            'train_labels': self.train_labels,
            'test_data': self.test_data.astype(np.float32),
            'test_labels': self.test_labels,
            # query_data and groundtruth are built from labels in run_baseline.py
        }


class FashionMNISTDataset:
    """Fashion-MNIST dataset loader."""
    
    def __init__(self, root: str = './data', download: bool = True):
        self.root = root
        
        train_dataset = datasets.FashionMNIST(
            root=root, train=True, download=download,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.FashionMNIST(
            root=root, train=False, download=download,
            transform=transforms.ToTensor()
        )
        
        self.train_data = train_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.train_labels = train_dataset.targets.numpy()
        self.test_data = test_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.test_labels = test_dataset.targets.numpy()
        
        print(f"Fashion-MNIST loaded: {len(self.train_data)} train, {len(self.test_data)} test")
    
    def get_splits(self) -> Dict[str, np.ndarray]:
        """Return train/test splits."""
        return {
            'train_data': self.train_data.astype(np.float32),
            'train_labels': self.train_labels,
            'test_data': self.test_data.astype(np.float32),
            'test_labels': self.test_labels,
        }


class SIFT1MDataset:
    """SIFT1M dataset loader (for ANN/retrieval evaluation)."""
    
    def __init__(self, root: str = './data/sift1m', subset_size: Optional[int] = None):
        """
        Args:
            root: Path to SIFT1M directory containing .fvecs and .ivecs files
            subset_size: If provided, use only a subset of the base vectors
        """
        self.root = Path(root)
        
        # Load base vectors (database)
        base_file = self.root / 'sift_base.fvecs'
        if not base_file.exists():
            raise FileNotFoundError(
                f"SIFT base file not found at {base_file}. "
                "Please run scripts/download_sift1m.sh first."
            )
        
        self.base_data = self._read_fvecs(base_file, subset_size)

        # Load learn vectors (for training, if available)
        learn_file = self.root / 'sift_learn.fvecs'
        self.learn_data = self._read_fvecs(learn_file) if learn_file.exists() else None
        
        # Load query vectors
        query_file = self.root / 'sift_query.fvecs'
        self.query_data = self._read_fvecs(query_file) if query_file.exists() else None
        
        # Load ground truth
        gt_file = self.root / 'sift_groundtruth.ivecs'
        self.groundtruth = self._read_ivecs(gt_file) if gt_file.exists() else None
        
        # Adjust groundtruth indices if using subset
        if subset_size is not None and self.groundtruth is not None:
            # Filter groundtruth to only include indices within subset
            valid_mask = self.groundtruth < subset_size
            # For each query, keep only valid neighbors
            filtered_gt = []
            for i in range(len(self.groundtruth)):
                valid_idx = self.groundtruth[i][valid_mask[i]]
                if len(valid_idx) == 0:
                    valid_idx = np.array([0])  # fallback
                filtered_gt.append(valid_idx)
            # Pad to uniform length
            max_len = max(len(g) for g in filtered_gt)
            self.groundtruth = np.zeros((len(filtered_gt), max_len), dtype=np.int32)
            for i, g in enumerate(filtered_gt):
                self.groundtruth[i, :len(g)] = g

        print(f"SIFT1M loaded: {len(self.base_data)} base vectors, "
              f"{len(self.query_data) if self.query_data is not None else 0} queries")
    
    @staticmethod
    def _read_fvecs(filepath: Path, max_vectors: Optional[int] = None) -> np.ndarray:
        """Read .fvecs file format."""
        vectors = []
        with open(filepath, 'rb') as f:
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                vector = struct.unpack('f' * dim, f.read(4 * dim))
                vectors.append(vector)
                if max_vectors and len(vectors) >= max_vectors:
                    break
        return np.array(vectors, dtype=np.float32)
    
    @staticmethod
    def _read_ivecs(filepath: Path, max_vectors: Optional[int] = None) -> np.ndarray:
        """Read .ivecs file format."""
        vectors = []
        with open(filepath, 'rb') as f:
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                vector = struct.unpack('i' * dim, f.read(4 * dim))
                vectors.append(vector)
                if max_vectors and len(vectors) >= max_vectors:
                    break
        return np.array(vectors, dtype=np.int32)
    
    def get_splits(self) -> Dict[str, np.ndarray]:
        """Return base/query/groundtruth splits with unified keys."""
        # Use learn_data for training if available, otherwise use base_data
        train_data = self.learn_data if self.learn_data is not None else self.base_data

        return {
            'train_data': train_data,          # For encoder training
            'train_labels': None,              # No labels
            'test_data': self.base_data,       # Database for retrieval
            'test_labels': None,               # No labels
            'query_data': self.query_data,     # Retrieval queries
            'groundtruth': self.groundtruth,   # Ground truth neighbors
        }


class GloVeDataset:
    """GloVe word embeddings dataset loader."""
    
    def __init__(self, root: str = './data/glove', dim: int = 100,
                 max_words: Optional[int] = None,
                 n_queries: int = 1000, k_groundtruth: int = 100):
        """
        Args:
            root: Path to GloVe directory
            dim: Embedding dimension (50, 100, 200, or 300)
            max_words: Maximum number of words to load
            n_queries: Number of query vectors to hold out
            k_groundtruth: Number of ground truth neighbors per query
        """
        self.root = Path(root)
        self.dim = dim
        self.n_queries = n_queries
        self.k_groundtruth = k_groundtruth
        
        glove_file = self.root / f'glove.6B.{dim}d.txt'
        if not glove_file.exists():
            raise FileNotFoundError(
                f"GloVe file not found at {glove_file}. "
                "Please run scripts/download_glove.sh first."
            )
        
        self.words = []
        self.vectors = []
        
        with open(glove_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_words and i >= max_words:
                    break
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self.words.append(word)
                self.vectors.append(vector)
        
        self.vectors = np.array(self.vectors)
        print(f"GloVe loaded: {len(self.words)} words, dim={dim}")
    
    def _compute_groundtruth(self, queries: np.ndarray, database: np.ndarray,
                             k: int) -> np.ndarray:
        """
        Compute ground truth nearest neighbors using cosine similarity.
        """
        print(f"  Computing ground truth ({len(queries)} queries, {len(database)} database, k={k})...")

        # Normalize for cosine similarity
        q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        db_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)

        groundtruth = np.zeros((len(queries), k), dtype=np.int32)

        batch_size = 100
        for i in range(0, len(queries), batch_size):
            batch_q = q_norm[i:i + batch_size]
            sims = batch_q @ db_norm.T  # [batch, n_database]
            topk = np.argsort(-sims, axis=1)[:, :k]
            groundtruth[i:i + len(batch_q)] = topk

        print(f"  Ground truth computed.")
        return groundtruth

    def get_splits(self, train_ratio: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Split into train/test/query with ground truth for retrieval.

        - Last n_queries vectors are used as queries
        - Remaining vectors form the database (test_data)
        - First train_ratio of database is used for training
        """
        n_total = len(self.vectors)
        n_queries = min(self.n_queries, int(n_total * 0.1))

        # Split: database | queries
        database = self.vectors[:-n_queries]
        query_data = self.vectors[-n_queries:]

        n_train = int(len(database) * train_ratio)
        train_data = database[:n_train]

        # Compute ground truth neighbors (cosine similarity in original space)
        groundtruth = self._compute_groundtruth(query_data, database, self.k_groundtruth)
        
        return {
            'train_data': train_data.astype(np.float32),
            'train_labels': None,
            'test_data': database.astype(np.float32),     # Database for retrieval
            'test_labels': None,
            'query_data': query_data.astype(np.float32),   # Queries
            'groundtruth': groundtruth,
        }


def load_dataset(name: str, root: str = './data', **kwargs) -> Dict[str, np.ndarray]:
    """
    Unified dataset loading function.
    
    Args:
        name: Dataset name ('mnist', 'fashion_mnist', 'sift1m', 'glove')
        root: Root directory for data
        **kwargs: Additional arguments passed to dataset constructor
    
    Returns:
        Dictionary with dataset splits
    """
    name = name.lower()
    
    if name == 'mnist':
        dataset = MNISTDataset(root=root, **kwargs)
    elif name in ('fashion_mnist', 'fashion-mnist'):
        dataset = FashionMNISTDataset(root=root, **kwargs)
    elif name == 'sift1m':
        dataset = SIFT1MDataset(root=root, **kwargs)
    elif name == 'glove':
        dataset = GloVeDataset(root=root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset.get_splits()


if __name__ == '__main__':
    print("Testing MNIST...")
    mnist = load_dataset('mnist')
    print(f"Train shape: {mnist['train_data'].shape}")
    print(f"Test shape: {mnist['test_data'].shape}")
    print(f"Has labels: {mnist.get('test_labels') is not None}")
    
    print("\nTesting Fashion-MNIST...")
    fashion = load_dataset('fashion_mnist')
    print(f"Train shape: {fashion['train_data'].shape}")
    print(f"Test shape: {fashion['test_data'].shape}")
