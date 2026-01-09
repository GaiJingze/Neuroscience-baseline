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
        
        # Load training and test data
        train_dataset = datasets.MNIST(
            root=root, 
            train=True, 
            download=download,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.MNIST(
            root=root, 
            train=False, 
            download=download,
            transform=transforms.ToTensor()
        )
        
        # Convert to numpy arrays
        self.train_data = train_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.train_labels = train_dataset.targets.numpy()
        self.test_data = test_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.test_labels = test_dataset.targets.numpy()
        
        print(f"MNIST loaded: {len(self.train_data)} train, {len(self.test_data)} test")
    
    def get_splits(self) -> Dict[str, np.ndarray]:
        """Return train/test splits."""
        return {
            'train_data': self.train_data,
            'train_labels': self.train_labels,
            'test_data': self.test_data,
            'test_labels': self.test_labels,
        }


class FashionMNISTDataset:
    """Fashion-MNIST dataset loader."""
    
    def __init__(self, root: str = './data', download: bool = True):
        self.root = root
        
        # Load training and test data
        train_dataset = datasets.FashionMNIST(
            root=root, 
            train=True, 
            download=download,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.FashionMNIST(
            root=root, 
            train=False, 
            download=download,
            transform=transforms.ToTensor()
        )
        
        # Convert to numpy arrays
        self.train_data = train_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.train_labels = train_dataset.targets.numpy()
        self.test_data = test_dataset.data.numpy().reshape(-1, 784) / 255.0
        self.test_labels = test_dataset.targets.numpy()
        
        print(f"Fashion-MNIST loaded: {len(self.train_data)} train, {len(self.test_data)} test")
    
    def get_splits(self) -> Dict[str, np.ndarray]:
        """Return train/test splits."""
        return {
            'train_data': self.train_data,
            'train_labels': self.train_labels,
            'test_data': self.test_data,
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
        
        # Load query vectors
        query_file = self.root / 'sift_query.fvecs'
        self.query_data = self._read_fvecs(query_file) if query_file.exists() else None
        
        # Load ground truth (if available)
        gt_file = self.root / 'sift_groundtruth.ivecs'
        self.groundtruth = self._read_ivecs(gt_file) if gt_file.exists() else None
        
        print(f"SIFT1M loaded: {len(self.base_data)} base vectors, "
              f"{len(self.query_data) if self.query_data is not None else 0} queries")
    
    @staticmethod
    def _read_fvecs(filepath: Path, max_vectors: Optional[int] = None) -> np.ndarray:
        """Read .fvecs file format (SIFT features)."""
        vectors = []
        with open(filepath, 'rb') as f:
            while True:
                # Read dimension
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                
                # Read vector
                vector = struct.unpack('f' * dim, f.read(4 * dim))
                vectors.append(vector)
                
                if max_vectors and len(vectors) >= max_vectors:
                    break
        
        return np.array(vectors, dtype=np.float32)
    
    @staticmethod
    def _read_ivecs(filepath: Path, max_vectors: Optional[int] = None) -> np.ndarray:
        """Read .ivecs file format (ground truth indices)."""
        vectors = []
        with open(filepath, 'rb') as f:
            while True:
                # Read dimension
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                
                # Read vector
                vector = struct.unpack('i' * dim, f.read(4 * dim))
                vectors.append(vector)
                
                if max_vectors and len(vectors) >= max_vectors:
                    break
        
        return np.array(vectors, dtype=np.int32)
    
    def get_splits(self) -> Dict[str, np.ndarray]:
        """Return base/query/groundtruth splits."""
        return {
            'base_data': self.base_data,
            'query_data': self.query_data,
            'groundtruth': self.groundtruth,
        }


class GloVeDataset:
    """GloVe word embeddings dataset loader."""
    
    def __init__(self, root: str = './data/glove', dim: int = 100, max_words: Optional[int] = None):
        """
        Args:
            root: Path to GloVe directory
            dim: Embedding dimension (50, 100, 200, or 300)
            max_words: Maximum number of words to load (for memory efficiency)
        """
        self.root = Path(root)
        self.dim = dim
        
        # Load GloVe file
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
    
    def get_splits(self, train_ratio: float = 0.8) -> Dict[str, np.ndarray]:
        """Split into train/test (no labels, unsupervised task)."""
        n_train = int(len(self.vectors) * train_ratio)
        
        return {
            'train_data': self.vectors[:n_train],
            'train_labels': None,  # No labels for unsupervised task
            'test_data': self.vectors[n_train:],
            'test_labels': None,
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
    elif name == 'fashion_mnist' or name == 'fashion-mnist':
        dataset = FashionMNISTDataset(root=root, **kwargs)
    elif name == 'sift1m':
        dataset = SIFT1MDataset(root=root, **kwargs)
    elif name == 'glove':
        dataset = GloVeDataset(root=root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset.get_splits()


if __name__ == '__main__':
    # Test dataset loading
    print("Testing MNIST...")
    mnist = load_dataset('mnist')
    print(f"Train shape: {mnist['train_data'].shape}")
    print(f"Test shape: {mnist['test_data'].shape}")
    
    print("\nTesting Fashion-MNIST...")
    fashion = load_dataset('fashion_mnist')
    print(f"Train shape: {fashion['train_data'].shape}")
    print(f"Test shape: {fashion['test_data'].shape}")
