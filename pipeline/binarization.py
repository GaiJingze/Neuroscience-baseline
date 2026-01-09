"""
Binarization and sparsification methods for neural codes.
"""

import numpy as np
from typing import Optional


def top_k_binarization(features: np.ndarray, k: int) -> np.ndarray:
    """
    Top-k binarization: Keep top k values, set rest to 0.
    
    Args:
        features: (n_samples, n_features) array of continuous values
        k: Number of top features to keep per sample
    
    Returns:
        Binary array of same shape
    """
    binary_codes = np.zeros_like(features)
    
    # For each sample, find top k indices
    top_k_indices = np.argsort(features, axis=1)[:, -k:]
    
    # Set top k values to 1
    rows = np.arange(len(features))[:, None]
    binary_codes[rows, top_k_indices] = 1
    
    return binary_codes


def top_k_percent_binarization(features: np.ndarray, percent: float) -> np.ndarray:
    """
    Top-k% binarization: Keep top k% of features.
    
    Args:
        features: (n_samples, n_features) array of continuous values
        percent: Percentage of features to keep (0-1)
    
    Returns:
        Binary array of same shape
    """
    k = max(1, int(features.shape[1] * percent))
    return top_k_binarization(features, k)


def threshold_binarization(features: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
    Threshold binarization: Values > threshold become 1, rest become 0.
    
    Args:
        features: (n_samples, n_features) array of continuous values
        threshold: Threshold value. If None, use median of each sample
    
    Returns:
        Binary array of same shape
    """
    if threshold is None:
        # Use per-sample median
        threshold = np.median(features, axis=1, keepdims=True)
    
    return (features > threshold).astype(np.float32)


def wta_binarization(features: np.ndarray) -> np.ndarray:
    """
    Winner-Take-All binarization: Only the maximum value becomes 1.
    
    Args:
        features: (n_samples, n_features) array of continuous values
    
    Returns:
        Binary array of same shape
    """
    binary_codes = np.zeros_like(features)
    max_indices = np.argmax(features, axis=1)
    binary_codes[np.arange(len(features)), max_indices] = 1
    
    return binary_codes


def soft_wta(features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Soft Winner-Take-All using softmax.
    
    Args:
        features: (n_samples, n_features) array of continuous values
        temperature: Softmax temperature (lower = sharper)
    
    Returns:
        Soft probability distribution (still continuous, not binary)
    """
    exp_features = np.exp(features / temperature)
    return exp_features / np.sum(exp_features, axis=1, keepdims=True)


def sign_binarization(features: np.ndarray) -> np.ndarray:
    """
    Sign binarization: Positive values -> 1, negative/zero -> 0.
    
    Args:
        features: (n_samples, n_features) array of continuous values
    
    Returns:
        Binary array of same shape
    """
    return (features > 0).astype(np.float32)


def normalize_features(features: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize features before binarization.
    
    Args:
        features: (n_samples, n_features) array
        method: 'l2', 'l1', 'max', or 'zscore'
    
    Returns:
        Normalized features
    """
    if method == 'l2':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms
    
    elif method == 'l1':
        norms = np.sum(np.abs(features), axis=1, keepdims=True)
        norms[norms == 0] = 1
        return features / norms
    
    elif method == 'max':
        maxs = np.max(np.abs(features), axis=1, keepdims=True)
        maxs[maxs == 0] = 1
        return features / maxs
    
    elif method == 'zscore':
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        std[std == 0] = 1
        return (features - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_code_statistics(codes: np.ndarray) -> dict:
    """
    Compute statistics about binary codes.
    
    Args:
        codes: (n_samples, code_length) binary array
    
    Returns:
        Dictionary with statistics
    """
    n_samples, code_length = codes.shape
    
    # Sparsity (per sample)
    sparsity_per_sample = 1.0 - np.mean(codes, axis=1)
    
    # Balance (per bit)
    balance_per_bit = np.mean(codes, axis=0)
    
    # Hamming distances between all pairs (expensive for large datasets)
    if n_samples <= 1000:
        # XOR and count: codes[i] XOR codes[j]
        hamming_dists = []
        for i in range(min(100, n_samples)):
            for j in range(i+1, min(100, n_samples)):
                dist = np.sum(codes[i] != codes[j])
                hamming_dists.append(dist)
        mean_hamming = np.mean(hamming_dists) if hamming_dists else 0
    else:
        # Sample-based estimate
        sample_size = min(100, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=False)
        hamming_dists = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                dist = np.sum(codes[indices[i]] != codes[indices[j]])
                hamming_dists.append(dist)
        mean_hamming = np.mean(hamming_dists)
    
    return {
        'code_length': code_length,
        'n_samples': n_samples,
        'mean_sparsity': float(np.mean(sparsity_per_sample)),
        'std_sparsity': float(np.std(sparsity_per_sample)),
        'mean_balance': float(np.mean(balance_per_bit)),
        'std_balance': float(np.std(balance_per_bit)),
        'mean_hamming_distance': float(mean_hamming),
    }


if __name__ == '__main__':
    # Test binarization methods
    np.random.seed(0)
    features = np.random.randn(100, 50)
    
    print("Testing binarization methods...")
    print(f"Input shape: {features.shape}")
    
    # Top-k
    binary_topk = top_k_binarization(features, k=5)
    print(f"\nTop-k (k=5): {np.sum(binary_topk, axis=1)[:5]} ones per sample")
    
    # Top-k%
    binary_topk_pct = top_k_percent_binarization(features, percent=0.1)
    print(f"Top-10%: {np.sum(binary_topk_pct, axis=1)[:5]} ones per sample")
    
    # Threshold
    binary_thresh = threshold_binarization(features)
    print(f"Threshold: {np.sum(binary_thresh, axis=1)[:5]} ones per sample")
    
    # WTA
    binary_wta = wta_binarization(features)
    print(f"WTA: {np.sum(binary_wta, axis=1)[:5]} ones per sample")
    
    # Statistics
    stats = compute_code_statistics(binary_topk)
    print(f"\nCode statistics (top-k):")
    for key, value in stats.items():
        print(f"  {key}: {value}")
