"""
Simple tests for pipeline modules.
Run with: pytest tests/test_pipeline.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    compute_nmi, compute_ari, compute_clustering_accuracy,
    compute_map, compute_recall_at_k,
    top_k_binarization, top_k_percent_binarization,
    set_seed
)


def test_clustering_metrics():
    """Test clustering metrics."""
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([1, 1, 0, 2, 2, 2, 0, 0, 0])
    
    nmi = compute_nmi(labels_true, labels_pred)
    ari = compute_ari(labels_true, labels_pred)
    acc = compute_clustering_accuracy(labels_true, labels_pred)
    
    assert 0 <= nmi <= 1, "NMI should be in [0, 1]"
    assert -1 <= ari <= 1, "ARI should be in [-1, 1]"
    assert 0 <= acc <= 1, "ACC should be in [0, 1]"
    
    print(f"Clustering metrics test passed: NMI={nmi:.3f}, ARI={ari:.3f}, ACC={acc:.3f}")


def test_retrieval_metrics():
    """Test retrieval metrics."""
    retrieved = np.array([[0, 1, 2, 3], [5, 6, 7, 8]])
    groundtruth = np.array([[0, 1, 10], [5, 6, 11]])
    
    map_score = compute_map(retrieved, groundtruth, k=4)
    recall_2 = compute_recall_at_k(retrieved, groundtruth, k=2)
    
    assert 0 <= map_score <= 1, "mAP should be in [0, 1]"
    assert 0 <= recall_2 <= 1, "Recall should be in [0, 1]"
    
    print(f"Retrieval metrics test passed: mAP={map_score:.3f}, Recall@2={recall_2:.3f}")


def test_binarization():
    """Test binarization methods."""
    set_seed(0)
    features = np.random.randn(100, 50)
    
    # Top-k
    binary_topk = top_k_binarization(features, k=5)
    assert binary_topk.shape == features.shape
    assert np.all(np.sum(binary_topk, axis=1) == 5), "Each sample should have exactly k=5 ones"
    
    # Top-k%
    binary_topk_pct = top_k_percent_binarization(features, percent=0.1)
    assert binary_topk_pct.shape == features.shape
    expected_k = int(features.shape[1] * 0.1)
    assert np.all(np.sum(binary_topk_pct, axis=1) == expected_k), f"Each sample should have {expected_k} ones"
    
    print("Binarization test passed")


def test_seed_reproducibility():
    """Test that seed setting works."""
    set_seed(42)
    a = np.random.randn(10)
    
    set_seed(42)
    b = np.random.randn(10)
    
    assert np.allclose(a, b), "Same seed should produce same random numbers"
    print("Seed reproducibility test passed")


if __name__ == '__main__':
    print("Running pipeline tests...")
    print()
    
    test_clustering_metrics()
    test_retrieval_metrics()
    test_binarization()
    test_seed_reproducibility()
    
    print()
    print("All tests passed! ✓")
