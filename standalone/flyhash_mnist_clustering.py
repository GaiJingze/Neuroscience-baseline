#!/usr/bin/env python
"""
FlyHash Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: FlyHash encoder + MNIST clustering pipeline.

Paper: "A neural algorithm for a fundamental computing problem"
       Dasgupta, Stevens & Navlakha, Science 2017

Algorithm:
    1. Sparse random binary projection matrix (each KC connects to ~10% of PNs)
    2. Random expansion: pre_code = data @ projection_matrix
    3. Winner-Take-All: keep top-k activations as binary code

Pipeline:
    1. Load MNIST (60k train / 10k test, 784-D, [0,1])
    2. Initialize sparse random projection (no training needed)
    3. Encode test set → expanded projections → WTA binary code
    4. Cluster with K-Means (k=10) and evaluate NMI / ARI / ACC

Usage:
    python standalone/flyhash_mnist_clustering.py
    python standalone/flyhash_mnist_clustering.py --projection_dim 15680 --hash_length 784

Requirements:
    pip install torch torchvision numpy scikit-learn scipy
"""

import argparse
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)
from torchvision import datasets, transforms


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"[seed] Random seed set to {seed}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset — MNIST
# ═══════════════════════════════════════════════════════════════════════════════

def load_mnist(data_root: str = "./data"):
    """Load MNIST, flatten to (N, 784), normalise to [0, 1]."""
    train_ds = datasets.MNIST(root=data_root, train=True, download=True,
                              transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root=data_root, train=False, download=True,
                             transform=transforms.ToTensor())

    train_data = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    train_labels = train_ds.targets.numpy()
    test_data = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    test_labels = test_ds.targets.numpy()

    print(f"[data] MNIST loaded — train: {train_data.shape}, test: {test_data.shape}")
    return train_data, train_labels, test_data, test_labels


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FlyHash — sparse random projection + WTA
# ═══════════════════════════════════════════════════════════════════════════════

def create_sparse_projection(input_dim: int, projection_dim: int,
                              sampling_ratio: float = 0.1) -> np.ndarray:
    """
    Create sparse binary projection matrix.

    Each output neuron (KC) connects to ~sampling_ratio of input neurons (PNs)
    with binary weights (0/1).
    """
    n_connections = max(1, int(input_dim * sampling_ratio))
    projection = np.zeros((input_dim, projection_dim), dtype=np.float32)

    for j in range(projection_dim):
        selected = np.random.choice(input_dim, size=n_connections, replace=False)
        projection[selected, j] = 1.0

    return projection


def flyhash_encode(data: np.ndarray, projection_matrix: np.ndarray,
                   hash_length: int) -> tuple:
    """
    Encode data using FlyHash: sparse projection + WTA.

    Returns:
        (pre_code, code) — continuous projections and binary WTA codes
    """
    # Step 1: Random expansion
    pre_code = data @ projection_matrix  # (n_samples, projection_dim)

    # Step 2: Winner-Take-All (top-k)
    code = np.zeros_like(pre_code)
    top_k = np.argsort(pre_code, axis=1)[:, -hash_length:]
    rows = np.arange(len(data))[:, None]
    code[rows, top_k] = 1.0

    return pre_code, code


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Clustering — K-Means + metrics
# ═══════════════════════════════════════════════════════════════════════════════

def clustering_accuracy(labels_true, labels_pred):
    n_clusters = max(labels_true.max(), labels_pred.max()) + 1
    contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        contingency[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    return float(contingency[row_ind, col_ind].sum()) / len(labels_true)


def clustering_purity(labels_true, labels_pred):
    contingency = np.zeros((labels_pred.max() + 1, labels_true.max() + 1), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        contingency[p, t] += 1
    return float(contingency.max(axis=1).sum()) / len(labels_true)


def evaluate_clustering(codes, labels_true, n_clusters=10):
    print(f"\n{'='*70}")
    print(f"[cluster] Running K-Means (k={n_clusters}) on {codes.shape} codes ...")
    print(f"{'='*70}")

    kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=0)
    labels_pred = kmeans.fit_predict(codes)

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    acc = clustering_accuracy(labels_true, labels_pred)
    purity = clustering_purity(labels_true, labels_pred)
    n_unique = len(set(labels_pred))
    sil = silhouette_score(codes, labels_pred) if n_unique > 1 else 0.0
    db = davies_bouldin_score(codes, labels_pred) if n_unique > 1 else float("inf")

    print(f"\n  {'Metric':<25s} {'Value':>10s}")
    print(f"  {'-'*36}")
    print(f"  {'NMI':<25s} {nmi:>10.4f}")
    print(f"  {'ARI':<25s} {ari:>10.4f}")
    print(f"  {'ACC (Hungarian)':<25s} {acc:>10.4f}")
    print(f"  {'Purity':<25s} {purity:>10.4f}")
    print(f"  {'Silhouette':<25s} {sil:>10.4f}")
    print(f"  {'Davies-Bouldin (↓)':<25s} {db:>10.4f}")
    print()

    return {"nmi": nmi, "ari": ari, "acc": acc,
            "purity": purity, "silhouette": sil, "davies_bouldin": db}


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FlyHash → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")

    # FlyHash parameters
    parser.add_argument("--projection_dim", type=int, default=15680,
                        help="Expansion dimension (default: 20x input_dim)")
    parser.add_argument("--hash_length", type=int, default=784,
                        help="Number of top-k winners (5%% of projection_dim)")
    parser.add_argument("--sampling_ratio", type=float, default=0.1,
                        help="Connection probability per KC")

    # Subset
    parser.add_argument("--n_test", type=int, default=None)

    # Clustering
    parser.add_argument("--n_clusters", type=int, default=10)

    args = parser.parse_args()
    set_seed(args.seed)

    # 1. Load MNIST
    train_data, train_labels, test_data, test_labels = load_mnist(args.data_root)

    if args.n_test is not None:
        test_data = test_data[:args.n_test]
        test_labels = test_labels[:args.n_test]
        print(f"[data] Subsampled to {args.n_test} test samples")

    input_dim = train_data.shape[1]

    # 2. Create sparse projection matrix (no training)
    print(f"\n[flyhash] Creating sparse projection: "
          f"{input_dim} → {args.projection_dim} (k={args.hash_length})")
    t0 = time.time()
    proj = create_sparse_projection(input_dim, args.projection_dim,
                                     args.sampling_ratio)
    print(f"[flyhash] Projection ready ({time.time()-t0:.1f}s)")

    # 3. Encode test set
    t0 = time.time()
    pre_code, codes = flyhash_encode(test_data, proj, args.hash_length)
    print(f"[encode] Encoded {len(test_data)} samples ({time.time()-t0:.1f}s)")
    sparsity = 1 - codes.mean()
    print(f"[encode] Code shape: {codes.shape}, sparsity: {sparsity:.3f}")

    # 4. Cluster & evaluate
    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    # Summary
    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : FlyHash (Dasgupta et al. 2017)")
    print(f"  Dataset      : MNIST")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  Sparsity     : {sparsity:.3f}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
