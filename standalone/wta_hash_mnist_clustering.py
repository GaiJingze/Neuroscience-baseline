#!/usr/bin/env python
"""
WTA Hash Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: WTA Hash encoder + MNIST clustering pipeline.

Paper: "The Power of Comparative Reasoning"
       Yagnik et al., ICCV 2011

Algorithm:
    1. Randomly group input features into windows
    2. Within each window, only the argmax position fires (one-hot)
    3. Concatenate → sparse binary code

Pipeline:
    1. Load MNIST
    2. Initialize random feature permutations (no training)
    3. Encode test set → windowed WTA binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/wta_hash_mnist_clustering.py
    python standalone/wta_hash_mnist_clustering.py --n_hashes 128 --window_size 16

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


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[seed] Random seed set to {seed}")


def load_mnist(data_root: str = "./data"):
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
# WTA Hash — random windowing + local WTA
# ═══════════════════════════════════════════════════════════════════════════════

def create_permutations(input_dim: int, n_hashes: int,
                         window_size: int) -> np.ndarray:
    """Generate random feature selections for each window."""
    perms = np.zeros((n_hashes, window_size), dtype=np.int64)
    for h in range(n_hashes):
        perms[h] = np.random.choice(input_dim, size=window_size, replace=False)
    return perms


def wta_hash_encode(data: np.ndarray, perms: np.ndarray,
                     n_hashes: int, window_size: int) -> tuple:
    """WTA hash: per-window argmax → one-hot → concatenation."""
    n_samples = len(data)
    output_dim = n_hashes * window_size
    pre_code = np.zeros((n_samples, output_dim), dtype=np.float32)
    code = np.zeros((n_samples, output_dim), dtype=np.float32)

    for h in range(n_hashes):
        windowed = data[:, perms[h]]
        offset = h * window_size
        pre_code[:, offset:offset + window_size] = windowed
        max_idx = np.argmax(windowed, axis=1)
        code[np.arange(n_samples), offset + max_idx] = 1.0

    return pre_code, code


# ═══════════════════════════════════════════════════════════════════════════════
# Clustering
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
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WTA Hash → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--n_hashes", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--n_clusters", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    train_data, train_labels, test_data, test_labels = load_mnist(args.data_root)
    if args.n_test is not None:
        test_data = test_data[:args.n_test]
        test_labels = test_labels[:args.n_test]

    input_dim = train_data.shape[1]
    output_dim = args.n_hashes * args.window_size
    print(f"\n[wta_hash] {input_dim} → {output_dim} "
          f"({args.n_hashes} windows x {args.window_size})")

    perms = create_permutations(input_dim, args.n_hashes, args.window_size)

    t0 = time.time()
    pre_code, codes = wta_hash_encode(test_data, perms,
                                       args.n_hashes, args.window_size)
    print(f"[encode] Encoded {len(test_data)} samples ({time.time()-t0:.1f}s)")
    sparsity = 1 - codes.mean()
    print(f"[encode] Code shape: {codes.shape}, sparsity: {sparsity:.3f}")

    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : WTA Hash (Yagnik et al. 2011)")
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
