#!/usr/bin/env python
"""
BioHash Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: BioHash encoder + MNIST clustering pipeline.

Bio-inspired hashing with Hebbian-learned sparse projections and WTA binarization.
Combines fruit-fly olfactory circuit principles with data-adaptive Hebbian learning.

Algorithm:
    1. Initialize sparse random connectivity (10% connections)
    2. Hebbian learning: ΔW = η * y * x^T (with LR annealing + L2 norm)
    3. Encode: data @ W → WTA top-k → binary code

Pipeline:
    1. Load MNIST
    2. Train sparse Hebbian projection on training set
    3. Encode test set → WTA binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/biohash_mnist_clustering.py
    python standalone/biohash_mnist_clustering.py --hash_dim 256 --n_epochs 10

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
# BioHash — Hebbian sparse projection + WTA
# ═══════════════════════════════════════════════════════════════════════════════

def train_biohash(train_data, input_dim, hash_dim, sparse_ratio,
                  k_winners, n_epochs, batch_size, lr):
    """Train BioHash using Hebbian learning on sparse projections."""
    n_samples = len(train_data)
    n_connections = int(input_dim * sparse_ratio)

    # Initialize sparse projection matrix
    W = np.zeros((input_dim, hash_dim), dtype=np.float32)
    mask = np.zeros_like(W, dtype=bool)
    for j in range(hash_dim):
        indices = np.random.choice(input_dim, n_connections, replace=False)
        W[indices, j] = np.random.randn(n_connections).astype(np.float32)
        mask[indices, j] = True

    # L2 normalize columns
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms[norms == 0] = 1
    W /= norms

    print(f"\n[biohash] Training: {input_dim} → {hash_dim} (k={k_winners})")
    print(f"  sparse_ratio={sparse_ratio}, epochs={n_epochs}, lr={lr}")

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        n_batches = n_samples // batch_size

        for b in range(n_batches):
            batch = train_data[indices[b * batch_size:(b + 1) * batch_size]]
            activations = batch @ W
            topk_idx = np.argsort(activations, axis=1)[:, -k_winners:]
            winners_mask = np.zeros_like(activations)
            rows = np.arange(len(batch))[:, None]
            winners_mask[rows, topk_idx] = 1.0
            winners = activations * winners_mask

            delta_W = (batch.T @ winners) / len(batch)
            delta_W[~mask] = 0

            current_lr = lr * (1 - epoch / n_epochs)
            W += current_lr * delta_W

            norms = np.linalg.norm(W, axis=0, keepdims=True)
            norms[norms == 0] = 1
            W /= norms

        print(f"  Epoch {epoch+1}/{n_epochs} (lr={current_lr:.4f})")

    print("[biohash] Training complete!")
    return W


def biohash_encode(data, W, k_winners):
    """Encode: project + WTA top-k binarization."""
    pre_code = data @ W
    code = np.zeros_like(pre_code)
    top_k = np.argsort(pre_code, axis=1)[:, -k_winners:]
    rows = np.arange(len(data))[:, None]
    code[rows, top_k] = 1.0
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
        description="BioHash → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--hash_dim", type=int, default=256)
    parser.add_argument("--sparse_ratio", type=float, default=0.1)
    parser.add_argument("--k_winners", type=int, default=None,
                        help="Top-k winners (default: 5%% of hash_dim)")
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--n_clusters", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    train_data, train_labels, test_data, test_labels = load_mnist(args.data_root)

    if args.n_train is not None:
        train_data = train_data[:args.n_train]
        train_labels = train_labels[:args.n_train]
    if args.n_test is not None:
        test_data = test_data[:args.n_test]
        test_labels = test_labels[:args.n_test]

    input_dim = train_data.shape[1]
    k_winners = args.k_winners or max(1, int(args.hash_dim * 0.05))

    t0 = time.time()
    W = train_biohash(train_data, input_dim, args.hash_dim, args.sparse_ratio,
                      k_winners, args.n_epochs, args.batch_size, args.lr)
    print(f"[train] Training time: {time.time()-t0:.1f}s")

    t0 = time.time()
    pre_code, codes = biohash_encode(test_data, W, k_winners)
    print(f"[encode] Encoded {len(test_data)} samples ({time.time()-t0:.1f}s)")
    sparsity = 1 - codes.mean()
    print(f"[encode] Code shape: {codes.shape}, sparsity: {sparsity:.3f}")

    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : BioHash (Hebbian sparse projection)")
    print(f"  Dataset      : MNIST")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  Sparsity     : {sparsity:.3f}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
