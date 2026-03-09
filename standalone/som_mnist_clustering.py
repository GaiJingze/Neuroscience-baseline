#!/usr/bin/env python
"""
Self-Organizing Map (SOM) Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: SOM encoder + MNIST clustering pipeline.

Paper: "Self-organized formation of topologically correct feature maps"
       Kohonen, Biological Cybernetics, 1982

Algorithm:
    1. Initialize neuron grid with random data samples
    2. For each input: find BMU, update BMU + neighbors (Gaussian neighborhood)
    3. Learning rate and neighborhood radius decay over training
    4. Encode: negative distance to all neurons → top-k binarization

Pipeline:
    1. Load MNIST
    2. Train SOM on training set
    3. Encode test set → distance-based binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/som_mnist_clustering.py
    python standalone/som_mnist_clustering.py --map_height 20 --map_width 20

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
# SOM — Self-Organizing Map
# ═══════════════════════════════════════════════════════════════════════════════

def train_som(train_data, input_dim, map_height, map_width, n_epochs,
              lr_init, lr_final, sigma_init, sigma_final):
    """Train SOM using competitive learning. Returns weights [n_neurons, input_dim]."""
    n_neurons = map_height * map_width
    n_samples = len(train_data)

    # Grid positions
    positions = np.array(
        [[i, j] for i in range(map_height) for j in range(map_width)],
        dtype=np.float32)

    # Pairwise grid distances
    pos_diff = positions[:, None, :] - positions[None, :, :]
    grid_dist_sq = np.sum(pos_diff ** 2, axis=2)

    # Initialize weights from random data samples
    init_idx = np.random.choice(n_samples, n_neurons, replace=True)
    weights = train_data[init_idx].copy().astype(np.float32)
    weights += np.random.randn(*weights.shape).astype(np.float32) * 0.01

    if sigma_init is None:
        sigma_init = max(map_height, map_width) / 2.0

    total_steps = n_epochs * n_samples
    step = 0

    print(f"\n[som] Training: {map_height}x{map_width} = {n_neurons} neurons")
    print(f"  epochs={n_epochs}, lr={lr_init}→{lr_final}, sigma={sigma_init:.1f}→{sigma_final:.1f}")

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)

        for i in range(n_samples):
            x = train_data[indices[i]]
            progress = step / total_steps

            lr = lr_init * (lr_final / lr_init) ** progress
            sigma = sigma_init * (sigma_final / sigma_init) ** progress
            sigma_sq = sigma ** 2

            # Find BMU
            diffs = weights - x
            dists = np.sum(diffs ** 2, axis=1)
            bmu_idx = np.argmin(dists)

            # Gaussian neighborhood
            h = np.exp(-grid_dist_sq[bmu_idx] / (2 * sigma_sq))

            # Update weights
            weights += lr * h[:, np.newaxis] * (x - weights)
            step += 1

        print(f"  Epoch {epoch+1}/{n_epochs} (lr={lr:.4f}, sigma={sigma:.2f})")

    print("[som] Training complete!")
    return weights, positions, grid_dist_sq


def som_encode(data, weights, k_active):
    """Encode: negative distances to neurons → top-k binarization."""
    n_samples = len(data)
    n_neurons = len(weights)

    pre_code = np.zeros((n_samples, n_neurons), dtype=np.float32)
    batch_size = 512
    for i in range(0, n_samples, batch_size):
        batch = data[i:i + batch_size]
        diffs = batch[:, np.newaxis, :] - weights[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        pre_code[i:i + len(batch)] = -dists

    code = np.zeros_like(pre_code)
    top_k = np.argsort(pre_code, axis=1)[:, -k_active:]
    rows = np.arange(n_samples)[:, None]
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
        description="SOM → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--map_height", type=int, default=20)
    parser.add_argument("--map_width", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr_init", type=float, default=0.5)
    parser.add_argument("--lr_final", type=float, default=0.01)
    parser.add_argument("--sigma_init", type=float, default=None)
    parser.add_argument("--sigma_final", type=float, default=1.0)
    parser.add_argument("--k_active", type=int, default=None,
                        help="Top-k active neurons (default: 5%% of neurons)")
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

    n_neurons = args.map_height * args.map_width
    k_active = args.k_active or max(1, int(n_neurons * 0.05))

    t0 = time.time()
    weights, positions, grid_dist_sq = train_som(
        train_data, train_data.shape[1],
        args.map_height, args.map_width, args.n_epochs,
        args.lr_init, args.lr_final, args.sigma_init, args.sigma_final)
    print(f"[train] Training time: {time.time()-t0:.1f}s")

    t0 = time.time()
    pre_code, codes = som_encode(test_data, weights, k_active)
    print(f"[encode] Encoded {len(test_data)} samples ({time.time()-t0:.1f}s)")
    sparsity = 1 - codes.mean()
    print(f"[encode] Code shape: {codes.shape}, sparsity: {sparsity:.3f}")

    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : SOM (Kohonen 1982)")
    print(f"  Dataset      : MNIST")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Map size     : {args.map_height}x{args.map_width}")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  Sparsity     : {sparsity:.3f}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
