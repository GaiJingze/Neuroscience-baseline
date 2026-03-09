#!/usr/bin/env python
"""
Krotov-Hopfield Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: Krotov-Hopfield encoder + MNIST clustering.

Paper: "Unsupervised Learning by Competing Hidden Units"
       Krotov & Hopfield, PNAS 2019

Algorithm:
    1. Compute activation: Q = sign(W) * |W|^(p-1) @ X
    2. k-WTA: top-1 gets +1, top-k gets -delta (anti-Hebbian)
    3. Update: W += lr * (winner * X - winner * Q * W)
    4. Encode: top-5% activations → binary code

Pipeline:
    1. Load MNIST
    2. Train Krotov-Hopfield network
    3. Encode test set → binary code (top-k binarization)
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/krotov_mnist_clustering.py
    python standalone/krotov_mnist_clustering.py --n_neurons 400 --n_epochs 200

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
# Krotov-Hopfield — competing hidden units
# ═══════════════════════════════════════════════════════════════════════════════

def train_krotov(train_data, input_dim, n_neurons, n_epochs, batch_size,
                 lr, delta, p, k, prec=1e-30):
    """
    Train Krotov-Hopfield network.

    Returns weight matrix W [n_neurons, input_dim].
    """
    n_samples = len(train_data)
    W = np.random.normal(0.0, 1.0, (n_neurons, input_dim))

    n_batches = n_samples // batch_size
    print(f"\n[krotov] Training: {input_dim} → {n_neurons} neurons")
    print(f"  epochs={n_epochs}, delta={delta}, p={p}, k={k}")

    for epoch in range(n_epochs):
        eps = lr * (1 - epoch / n_epochs)
        shuffled = np.random.permutation(n_samples)
        data_shuffled = train_data[shuffled]

        for b in range(n_batches):
            batch = data_shuffled[b * batch_size:(b + 1) * batch_size]
            inputs = batch.T  # [input_dim, batch_size]

            sig = np.sign(W)
            weights_powered = sig * np.abs(W) ** (p - 1)
            tot_input = np.dot(weights_powered, inputs)  # [n_neurons, batch_size]

            y = np.argsort(tot_input, axis=0)
            yl = np.zeros((n_neurons, batch_size))
            yl[y[-1, :], np.arange(batch_size)] = 1.0
            yl[y[-k, :], np.arange(batch_size)] = -delta

            hebbian = np.dot(yl, inputs.T)
            xx = np.sum(yl * tot_input, axis=1)
            anti_hebbian = np.outer(xx, np.ones(input_dim)) * W

            ds = hebbian - anti_hebbian
            nc = max(np.max(np.abs(ds)), prec)
            W += eps * ds / nc

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} (lr={eps:.4f})")

    print("[krotov] Training complete!")
    return W


def krotov_encode(data, W, p, binarize_percent=0.05):
    """Encode: power-law activation + top-k binarization."""
    sig = np.sign(W)
    weights_powered = sig * np.abs(W) ** (p - 1)
    pre_code = np.dot(weights_powered, data.T).T  # [n_samples, n_neurons]

    k_active = max(int(W.shape[0] * binarize_percent), 1)
    code = np.zeros_like(pre_code)
    top_k = np.argsort(pre_code, axis=1)[:, -k_active:]
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
        description="Krotov-Hopfield → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--n_neurons", type=int, default=400)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=0.4)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--binarize_percent", type=float, default=0.05)
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

    t0 = time.time()
    W = train_krotov(train_data, input_dim, args.n_neurons, args.n_epochs,
                     args.batch_size, args.lr, args.delta, args.p, args.k)
    print(f"[train] Training time: {time.time()-t0:.1f}s")

    t0 = time.time()
    pre_code, codes = krotov_encode(test_data, W, args.p, args.binarize_percent)
    print(f"[encode] Encoded {len(test_data)} samples ({time.time()-t0:.1f}s)")
    sparsity = 1 - codes.mean()
    print(f"[encode] Code shape: {codes.shape}, sparsity: {sparsity:.3f}")

    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : Krotov-Hopfield (2019)")
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
