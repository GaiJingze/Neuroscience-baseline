#!/usr/bin/env python
"""
SoftHebb Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: SoftHebb encoder + MNIST clustering pipeline.

Paper: "SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks"
       Moraitis et al., Neural Computation and Engineering 2022 / ICLR 2023

Algorithm:
    1. Multi-layer feedforward network with soft-WTA (temperature-scaled softmax)
    2. Self-normalizing Hebbian update: ΔW = η * y * (x - u * w)
    3. L2 row normalization after each update
    4. Encode: softmax output → top-k binarization

Pipeline:
    1. Load MNIST
    2. Train SoftHebb network with Hebbian learning
    3. Encode test set → binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/softhebb_mnist_clustering.py
    python standalone/softhebb_mnist_clustering.py --hidden_dims 1000 500 --output_dim 400

Requirements:
    pip install torch torchvision numpy scikit-learn scipy
"""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
# SoftHebb Network
# ═══════════════════════════════════════════════════════════════════════════════

class SoftHebbLayer(nn.Module):
    """Single SoftHebb layer: linear → softmax soft-WTA → Hebbian update."""

    def __init__(self, input_dim, output_dim, t_invert=5.0, eta=0.01):
        super().__init__()
        self.t_invert = t_invert
        self.eta = eta
        W_init = torch.randn(output_dim, input_dim)
        W_init = F.normalize(W_init, p=2, dim=1)
        self.W = nn.Parameter(W_init)

    def forward(self, x):
        pre_act = F.linear(x, self.W)
        wta = F.softmax(self.t_invert * pre_act, dim=1)
        return wta, pre_act

    def hebbian_update(self, x, wta, pre_act):
        with torch.no_grad():
            batch_size = x.size(0)
            yx = torch.matmul(wta.T, x) / batch_size
            yu = torch.sum(wta * pre_act, dim=0) / batch_size
            delta_W = yx - yu.unsqueeze(1) * self.W
            self.W.data += self.eta * delta_W
            self.W.data = F.normalize(self.W.data, p=2, dim=1)


class SoftHebbNetwork(nn.Module):
    """Multi-layer SoftHebb network."""

    def __init__(self, layer_dims, t_invert=5.0, eta=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                SoftHebbLayer(layer_dims[i], layer_dims[i+1], t_invert, eta))

    def forward(self, x):
        h = x
        for layer in self.layers:
            wta, _ = layer(h)
            h = wta
        return h

    def train_step(self, x):
        h = x
        for layer in self.layers:
            wta, pre_act = layer(h)
            layer.hebbian_update(h, wta, pre_act)
            h = wta


# ═══════════════════════════════════════════════════════════════════════════════
# Training + Encoding
# ═══════════════════════════════════════════════════════════════════════════════

def train_softhebb(train_data, layer_dims, t_invert, eta, n_epochs,
                   batch_size, device):
    """Train SoftHebb network. Returns the trained network."""
    network = SoftHebbNetwork(layer_dims, t_invert, eta).to(device)

    print(f"\n[softhebb] Training: {' → '.join(map(str, layer_dims))}")
    print(f"  t_invert={t_invert}, eta={eta}, epochs={n_epochs}, device={device}")

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_data).float())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    network.train()
    for epoch in range(n_epochs):
        for batch_idx, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)
            network.train_step(batch_x)
        print(f"  Epoch {epoch+1}/{n_epochs} complete")

    print("[softhebb] Training complete!")
    return network


def softhebb_encode(data, network, output_dim, batch_size, device,
                    binarize_percent=0.05):
    """Encode data using trained SoftHebb network."""
    network.eval()
    data_tensor = torch.from_numpy(data).float().to(device)

    all_outputs = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size]
            output = network(batch)
            all_outputs.append(output.cpu())

    pre_code = torch.cat(all_outputs, dim=0).numpy()

    k = max(int(output_dim * binarize_percent), 1)
    code = np.zeros_like(pre_code)
    top_k = np.argsort(pre_code, axis=1)[:, -k:]
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
        description="SoftHebb → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1000, 500])
    parser.add_argument("--output_dim", type=int, default=400)
    parser.add_argument("--t_invert", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--binarize_percent", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--n_clusters", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    train_data, train_labels, test_data, test_labels = load_mnist(args.data_root)
    if args.n_train is not None:
        train_data = train_data[:args.n_train]
    if args.n_test is not None:
        test_data = test_data[:args.n_test]
        test_labels = test_labels[:args.n_test]

    input_dim = train_data.shape[1]
    layer_dims = [input_dim] + args.hidden_dims + [args.output_dim]

    t0 = time.time()
    network = train_softhebb(train_data, layer_dims, args.t_invert, args.eta,
                              args.n_epochs, args.batch_size, args.device)
    print(f"[train] Training time: {time.time()-t0:.1f}s")

    t0 = time.time()
    pre_code, codes = softhebb_encode(
        test_data, network, args.output_dim, args.batch_size,
        args.device, args.binarize_percent)
    print(f"[encode] Encoded {len(test_data)} samples ({time.time()-t0:.1f}s)")
    sparsity = 1 - codes.mean()
    print(f"[encode] Code shape: {codes.shape}, sparsity: {sparsity:.3f}")

    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : SoftHebb (Moraitis et al. 2022/2023)")
    print(f"  Dataset      : MNIST")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Architecture : {' → '.join(map(str, layer_dims))}")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  Sparsity     : {sparsity:.3f}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
