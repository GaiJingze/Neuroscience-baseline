#!/usr/bin/env python
"""
LC-SNN (Locally Connected SNN) Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: LC-SNN encoder + MNIST clustering.

Paper: "Locally Connected Spiking Neural Networks for Unsupervised Feature Learning"
       Saunders, Patel, Hazan, Siegelmann & Kozma, Neural Networks 2019

Algorithm:
    - Input image divided into non-overlapping local patches
    - Each patch has its own group of excitatory neurons trained with STDP
    - Local WTA inhibition within each patch group
    - Output: concatenation of spike counts across all patch groups

Pipeline:
    1. Load MNIST
    2. Train per-patch STDP-WTA networks
    3. Encode test set → spike counts → top-k binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/lc_snn_mnist_clustering.py --n_train 1000
    python standalone/lc_snn_mnist_clustering.py --neurons_per_patch 100

Requirements:
    pip install torch torchvision numpy scikit-learn scipy
    pip install git+https://github.com/BindsNET/bindsnet.git
"""

import argparse
import random
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)
from torchvision import datasets, transforms

from bindsnet.encoding import poisson
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection


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
# LC-SNN — Locally Connected SNN
# ═══════════════════════════════════════════════════════════════════════════════

def extract_patches(images, image_side, patch_size, patch_stride):
    """Extract patches: (n_samples, input_dim) → (n_samples, n_patches, patch_dim)."""
    n = len(images)
    n_patches_side = (image_side - patch_size) // patch_stride + 1
    n_patches = n_patches_side ** 2
    patch_dim = patch_size ** 2
    imgs = images.reshape(n, image_side, image_side)
    patches = np.zeros((n, n_patches, patch_dim))
    idx = 0
    for r in range(n_patches_side):
        for c in range(n_patches_side):
            r0 = r * patch_stride
            c0 = c * patch_stride
            patch = imgs[:, r0:r0+patch_size, c0:c0+patch_size]
            patches[:, idx] = patch.reshape(n, -1)
            idx += 1
    return patches


def build_patch_network(patch_dim, neurons_per_patch, dt=1.0,
                         nu=(1e-4, 1e-2), thresh=-52.0, rest=-65.0):
    """Build a small SNN for a single patch."""
    network = Network(dt=dt)
    inp = Input(n=patch_dim, traces=True)
    exc = DiehlAndCookNodes(n=neurons_per_patch, traces=True, rest=rest,
                            reset=-65.0, thresh=thresh, refrac=5,
                            tc_decay=100.0, trace_tc=5e-2,
                            theta_plus=0.05, tc_theta_decay=1e7)
    inh = LIFNodes(n=neurons_per_patch, traces=False, rest=-60.0, reset=-45.0,
                   thresh=-40.0, tc_decay=10.0, refrac=2, trace_tc=5e-2)
    network.add_layer(inp, name="Input")
    network.add_layer(exc, name="Excitatory")
    network.add_layer(inh, name="Inhibitory")

    norm_val = 78.4 * (patch_dim / 784.0)
    w = 0.3 * torch.rand(patch_dim, neurons_per_patch)
    network.add_connection(
        Connection(source=inp, target=exc, w=w, update_rule=PostPre, nu=nu,
                   reduction=None, wmin=0.0, wmax=1.0, norm=norm_val),
        source="Input", target="Excitatory")
    w_ei = 22.5 * torch.eye(neurons_per_patch)
    network.add_connection(
        Connection(source=exc, target=inh, w=w_ei, wmin=0, wmax=22.5),
        source="Excitatory", target="Inhibitory")
    w_ie = -17.5 * (torch.ones(neurons_per_patch, neurons_per_patch) -
                     torch.diag(torch.ones(neurons_per_patch)))
    network.add_connection(
        Connection(source=inh, target=exc, w=w_ie, wmin=-120.0, wmax=0.0),
        source="Inhibitory", target="Excitatory")
    network.add_monitor(Monitor(exc, state_vars=["s"]), name="ExcitatoryMonitor")
    return network


def binarize_top_k(features, percent=0.05):
    n_features = features.shape[1]
    k = max(1, int(n_features * percent))
    binary = np.zeros_like(features)
    top_k_idx = np.argsort(features, axis=1)[:, -k:]
    rows = np.arange(len(features))[:, None]
    binary[rows, top_k_idx] = 1
    sparsity = 1.0 - binary.mean()
    unique = len(np.unique(binary, axis=0))
    print(f"[binarize] top-{percent*100:.0f}%  k={k}  "
          f"sparsity={sparsity:.3f}  unique_codes={unique}/{len(features)}")
    return binary


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
        description="LC-SNN → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--patch_stride", type=int, default=14)
    parser.add_argument("--neurons_per_patch", type=int, default=100)
    parser.add_argument("--sim_time", type=int, default=350)
    parser.add_argument("--intensity", type=float, default=128.0)
    parser.add_argument("--binarize_percent", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--n_clusters", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    train_data, train_labels, test_data, test_labels = load_mnist(args.data_root)
    if args.n_train is not None:
        train_data = train_data[:args.n_train]
        train_labels = train_labels[:args.n_train]
    if args.n_test is not None:
        test_data = test_data[:args.n_test]
        test_labels = test_labels[:args.n_test]

    image_side = 28
    n_patches_side = (image_side - args.patch_size) // args.patch_stride + 1
    n_patches = n_patches_side ** 2
    patch_dim = args.patch_size ** 2
    n_neurons = n_patches * args.neurons_per_patch
    dt = 1.0
    nu = (1e-4, 1e-2)

    print(f"\n{'='*70}")
    print(f"LC-SNN: {n_patches} patches ({n_patches_side}x{n_patches_side}), "
          f"{args.neurons_per_patch} neurons/patch, total={n_neurons}")
    print(f"{'='*70}")

    # Build per-patch networks
    networks = []
    for p in range(n_patches):
        net = build_patch_network(patch_dim, args.neurons_per_patch, dt, nu)
        net.to(args.device)
        networks.append(net)

    # Scale + extract patches
    train_scaled = np.clip(train_data, 0, 1) * args.intensity
    train_patches = extract_patches(train_scaled, image_side,
                                     args.patch_size, args.patch_stride)

    # Train
    print(f"[train] Training on {len(train_data)} samples ...")
    t0 = time.time()
    for i in range(len(train_data)):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(train_data) - (i + 1)) / speed / 60
            print(f"  [{i+1}/{len(train_data)}] "
                  f"speed={speed:.1f} samp/s  ETA={eta:.1f} min")
        for p in range(n_patches):
            patch_tensor = torch.from_numpy(train_patches[i, p]).float()
            encoded = poisson(datum=patch_tensor, time=int(args.sim_time), dt=dt)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(args.device)
            networks[p].run(inputs={"Input": encoded}, time=int(args.sim_time))
            networks[p].reset_state_variables()

    print(f"[train] Training time: {(time.time()-t0)/60:.1f} min")

    # Encode test set
    print(f"\n[encode] Encoding {len(test_data)} test samples ...")
    for net in networks:
        net.train(mode=False)

    test_scaled = np.clip(test_data, 0, 1) * args.intensity
    test_patches = extract_patches(test_scaled, image_side,
                                    args.patch_size, args.patch_stride)
    spike_counts = np.zeros((len(test_data), n_neurons))

    t0 = time.time()
    for i in range(len(test_data)):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(test_data)}]")
        for p in range(n_patches):
            patch_tensor = torch.from_numpy(test_patches[i, p]).float()
            encoded = poisson(datum=patch_tensor, time=int(args.sim_time), dt=dt)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(args.device)
            networks[p].run(inputs={"Input": encoded}, time=int(args.sim_time))
            spikes = networks[p].monitors["ExcitatoryMonitor"].get("s")
            counts = torch.sum(spikes, dim=0).cpu().numpy().flatten()
            offset = p * args.neurons_per_patch
            spike_counts[i, offset:offset+args.neurons_per_patch] = \
                counts[:args.neurons_per_patch]
            networks[p].reset_state_variables()

    print(f"[encode] Encoding time: {(time.time()-t0)/60:.1f} min")

    codes = binarize_top_k(spike_counts, args.binarize_percent)
    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : LC-SNN (Saunders et al. 2019)")
    print(f"  Dataset      : MNIST")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Patches      : {n_patches} ({n_patches_side}x{n_patches_side})")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
