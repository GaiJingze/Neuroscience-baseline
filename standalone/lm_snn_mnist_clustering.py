#!/usr/bin/env python
"""
LM-SNN (Lattice Map SNN) Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: LM-SNN encoder + MNIST clustering.

Paper: "Unsupervised Learning with Self-Organizing Spiking Neural Networks"
       Hazan, Saunders et al., IJCNN 2018 / Annals of Math & AI 2020

Algorithm:
    Like Diehl & Cook but with topological (SOM-inspired) inhibition:
    - Neurons arranged on a 2D grid
    - Inhibitory weights decay with grid distance (Gaussian falloff)
    - Encourages spatially-organized receptive fields

Pipeline:
    1. Load MNIST
    2. Train STDP-WTA SNN with topological inhibition
    3. Encode test set → spike counts → top-k binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/lm_snn_mnist_clustering.py --n_train 1000
    python standalone/lm_snn_mnist_clustering.py --n_neurons 400 --device cuda

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
# LM-SNN — Lattice Map SNN with topological inhibition
# ═══════════════════════════════════════════════════════════════════════════════

def build_lm_snn(n_input, n_neurons, grid_side, inh_sigma, inh_strength,
                 dt=1.0, nu=(1e-4, 1e-2), thresh=-52.0, rest=-65.0):
    """Build SNN with distance-dependent (Gaussian) lateral inhibition."""
    network = Network(dt=dt)

    input_layer = Input(n=n_input, traces=True)
    exc_layer = DiehlAndCookNodes(n=n_neurons, traces=True, rest=rest,
                                  reset=-65.0, thresh=thresh, refrac=5,
                                  tc_decay=100.0, trace_tc=5e-2,
                                  theta_plus=0.05, tc_theta_decay=1e7)
    inh_layer = LIFNodes(n=n_neurons, traces=False, rest=-60.0, reset=-45.0,
                         thresh=-40.0, tc_decay=10.0, refrac=2, trace_tc=5e-2)

    network.add_layer(input_layer, name="Input")
    network.add_layer(exc_layer, name="Excitatory")
    network.add_layer(inh_layer, name="Inhibitory")

    # Input → Excitatory (STDP)
    w_inp = 0.3 * torch.rand(n_input, n_neurons)
    network.add_connection(
        Connection(source=input_layer, target=exc_layer, w=w_inp,
                   update_rule=PostPre, nu=nu, reduction=None,
                   wmin=0.0, wmax=1.0, norm=78.4),
        source="Input", target="Excitatory")

    # Exc → Inh (one-to-one)
    w_ei = 22.5 * torch.eye(n_neurons)
    network.add_connection(
        Connection(source=exc_layer, target=inh_layer, w=w_ei, wmin=0, wmax=22.5),
        source="Excitatory", target="Inhibitory")

    # Inh → Exc (TOPOLOGICAL: Gaussian distance-dependent)
    positions = np.array([[idx // grid_side, idx % grid_side]
                          for idx in range(n_neurons)], dtype=np.float32)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    gauss = np.exp(-dist ** 2 / (2.0 * inh_sigma ** 2))
    np.fill_diagonal(gauss, 0.0)
    w_ie = torch.tensor(inh_strength * gauss, dtype=torch.float32)
    network.add_connection(
        Connection(source=inh_layer, target=exc_layer, w=w_ie,
                   wmin=-120.0, wmax=0.0),
        source="Inhibitory", target="Excitatory")

    # Monitor
    network.add_monitor(Monitor(exc_layer, state_vars=["s"]),
                        name="ExcitatoryMonitor")
    return network


def train_encode(network, data, n_neurons, sim_time, dt, device, training=True):
    """Run data through network (train or eval). Returns spike counts."""
    network.train(mode=training)
    network.to(device)
    n_samples = len(data)
    spike_counts = np.zeros((n_samples, n_neurons), dtype=np.float32)

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n_samples}]")
        image_tensor = torch.from_numpy(data[i]).float()
        encoded = poisson(datum=image_tensor, time=int(sim_time), dt=dt)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.to(device)
        network.run(inputs={"Input": encoded}, time=int(sim_time))
        spikes = network.monitors["ExcitatoryMonitor"].get("s")
        counts = torch.sum(spikes, dim=0).cpu().numpy().flatten()
        spike_counts[i] = counts[:n_neurons]
        network.reset_state_variables()

    return spike_counts


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
        description="LM-SNN → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--n_neurons", type=int, default=400)
    parser.add_argument("--sim_time", type=int, default=350)
    parser.add_argument("--intensity", type=float, default=128.0)
    parser.add_argument("--inh_sigma", type=float, default=None,
                        help="Inhibition Gaussian sigma (default: grid_side/4)")
    parser.add_argument("--inh_strength", type=float, default=-17.5)
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

    grid_side = int(np.sqrt(args.n_neurons))
    n_neurons = grid_side ** 2
    inh_sigma = args.inh_sigma or grid_side / 4.0
    dt = 1.0
    nu = (1e-4, 1e-2)

    print(f"\n{'='*70}")
    print(f"LM-SNN: 784 → {n_neurons} neurons ({grid_side}x{grid_side} grid)")
    print(f"  inh_sigma={inh_sigma:.1f}, inh_strength={args.inh_strength}")
    print(f"{'='*70}")

    network = build_lm_snn(784, n_neurons, grid_side, inh_sigma,
                            args.inh_strength, dt, nu)

    # Train
    train_scaled = np.clip(train_data, 0, 1) * args.intensity
    print(f"\n[train] Training on {len(train_data)} samples ...")
    t0 = time.time()
    train_encode(network, train_scaled, n_neurons, args.sim_time, dt,
                  args.device, training=True)
    print(f"[train] Training time: {(time.time()-t0)/60:.1f} min")

    # Encode test set
    test_scaled = np.clip(test_data, 0, 1) * args.intensity
    print(f"\n[encode] Encoding {len(test_data)} test samples ...")
    t0 = time.time()
    spike_counts = train_encode(network, test_scaled, n_neurons, args.sim_time,
                                 dt, args.device, training=False)
    print(f"[encode] Encoding time: {(time.time()-t0)/60:.1f} min")

    total = spike_counts.sum(axis=1)
    print(f"  Spikes/sample: mean={total.mean():.1f}, "
          f"median={np.median(total):.1f}, zeros={int((total==0).sum())}")

    codes = binarize_top_k(spike_counts, args.binarize_percent)
    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : LM-SNN (Hazan et al. 2018)")
    print(f"  Dataset      : MNIST")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Grid         : {grid_side}x{grid_side} = {n_neurons} neurons")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
