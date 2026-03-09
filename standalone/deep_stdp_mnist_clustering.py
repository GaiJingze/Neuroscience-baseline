#!/usr/bin/env python
"""
Deep STDP Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: Deep STDP encoder + MNIST clustering.

Paper: "Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity"
       Sen Lu & Abhronil Sengupta, Neuromorphic Computing and Engineering, 2024

Algorithm:
    Multi-layer STDP-WTA with alternating STDP + K-means pseudo-labeling:
    1. Layer l: STDP training → spike count features
    2. K-means on features → pseudo-labels → reorder data by cluster
    3. Repeat STDP with reordered data (encourages specialization)
    4. Features become input for layer l+1

Pipeline:
    1. Load MNIST
    2. Train multi-layer STDP-WTA network with pseudo-label guidance
    3. Encode test set → spike counts → top-k binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/deep_stdp_mnist_clustering.py --n_train 1000
    python standalone/deep_stdp_mnist_clustering.py --layer_sizes 400 200 --device cuda

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
# Deep STDP Network
# ═══════════════════════════════════════════════════════════════════════════════

def build_layer(in_dim, n_neurons, dt=1.0, nu=(1e-4, 1e-2),
                thresh=-52.0, rest=-65.0, refrac=5):
    """Build one STDP-WTA spiking layer."""
    network = Network(dt=dt)
    inp = Input(n=in_dim, traces=True)
    exc = DiehlAndCookNodes(n=n_neurons, traces=True, rest=rest, reset=-65.0,
                            thresh=thresh, refrac=refrac, tc_decay=100.0,
                            trace_tc=5e-2, theta_plus=0.05, tc_theta_decay=1e7)
    inh = LIFNodes(n=n_neurons, traces=False, rest=-60.0, reset=-45.0,
                   thresh=-40.0, tc_decay=10.0, refrac=2, trace_tc=5e-2)
    network.add_layer(inp, name="Input")
    network.add_layer(exc, name="Excitatory")
    network.add_layer(inh, name="Inhibitory")

    norm_val = 78.4 * (in_dim / 784.0)
    w = 0.3 * torch.rand(in_dim, n_neurons)
    network.add_connection(
        Connection(source=inp, target=exc, w=w, update_rule=PostPre, nu=nu,
                   reduction=None, wmin=0.0, wmax=1.0, norm=norm_val),
        source="Input", target="Excitatory")
    w_ei = 22.5 * torch.eye(n_neurons)
    network.add_connection(
        Connection(source=exc, target=inh, w=w_ei, wmin=0, wmax=22.5),
        source="Excitatory", target="Inhibitory")
    w_ie = -17.5 * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
    network.add_connection(
        Connection(source=inh, target=exc, w=w_ie, wmin=-120.0, wmax=0.0),
        source="Inhibitory", target="Excitatory")
    network.add_monitor(Monitor(exc, state_vars=["s"]), name="ExcitatoryMonitor")
    return network


def run_layer(network, data, n_neurons, sim_time, dt, device, training=True):
    """Run data through one STDP-WTA layer, return spike counts."""
    network.train(mode=training)
    network.to(device)
    n_samples = len(data)
    spike_counts = np.zeros((n_samples, n_neurons))

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"    Sample {i+1}/{n_samples}")
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
    unique_codes = len(np.unique(binary, axis=0))
    print(f"[binarize] top-{percent*100:.0f}%  k={k}  "
          f"sparsity={sparsity:.3f}  unique_codes={unique_codes}/{len(features)}")
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
        description="Deep STDP → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--layer_sizes", type=int, nargs="+", default=[400, 200])
    parser.add_argument("--n_rounds", type=int, default=2)
    parser.add_argument("--n_clusters_pseudo", type=int, default=10)
    parser.add_argument("--sim_time", type=int, default=350)
    parser.add_argument("--intensity", type=float, default=128.0)
    parser.add_argument("--nu_pre", type=float, default=1e-4)
    parser.add_argument("--nu_post", type=float, default=1e-2)
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

    # Scale for Poisson encoding
    data_scaled = np.clip(train_data, 0, 1) * args.intensity
    nu = (args.nu_pre, args.nu_post)
    dt = 1.0

    print(f"\n{'='*70}")
    print(f"Deep STDP: layers={[784]+args.layer_sizes}, rounds={args.n_rounds}")
    print(f"{'='*70}")

    layer_networks = []
    current_input = data_scaled
    current_dim = 784
    t0 = time.time()

    for layer_idx, n_neurons in enumerate(args.layer_sizes):
        print(f"\n--- Layer {layer_idx+1}/{len(args.layer_sizes)}: "
              f"{current_dim} → {n_neurons} ---")

        prev_features = None
        for r in range(args.n_rounds):
            print(f"  Round {r+1}/{args.n_rounds}:")
            network = build_layer(current_dim, n_neurons, dt, nu)
            train_input = current_input
            if r > 0 and prev_features is not None:
                norms = np.linalg.norm(prev_features, axis=1, keepdims=True)
                norms[norms == 0] = 1
                km = KMeans(n_clusters=args.n_clusters_pseudo, n_init=3,
                            max_iter=100, random_state=42)
                pseudo = km.fit_predict(prev_features / norms)
                sort_idx = np.argsort(pseudo)
                train_input = current_input[sort_idx]

            prev_features = run_layer(network, train_input, n_neurons,
                                       args.sim_time, dt, args.device, True)
            total = prev_features.sum(axis=1)
            print(f"    Spikes: mean={total.mean():.1f}, "
                  f"zeros={int((total==0).sum())}/{len(train_input)}")

        layer_networks.append(network)
        # Generate features for next layer
        layer_features = run_layer(network, current_input, n_neurons,
                                    args.sim_time, dt, args.device, False)
        max_val = layer_features.max()
        if max_val > 0:
            current_input = (layer_features / max_val) * args.intensity
        current_dim = n_neurons

    print(f"\n[train] Total training time: {(time.time()-t0)/60:.1f} min")

    # Encode test set through all layers
    print(f"\n[encode] Encoding {len(test_data)} test samples ...")
    test_scaled = np.clip(test_data, 0, 1) * args.intensity
    current_input = test_scaled

    for layer_idx, (net, n_neurons) in enumerate(
            zip(layer_networks, args.layer_sizes)):
        current_input_features = run_layer(
            net, current_input, n_neurons, args.sim_time, dt, args.device, False)
        if layer_idx < len(args.layer_sizes) - 1:
            max_val = current_input_features.max()
            if max_val > 0:
                current_input = (current_input_features / max_val) * args.intensity

    spike_counts = current_input_features
    codes = binarize_top_k(spike_counts, args.binarize_percent)
    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : Deep STDP (Lu & Sengupta 2024)")
    print(f"  Dataset      : MNIST")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Layers       : {[784]+args.layer_sizes}")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
