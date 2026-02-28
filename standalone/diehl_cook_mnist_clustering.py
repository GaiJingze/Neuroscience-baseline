#!/usr/bin/env python
"""
Diehl & Cook (2015) STDP-WTA Baseline — MNIST Clustering (Standalone)

Self-contained single-file script that integrates the Diehl & Cook SNN encoder
with the MNIST clustering evaluation pipeline.  Designed to be read in its
entirety by an LLM for analysis and modification.

Paper: "Unsupervised learning of digit recognition using spike-timing-dependent
        plasticity" — Diehl & Cook, Frontiers in Computational Neuroscience, 2015

Pipeline:
    1. Load MNIST (60 000 train / 10 000 test, flattened to 784-D, [0, 1])
    2. Build a 3-layer SNN:  Input (Poisson) → Excitatory (DiehlAndCookNodes
       with adaptive threshold theta, STDP learning) → Inhibitory (LIF, lateral
       inhibition)
    3. Train via STDP on training set (1 epoch)
    4. Encode test set → spike-count vectors (400-D)
    5. Binarize spike counts (top-5 %)
    6. Cluster with K-Means (k = 10) and evaluate NMI / ARI / ACC

Usage:
    python standalone/diehl_cook_mnist_clustering.py              # full 60 000 train
    python standalone/diehl_cook_mnist_clustering.py --n_train 1000  # quick test
    python standalone/diehl_cook_mnist_clustering.py --device cuda   # GPU

Requirements:
    pip install torch torchvision numpy scikit-learn scipy
    pip install bindsnet   # from GitHub: pip install git+https://github.com/BindsNET/bindsnet.git
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


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
# 2.  Network — Diehl & Cook SNN (BindsNET)
# ═══════════════════════════════════════════════════════════════════════════════

def build_network(
    n_input: int = 784,
    n_neurons: int = 400,
    dt: float = 1.0,
    nu: tuple = (1e-4, 1e-2),
    thresh: float = -52.0,
    rest: float = -65.0,
    refrac: int = 5,
) -> Network:
    """
    Build the Diehl & Cook 3-layer SNN.

    Layers
    ------
    Input       Poisson-rate-coded, with traces for STDP.
    Excitatory  DiehlAndCookNodes — LIF + adaptive threshold theta.
                theta_plus = 0.05 increments threshold after each spike,
                preventing monopolisation by a few neurons.
    Inhibitory  Standard LIF — lateral inhibition (all-to-all minus diagonal).

    Connections
    -----------
    Input → Exc   STDP (PostPre), W ∈ [0, 1], L1 weight norm = 78.4
    Exc → Inh     One-to-one, w = 22.5
    Inh → Exc     All-to-all minus diagonal, w = −17.5 (inhibition)
    """
    network = Network(dt=dt)

    # --- Layers ---
    input_layer = Input(n=n_input, traces=True)

    exc_layer = DiehlAndCookNodes(
        n=n_neurons, traces=True,
        rest=rest, reset=rest,          # reset == rest per original paper
        thresh=thresh, refrac=refrac,
        tc_decay=100.0,                 # membrane time constant (ms)
        trace_tc=5e-2,
        theta_plus=0.05,                # adaptive threshold increment
        tc_theta_decay=1e7,             # adaptive threshold decay (very slow)
    )

    inh_layer = LIFNodes(
        n=n_neurons, traces=False,
        rest=-60.0, reset=-45.0, thresh=-40.0,
        tc_decay=10.0, refrac=2, trace_tc=5e-2,
    )

    network.add_layer(input_layer, name="Input")
    network.add_layer(exc_layer, name="Excitatory")
    network.add_layer(inh_layer, name="Inhibitory")

    # --- Connections ---
    # Input → Excitatory  (STDP)
    w_ie = 0.3 * torch.rand(n_input, n_neurons)
    conn_ie = Connection(
        source=input_layer, target=exc_layer, w=w_ie,
        update_rule=PostPre, nu=nu, reduction=None,
        wmin=0.0, wmax=1.0, norm=78.4,
    )
    network.add_connection(conn_ie, source="Input", target="Excitatory")

    # Excitatory → Inhibitory  (one-to-one)
    w_ei = 22.5 * torch.eye(n_neurons)
    conn_ei = Connection(
        source=exc_layer, target=inh_layer, w=w_ei,
        wmin=0.0, wmax=22.5,
    )
    network.add_connection(conn_ei, source="Excitatory", target="Inhibitory")

    # Inhibitory → Excitatory  (lateral inhibition, negative weights)
    w_ix = -17.5 * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
    conn_ix = Connection(
        source=inh_layer, target=exc_layer, w=w_ix,
        wmin=-120.0, wmax=0.0,
    )
    network.add_connection(conn_ix, source="Inhibitory", target="Excitatory")

    # --- Monitor (spike recording for Excitatory layer) ---
    monitor = Monitor(exc_layer, state_vars=["s"])
    network.add_monitor(monitor, name="exc_monitor")

    return network


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Training — STDP
# ═══════════════════════════════════════════════════════════════════════════════

def train(network: Network, train_data: np.ndarray, train_labels: np.ndarray,
          sim_time: int = 350, intensity: float = 128.0, device: str = "cpu"):
    """
    Train the SNN with STDP on MNIST training set.

    Each image is converted to a Poisson spike train of length `sim_time` ms.
    Pixel values are scaled by `intensity` so that the Poisson rates are in
    a meaningful range (e.g. 0–128 Hz rather than 0–1 Hz).

    STDP weight updates happen automatically inside BindsNET during
    `network.run()`.

    Training logs are printed every 500 samples.
    """
    network.to(device)
    n_samples = len(train_data)

    # Scale data to [0, intensity] for Poisson encoding
    data_scaled = np.clip(train_data, 0.0, 1.0) * intensity

    print(f"\n{'='*70}")
    print(f"[train] Starting STDP training — {n_samples} samples, "
          f"sim_time={sim_time} ms, intensity={intensity}, device={device}")
    print(f"{'='*70}")

    t0 = time.time()
    last_log = t0

    for i in range(n_samples):
        # --- Log every 500 samples ---
        if (i + 1) % 500 == 0 or i == 0:
            now = time.time()
            elapsed = now - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_samples - i - 1) / speed if speed > 0 else 0

            # Gather weight statistics from Input → Excitatory connection
            w = network.connections[("Input", "Excitatory")].w
            w_np = w.detach().cpu().numpy()

            # Gather theta (adaptive threshold) statistics
            theta = network.layers["Excitatory"].theta.detach().cpu().numpy()

            print(f"  [{i+1:>6d}/{n_samples}]  "
                  f"label={train_labels[i]}  |  "
                  f"W  min={w_np.min():.4f}  mean={w_np.mean():.4f}  max={w_np.max():.4f}  |  "
                  f"theta  min={theta.min():.3f}  mean={theta.mean():.3f}  max={theta.max():.3f}  |  "
                  f"speed={speed:.1f} samp/s  ETA={remaining/60:.1f} min")
            last_log = now

        # Poisson encode (CPU — BindsNET creates internal tensors on CPU)
        image_tensor = torch.from_numpy(data_scaled[i]).float()
        encoded = poisson(datum=image_tensor, time=sim_time, dt=network.dt)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.to(device)

        # Run simulation (STDP updates happen here)
        network.run(inputs={"Input": encoded}, time=sim_time)
        network.reset_state_variables()

    elapsed = time.time() - t0
    print(f"\n[train] Training complete — {elapsed:.1f}s "
          f"({elapsed/60:.1f} min, {n_samples/elapsed:.1f} samp/s)")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Encoding — spike count extraction
# ═══════════════════════════════════════════════════════════════════════════════

def encode(network: Network, data: np.ndarray,
           sim_time: int = 350, intensity: float = 128.0,
           device: str = "cpu") -> np.ndarray:
    """
    Encode data through trained SNN → spike count vectors.

    Returns (n_samples, n_neurons) array of spike counts.
    """
    network.to(device)
    network.train(mode=False)  # disable STDP learning

    n_samples = len(data)
    n_neurons = network.layers["Excitatory"].n
    spike_counts = np.zeros((n_samples, n_neurons), dtype=np.float32)

    data_scaled = np.clip(data, 0.0, 1.0) * intensity

    print(f"\n[encode] Encoding {n_samples} samples ...")
    t0 = time.time()

    for i in range(n_samples):
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:>6d}/{n_samples}]  speed={speed:.1f} samp/s")

        image_tensor = torch.from_numpy(data_scaled[i]).float()
        encoded_spikes = poisson(datum=image_tensor, time=sim_time, dt=network.dt)
        if isinstance(encoded_spikes, torch.Tensor):
            encoded_spikes = encoded_spikes.to(device)

        network.run(inputs={"Input": encoded_spikes}, time=sim_time)

        spikes = network.monitors["exc_monitor"].get("s")
        counts = torch.sum(spikes, dim=0).cpu().numpy().flatten()
        spike_counts[i] = counts[:n_neurons]

        network.reset_state_variables()

    elapsed = time.time() - t0
    total_spikes = spike_counts.sum(axis=1)
    active_neurons = (spike_counts > 0).sum(axis=1)

    print(f"\n[encode] Done — {elapsed:.1f}s")
    print(f"  spike/sample — mean={total_spikes.mean():.1f}  "
          f"median={np.median(total_spikes):.1f}  max={total_spikes.max():.0f}  "
          f"zero_samples={int((total_spikes == 0).sum())}/{n_samples}")
    print(f"  active_neurons/sample — mean={active_neurons.mean():.1f}  "
          f"median={np.median(active_neurons):.0f}")

    return spike_counts


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Binarization — top-k%
# ═══════════════════════════════════════════════════════════════════════════════

def binarize_top_k(features: np.ndarray, percent: float = 0.05) -> np.ndarray:
    """
    Keep top k% of values per sample as 1, rest as 0.

    With n_neurons=400 and percent=0.05, k = 20 active bits per sample.
    """
    n_features = features.shape[1]
    k = max(1, int(n_features * percent))
    binary = np.zeros_like(features)
    top_k_idx = np.argsort(features, axis=1)[:, -k:]
    rows = np.arange(len(features))[:, None]
    binary[rows, top_k_idx] = 1

    sparsity = 1.0 - binary.mean()
    unique_codes = len(np.unique(binary, axis=0))
    print(f"\n[binarize] top-{percent*100:.0f}%  k={k}  "
          f"sparsity={sparsity:.3f}  unique_codes={unique_codes}/{len(features)}")
    return binary


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Clustering — K-Means + metrics
# ═══════════════════════════════════════════════════════════════════════════════

def clustering_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Clustering accuracy via Hungarian optimal assignment."""
    n_clusters = max(labels_true.max(), labels_pred.max()) + 1
    contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        contingency[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    return float(contingency[row_ind, col_ind].sum()) / len(labels_true)


def clustering_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Clustering purity (majority vote per cluster)."""
    contingency = np.zeros((labels_pred.max() + 1, labels_true.max() + 1), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        contingency[p, t] += 1
    return float(contingency.max(axis=1).sum()) / len(labels_true)


def evaluate_clustering(codes: np.ndarray, labels_true: np.ndarray,
                        n_clusters: int = 10):
    """
    Run K-Means clustering (k=n_clusters) and print all metrics.

    Metrics:
        NMI   — Normalised Mutual Information  (0–1, higher better)
        ARI   — Adjusted Rand Index            (-1–1, higher better)
        ACC   — Clustering Accuracy (Hungarian) (0–1, higher better)
        Purity                                  (0–1, higher better)
        Silhouette Score                        (-1–1, higher better)
        Davies-Bouldin Index                    (>0, lower better)
    """
    print(f"\n{'='*70}")
    print(f"[cluster] Running K-Means (k={n_clusters}) on {codes.shape} codes ...")
    print(f"{'='*70}")

    kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=0)
    labels_pred = kmeans.fit_predict(codes)

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    acc = clustering_accuracy(labels_true, labels_pred)
    purity = clustering_purity(labels_true, labels_pred)
    sil = silhouette_score(codes, labels_pred)
    db = davies_bouldin_score(codes, labels_pred)

    print(f"\n  {'Metric':<25s} {'Value':>10s}")
    print(f"  {'-'*36}")
    print(f"  {'NMI':<25s} {nmi:>10.4f}")
    print(f"  {'ARI':<25s} {ari:>10.4f}")
    print(f"  {'ACC (Hungarian)':<25s} {acc:>10.4f}")
    print(f"  {'Purity':<25s} {purity:>10.4f}")
    print(f"  {'Silhouette':<25s} {sil:>10.4f}")
    print(f"  {'Davies-Bouldin (↓)':<25s} {db:>10.4f}")
    print()

    return {
        "nmi": nmi, "ari": ari, "acc": acc,
        "purity": purity, "silhouette": sil, "davies_bouldin": db,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diehl & Cook STDP → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--data_root", type=str, default="./data")

    # Network architecture
    parser.add_argument("--n_neurons", type=int, default=400,
                        help="Number of excitatory neurons")
    parser.add_argument("--sim_time", type=int, default=350,
                        help="Simulation time per sample (ms)")
    parser.add_argument("--intensity", type=float, default=128.0,
                        help="Poisson encoding intensity (Hz scaling)")
    parser.add_argument("--nu_pre", type=float, default=1e-4,
                        help="STDP pre-synaptic learning rate")
    parser.add_argument("--nu_post", type=float, default=1e-2,
                        help="STDP post-synaptic learning rate")
    parser.add_argument("--thresh", type=float, default=-52.0,
                        help="Excitatory neuron threshold (mV)")
    parser.add_argument("--rest", type=float, default=-65.0,
                        help="Excitatory neuron resting potential (mV)")

    # Training
    parser.add_argument("--n_train", type=int, default=None,
                        help="Number of training samples (None = all 60000)")

    # Binarization
    parser.add_argument("--binarize_percent", type=float, default=0.05,
                        help="Top-k%% binarization (default: 5%%)")

    # Clustering
    parser.add_argument("--n_clusters", type=int, default=10)

    args = parser.parse_args()

    # --- Setup ---
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    # --- 1. Load MNIST ---
    train_data, train_labels, test_data, test_labels = load_mnist(args.data_root)

    if args.n_train is not None:
        train_data = train_data[:args.n_train]
        train_labels = train_labels[:args.n_train]
        print(f"[data] Subsampled to {args.n_train} training samples")

    # --- 2. Build network ---
    print(f"\n[net] Building Diehl & Cook SNN  "
          f"({train_data.shape[1]} → {args.n_neurons} neurons)")
    network = build_network(
        n_input=train_data.shape[1],
        n_neurons=args.n_neurons,
        nu=(args.nu_pre, args.nu_post),
        thresh=args.thresh,
        rest=args.rest,
    )

    # --- 3. Train ---
    train(network, train_data, train_labels,
          sim_time=args.sim_time, intensity=args.intensity,
          device=args.device)

    # --- 4. Encode test set ---
    spike_counts = encode(network, test_data,
                          sim_time=args.sim_time, intensity=args.intensity,
                          device=args.device)

    # --- 5. Binarize ---
    codes = binarize_top_k(spike_counts, percent=args.binarize_percent)

    # --- 6. Cluster & evaluate ---
    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    # --- Summary ---
    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : Diehl & Cook (2015) STDP-WTA")
    print(f"  Dataset      : MNIST")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples : {len(test_data)}")
    print(f"  Code dim     : {codes.shape[1]}")
    print(f"  Sparsity     : {1 - codes.mean():.3f}")
    print(f"  NMI          : {metrics['nmi']:.4f}")
    print(f"  ARI          : {metrics['ari']:.4f}")
    print(f"  ACC          : {metrics['acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
