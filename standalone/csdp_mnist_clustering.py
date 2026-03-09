#!/usr/bin/env python
"""
CSDP Baseline — MNIST Clustering (Standalone)

Self-contained single-file script: CSDP spiking encoder + MNIST clustering.

Paper: "Contrastive signal-dependent plasticity: Self-supervised learning
        in spiking neural circuits"
       Ororbia, Science Advances 10(43):eadn6076, 2024

Algorithm:
    Each spiking layer learns to distinguish positive (real) from negative
    (fabricated) data using a local goodness score:
        goodness = sum(z^2),  p(positive) = σ(goodness − θ_z)
    Weight update (analytic three-factor Hebbian):
        δ_i = 2·z_i·(σ(g−θ_z) − y_type)
        ΔW_ij = R_E·δ_i·s_pre_j + λ_d·s_post_i·(1−s_pre_j)
    Negatives: x_neg = η·x + (1−η)·rotate(x_partner, θ)

Pipeline:
    1. Load MNIST
    2. Train multi-layer spiking network with CSDP
    3. Encode test set → spike counts → top-k binary code
    4. Cluster with K-Means (k=10) and evaluate

Usage:
    python standalone/csdp_mnist_clustering.py
    python standalone/csdp_mnist_clustering.py --hidden_dims 500 --output_dim 400

Requirements:
    pip install torch torchvision numpy scikit-learn scipy
"""

import argparse
import math
import random
import time

import numpy as np
import torch
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
# CSDP Spiking Layer
# ═══════════════════════════════════════════════════════════════════════════════

class CSDPLayer:
    """One spiking layer with contrastive signal-dependent plasticity."""

    def __init__(self, in_dim, n_neurons, dt=1.0, tau_m=20.0, tau_tr=3.0,
                 gamma=0.5, theta_z=2.0, lambda_v=0.001, lambda_d=0.01,
                 R_E=1.0, R_I=1.0, v_thr_init=0.3, device='cpu'):
        self.n_neurons = n_neurons
        self.in_dim = in_dim
        self.dt, self.tau_m, self.tau_tr = dt, tau_m, tau_tr
        self.gamma, self.theta_z = gamma, theta_z
        self.lambda_v, self.lambda_d = lambda_v, lambda_d
        self.R_E, self.R_I = R_E, R_I
        self.device = device

        self.W = torch.empty(n_neurons, in_dim, device=device).uniform_(-1, 1)
        self.M = torch.empty(n_neurons, n_neurons, device=device).uniform_(0, 1)
        self.M.fill_diagonal_(0.0)
        self.v_thr = torch.full((n_neurons,), v_thr_init, device=device)
        self.v = self.z = self.s = self.s_prev = None
        self.dW_accum = self.dM_accum = None

    def to(self, device):
        self.device = device
        self.W = self.W.to(device)
        self.M = self.M.to(device)
        self.v_thr = self.v_thr.to(device)
        return self

    def parameters(self):
        return [self.W, self.M]

    def reset_state(self, batch_size):
        dev = self.device
        self.v = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.z = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.s = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.s_prev = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.dW_accum = torch.zeros_like(self.W)
        self.dM_accum = torch.zeros_like(self.M)

    def step(self, s_in):
        d = self.R_E * (s_in @ self.W.t()) - self.R_I * (self.s_prev @ self.M.t())
        self.v = self.v + (self.dt / self.tau_m) * (-self.v + d)
        self.s = (self.v >= self.v_thr.unsqueeze(0)).float()
        self.v = self.v * (1.0 - self.s)
        self.z = self.z + (self.dt / self.tau_tr) * (-self.z + self.gamma * self.s)
        self.s_prev = self.s
        return self.s

    def accumulate_gradients(self, s_in_prev, y_type):
        batch = self.s.size(0)
        g = (self.z ** 2).sum(dim=1)
        p = torch.sigmoid(g - self.theta_z)
        mod = (p - y_type).unsqueeze(1)
        delta = 2.0 * self.z * mod

        term1_W = (delta.t() @ s_in_prev) / batch
        term2_W = (self.s.t() @ (1.0 - s_in_prev)) / batch
        self.dW_accum += self.R_E * term1_W + self.lambda_d * term2_W

        term1_M = (delta.t() @ self.s_prev) / batch
        term2_M = (self.s.t() @ (1.0 - self.s_prev)) / batch
        self.dM_accum += self.R_I * term1_M + self.lambda_d * term2_M

    def clamp_weights(self):
        self.W.clamp_(-1.0, 1.0)
        self.M.clamp_(0.0, 1.0)
        self.M.fill_diagonal_(0.0)

    def adapt_threshold(self):
        mean_spikes = self.s.mean(dim=0)
        self.v_thr += self.lambda_v * (mean_spikes - 1.0 / self.n_neurons)
        self.v_thr.clamp_(min=0.01)

    def contrastive_loss(self, y_type):
        g = (self.z ** 2).sum(dim=1)
        p = torch.sigmoid(g - self.theta_z).clamp(1e-7, 1 - 1e-7)
        loss = -(y_type * torch.log(p) + (1 - y_type) * torch.log(1 - p))
        return loss.mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# CSDP Network + Negative sample generation
# ═══════════════════════════════════════════════════════════════════════════════

class CSDPNetwork:
    def __init__(self, layer_dims, **kwargs):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(CSDPLayer(layer_dims[i], layer_dims[i+1], **kwargs))

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self

    def simulate(self, x_spikes, n_steps, training=False, y_type=None):
        batch_size = x_spikes.size(0)
        for layer in self.layers:
            layer.reset_state(batch_size)

        spike_counts = [torch.zeros(batch_size, l.n_neurons, device=l.device)
                        for l in self.layers]
        s_in_prev = [torch.zeros(batch_size, l.in_dim, device=l.device)
                     for l in self.layers]

        for t in range(n_steps):
            s_in = (torch.rand_like(x_spikes) < x_spikes).float()
            for ell, layer in enumerate(self.layers):
                s_out = layer.step(s_in)
                spike_counts[ell] += s_out
                if training and y_type is not None:
                    layer.accumulate_gradients(s_in_prev[ell], y_type)
                s_in_prev[ell] = s_in
                s_in = s_out

        for layer in self.layers:
            layer.adapt_threshold()

        return spike_counts


def _batch_rotate(images, angles_deg):
    batch_size = images.size(0)
    device = images.device
    angles_rad = angles_deg * (math.pi / 180.0)
    cos_a, sin_a = torch.cos(angles_rad), torch.sin(angles_rad)
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, mode='bilinear',
                         padding_mode='zeros')


def make_negatives(x, eta_mix=0.55, img_size=28):
    batch_size, input_dim = x.size()
    perm = torch.randperm(batch_size, device=x.device)
    x_img = x[perm].view(batch_size, 1, img_size, img_size)
    angles = torch.empty(batch_size, device=x.device).uniform_(45, 315)
    rotated = _batch_rotate(x_img, angles).view(batch_size, input_dim)
    return (eta_mix * x + (1 - eta_mix) * rotated).clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Training + Encoding
# ═══════════════════════════════════════════════════════════════════════════════

def train_csdp(train_data, layer_dims, n_steps, n_epochs, batch_size, lr,
               theta_z, eta_mix, lambda_d, v_thr_init, device):
    network = CSDPNetwork(
        layer_dims, dt=1.0, tau_m=20.0, tau_tr=3.0, gamma=0.5,
        theta_z=theta_z, lambda_v=0.001, lambda_d=lambda_d,
        R_E=1.0, R_I=1.0, v_thr_init=v_thr_init, device=device)
    network.to(device)

    print(f"\n[csdp] Training: {' → '.join(map(str, layer_dims))}")
    print(f"  n_steps={n_steps}, epochs={n_epochs}, lr={lr}, device={device}")

    data = np.clip(train_data, 0.0, 1.0)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, drop_last=True)

    optimizers = []
    for layer in network.layers:
        params = layer.parameters()
        for p in params:
            p.requires_grad_(True)
        optimizers.append(torch.optim.Adam(params, lr=lr))

    for epoch in range(n_epochs):
        epoch_losses = [0.0] * len(network.layers)
        n_batches = 0

        for batch_idx, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)
            bs = batch_x.size(0)

            # Positive phase
            y_pos = torch.ones(bs, device=device)
            with torch.no_grad():
                network.simulate(batch_x, n_steps, training=True, y_type=y_pos)
            pos_dW = [l.dW_accum.clone() for l in network.layers]
            pos_dM = [l.dM_accum.clone() for l in network.layers]

            # Negative phase
            batch_neg = make_negatives(batch_x, eta_mix)
            y_neg = torch.zeros(bs, device=device)
            with torch.no_grad():
                network.simulate(batch_neg, n_steps, training=True, y_type=y_neg)

            # Apply gradients
            for ell, layer in enumerate(network.layers):
                layer.W.grad = (pos_dW[ell] + layer.dW_accum) / (2.0 * n_steps)
                layer.M.grad = (pos_dM[ell] + layer.dM_accum) / (2.0 * n_steps)
                optimizers[ell].step()
                optimizers[ell].zero_grad()
                with torch.no_grad():
                    layer.clamp_weights()
                epoch_losses[ell] += layer.contrastive_loss(y_pos)
            n_batches += 1

        avg_str = ', '.join(f'L{i}={v/n_batches:.4f}'
                            for i, v in enumerate(epoch_losses))
        print(f"  Epoch {epoch+1}/{n_epochs} | {avg_str}")

    print("[csdp] Training complete!")
    return network


def csdp_encode(data, network, n_steps, output_dim, batch_size, device,
                binarize_percent=0.05):
    data = np.clip(data, 0.0, 1.0)
    data_tensor = torch.from_numpy(data.astype(np.float32))

    all_counts = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data_tensor[i:i + batch_size].to(device)
            spike_counts = network.simulate(batch, n_steps)
            all_counts.append(spike_counts[-1].cpu().numpy())

    pre_code = np.concatenate(all_counts, axis=0)
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
        description="CSDP → MNIST Clustering (standalone)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[500])
    parser.add_argument("--output_dim", type=int, default=400)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--theta_z", type=float, default=2.0)
    parser.add_argument("--eta_mix", type=float, default=0.55)
    parser.add_argument("--lambda_d", type=float, default=0.01)
    parser.add_argument("--v_thr_init", type=float, default=0.3)
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
    if args.n_test is not None:
        test_data = test_data[:args.n_test]
        test_labels = test_labels[:args.n_test]

    input_dim = train_data.shape[1]
    layer_dims = [input_dim] + args.hidden_dims + [args.output_dim]

    t0 = time.time()
    network = train_csdp(
        train_data, layer_dims, args.n_steps, args.n_epochs, args.batch_size,
        args.lr, args.theta_z, args.eta_mix, args.lambda_d,
        args.v_thr_init, args.device)
    print(f"[train] Training time: {time.time()-t0:.1f}s")

    t0 = time.time()
    pre_code, codes = csdp_encode(
        test_data, network, args.n_steps, args.output_dim,
        args.batch_size, args.device, args.binarize_percent)
    print(f"[encode] Encoded {len(test_data)} samples ({time.time()-t0:.1f}s)")
    sparsity = 1 - codes.mean()
    print(f"[encode] Code shape: {codes.shape}, sparsity: {sparsity:.3f}")

    metrics = evaluate_clustering(codes, test_labels, n_clusters=args.n_clusters)

    print("="*70)
    print("[done] Experiment finished.")
    print(f"  Baseline     : CSDP (Ororbia 2024)")
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
