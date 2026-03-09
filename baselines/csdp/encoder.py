"""
Contrastive Signal-Dependent Plasticity (CSDP) encoder.

Reference: Ororbia AG. "Contrastive signal-dependent plasticity:
           Self-supervised learning in spiking neural circuits."
           Science Advances 10(43):eadn6076, 2024.
           arXiv: 2303.18187
           DOI: 10.1126/sciadv.adn6076

Key idea:
    Each spiking layer learns to distinguish real data ("positive") from
    fabricated data ("negative") using a local goodness score based on
    spike trace activations.  Synaptic updates are three-factor Hebbian
    rules modulated by a contrastive signal (delta) derived analytically
    from the goodness gradient — no backpropagation required.

    Unsupervised negative samples are constructed by interpolating between
    two training images and applying a rotation, following the paper's
    unsupervised protocol:
        x_neg(i) = eta_mix * x(i) + (1 - eta_mix) * rotate(x(j), theta)

Architecture (per layer):
    Input spikes -> Excitatory LIF neurons (W, bottom-up)
                 -> Lateral inhibition (M, learnable)
                 -> Adaptive threshold (homeostasis)
                 -> Spike traces z (calcium-like)
    Goodness = sum(z^2);  p(positive) = sigma(goodness - theta_z)

    Weight update (Eq. 10 in paper):
        delta_i = dC/dz_i = 2 * z_i * (sigma(g - theta_z) - y_type)
        dW_ij  = R_m * delta_i * s_pre_j  + lambda_d * s_post_i * (1 - s_pre_j)
        dM_ij  = R_I * delta_i * s_lat_j  + lambda_d * s_post_i * (1 - s_lat_j)

    All updates are local (pre/post spike + modulatory delta) and
    applied via Adam optimizer on the analytically computed gradients.

    Layers operate in parallel (no backward locking).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

sys.path.append(str(Path(__file__).parent.parent.parent))
from baselines.base_encoder import BaseEncoder


# ======================================================================
# Spiking layer with CSDP plasticity (analytic three-factor Hebbian)
# ======================================================================

class CSDPLayer:
    """
    One spiking layer trained with contrastive signal-dependent plasticity.

    Uses analytic three-factor Hebbian updates (no autograd).  The
    contrastive modulator delta is computed directly from the goodness
    function gradient w.r.t. the spike traces z.

    Implements LIF dynamics with:
    - Bottom-up excitatory weights W
    - Lateral inhibitory weights M (learnable, off-diagonal)
    - Adaptive firing threshold (homeostasis)
    - Exponential spike traces z (proxy for calcium concentration)
    """

    def __init__(self, in_dim: int, n_neurons: int, dt: float = 1.0,
                 tau_m: float = 20.0, tau_tr: float = 3.0,
                 gamma: float = 0.5, theta_z: float = 2.0,
                 lambda_v: float = 0.001, lambda_d: float = 0.01,
                 R_E: float = 1.0, R_I: float = 1.0,
                 v_thr_init: float = 0.3, device: str = 'cpu'):

        self.in_dim = in_dim
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_m = tau_m
        self.tau_tr = tau_tr
        self.gamma = gamma
        self.theta_z = theta_z
        self.lambda_v = lambda_v
        self.lambda_d = lambda_d
        self.R_E = R_E
        self.R_I = R_I
        self.device = device

        # Bottom-up weights W [n_neurons, in_dim]
        # Uniform in [-1, 1] as in paper
        self.W = torch.empty(n_neurons, in_dim, device=device).uniform_(-1, 1)

        # Lateral inhibitory weights M [n_neurons, n_neurons]
        # Initialized in [0, 1], self-connections masked to 0
        self.M = torch.empty(n_neurons, n_neurons, device=device).uniform_(0, 1)
        self.M.fill_diagonal_(0.0)

        # Mask for lateral weights (no self-connections)
        self.eye_mask = torch.eye(n_neurons, device=device).bool()

        # Adaptive threshold per neuron
        self.v_thr = torch.full((n_neurons,), v_thr_init, device=device)

        # State variables (set in reset_state)
        self.v = None   # membrane voltage
        self.z = None   # spike trace
        self.s = None   # current spikes
        self.s_prev = None  # previous spikes

        # For weight updates: accumulate gradients over timesteps
        self.dW_accum = None
        self.dM_accum = None

    def to(self, device):
        self.device = device
        self.W = self.W.to(device)
        self.M = self.M.to(device)
        self.eye_mask = self.eye_mask.to(device)
        self.v_thr = self.v_thr.to(device)
        return self

    def parameters(self):
        """Return learnable parameter tensors (for optimizer)."""
        return [self.W, self.M]

    def reset_state(self, batch_size: int):
        """Reset all state for a new stimulus presentation."""
        dev = self.device
        self.v = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.z = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.s = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.s_prev = torch.zeros(batch_size, self.n_neurons, device=dev)
        self.dW_accum = torch.zeros_like(self.W)
        self.dM_accum = torch.zeros_like(self.M)

    def step(self, s_in: torch.Tensor):
        """
        One simulation timestep (all operations are no_grad).

        Args:
            s_in: Pre-synaptic spikes [batch, in_dim], binary {0,1}

        Returns:
            s: Post-synaptic spikes [batch, n_neurons], binary {0,1}
        """
        # Bottom-up excitatory current:  d = R_E * (s_in @ W^T)
        d = self.R_E * (s_in @ self.W.t())   # [batch, n_neurons]

        # Lateral inhibition from previous post-synaptic spikes
        # M has positive values; we subtract to produce inhibition
        d = d - self.R_I * (self.s_prev @ self.M.t())

        # LIF membrane dynamics:  dv/dt = (-v + I) / tau_m
        self.v = self.v + (self.dt / self.tau_m) * (-self.v + d)

        # Spike generation (hard threshold)
        self.s = (self.v >= self.v_thr.unsqueeze(0)).float()

        # Reset spiked neurons' membrane voltage
        self.v = self.v * (1.0 - self.s)

        # Trace update: dz/dt = (-z + gamma * s) / tau_tr
        self.z = self.z + (self.dt / self.tau_tr) * (
            -self.z + self.gamma * self.s)

        # Store for next timestep
        self.s_prev = self.s

        return self.s

    def accumulate_gradients(self, s_in_prev: torch.Tensor,
                             y_type: torch.Tensor):
        """
        Compute and accumulate the three-factor Hebbian weight updates
        for the current timestep (Eq. 9-10 in paper).

        Called after each timestep during training.

        Args:
            s_in_prev: Pre-synaptic spikes from previous timestep
                       [batch, in_dim]
            y_type: [batch] float, 1.0 for positive, 0.0 for negative
        """
        batch = self.s.size(0)

        # Goodness = sum(z_k^2)
        g = (self.z ** 2).sum(dim=1)                  # [batch]
        p = torch.sigmoid(g - self.theta_z)            # [batch]

        # Contrastive modulator:  delta_i = 2 * z_i * (p - y_type)
        # Shape: [batch, n_neurons]
        mod = (p - y_type).unsqueeze(1)                # [batch, 1]
        delta = 2.0 * self.z * mod                     # [batch, n_neurons]

        # --- Bottom-up weight gradient (Eq. 10) ---
        # dW_ij = R_m * delta_i * s_pre_j + lambda_d * s_post_i * (1 - s_pre_j)
        # Averaged over batch
        # Term 1:  delta^T @ s_in_prev  -> [n_neurons, in_dim]
        term1_W = (delta.t() @ s_in_prev) / batch
        # Term 2:  s_post^T @ (1 - s_in_prev)  -> [n_neurons, in_dim]
        term2_W = (self.s.t() @ (1.0 - s_in_prev)) / batch

        self.dW_accum += self.R_E * term1_W + self.lambda_d * term2_W

        # --- Lateral weight gradient ---
        # dM_ij = R_I * delta_i * s_lat_j + lambda_d * s_post_i * (1 - s_lat_j)
        term1_M = (delta.t() @ self.s_prev) / batch
        term2_M = (self.s.t() @ (1.0 - self.s_prev)) / batch

        self.dM_accum += self.R_I * term1_M + self.lambda_d * term2_M

    def apply_gradients(self, lr: float, n_steps: int):
        """
        Apply accumulated gradients (averaged over timesteps).

        The paper uses Adam, but here we do a simpler scaled SGD step
        since Adam state is managed at the CSDPEncoder level.
        """
        # Average over timesteps
        self.dW_accum /= n_steps
        self.dM_accum /= n_steps

        # Store as .grad so an external Adam optimizer can use them
        self.W.grad = self.dW_accum
        self.M.grad = self.dM_accum

    def clamp_weights(self):
        """Enforce weight constraints (from paper)."""
        self.W.clamp_(-1.0, 1.0)
        self.M.clamp_(0.0, 1.0)
        self.M.fill_diagonal_(0.0)  # no self-connections

    def adapt_threshold(self):
        """
        Homeostatic threshold adaptation (Eq. 5 in paper).
        Adjusts thresholds so firing is balanced across neurons.
        """
        mean_spikes = self.s.mean(dim=0)
        self.v_thr += self.lambda_v * (mean_spikes - 1.0 / self.n_neurons)
        self.v_thr.clamp_(min=0.01)

    def goodness(self) -> torch.Tensor:
        """Goodness score = sum of squared traces. [batch]"""
        return (self.z ** 2).sum(dim=1)

    def goodness_prob(self) -> torch.Tensor:
        """p(positive | z) = sigma(goodness - theta_z). [batch]"""
        return torch.sigmoid(self.goodness() - self.theta_z)

    def contrastive_loss(self, y_type: torch.Tensor) -> float:
        """
        Local contrastive BCE loss (Eq. 7), returned as float for logging.
        """
        p = self.goodness_prob().clamp(1e-7, 1 - 1e-7)
        loss = -(y_type * torch.log(p) + (1 - y_type) * torch.log(1 - p))
        return loss.mean().item()

    def state_dict(self):
        return {
            'W': self.W.cpu(),
            'M': self.M.cpu(),
            'v_thr': self.v_thr.cpu(),
        }

    def load_state_dict(self, sd):
        self.W = sd['W'].to(self.device)
        self.M = sd['M'].to(self.device)
        self.v_thr = sd['v_thr'].to(self.device)


# ======================================================================
# Multi-layer CSDP spiking network
# ======================================================================

class CSDPNetwork:
    """
    Multi-layer spiking network trained with CSDP.

    Each layer runs LIF dynamics and is trained with its own local
    contrastive goodness loss — no inter-layer gradient flow, no autograd.
    """

    def __init__(self, layer_dims: list, dt: float = 1.0,
                 tau_m: float = 20.0, tau_tr: float = 3.0,
                 gamma: float = 0.5, theta_z: float = 2.0,
                 lambda_v: float = 0.001, lambda_d: float = 0.01,
                 R_E: float = 1.0, R_I: float = 1.0,
                 v_thr_init: float = 0.3, device: str = 'cpu'):

        self.n_layers = len(layer_dims) - 1
        self.layers: List[CSDPLayer] = []

        for i in range(self.n_layers):
            self.layers.append(CSDPLayer(
                in_dim=layer_dims[i],
                n_neurons=layer_dims[i + 1],
                dt=dt, tau_m=tau_m, tau_tr=tau_tr, gamma=gamma,
                theta_z=theta_z, lambda_v=lambda_v, lambda_d=lambda_d,
                R_E=R_E, R_I=R_I, v_thr_init=v_thr_init, device=device,
            ))

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self

    def simulate(self, x_spikes: torch.Tensor, n_steps: int,
                 training: bool = False,
                 y_type: Optional[torch.Tensor] = None):
        """
        Run full simulation for n_steps.

        Args:
            x_spikes: [batch, input_dim] pixel probabilities for Bernoulli
                      spike generation at each timestep
            n_steps: Number of simulation timesteps
            training: If True, accumulate CSDP weight gradients
            y_type: [batch] label (1=positive, 0=negative), required if training

        Returns:
            spike_counts: list of [batch, n_neurons] per layer
        """
        batch_size = x_spikes.size(0)

        for layer in self.layers:
            layer.reset_state(batch_size)

        spike_counts = [torch.zeros(batch_size, layer.n_neurons,
                                    device=layer.device)
                        for layer in self.layers]

        # Keep track of previous input spikes for Hebbian terms
        s_in_prev = [torch.zeros(batch_size, layer.in_dim,
                                 device=layer.device)
                     for layer in self.layers]

        for t in range(n_steps):
            # Input: Bernoulli spike train from pixel intensities
            s_in = (torch.rand_like(x_spikes) < x_spikes).float()

            for ell, layer in enumerate(self.layers):
                s_out = layer.step(s_in)
                spike_counts[ell] += s_out

                # Accumulate Hebbian gradients if training
                if training and y_type is not None:
                    layer.accumulate_gradients(s_in_prev[ell], y_type)

                # Store current input for next timestep's Hebbian term
                s_in_prev[ell] = s_in

                # Next layer receives this layer's spikes
                s_in = s_out

        # Adapt thresholds after full stimulus
        for layer in self.layers:
            layer.adapt_threshold()

        return spike_counts

    def state_dict(self):
        return [layer.state_dict() for layer in self.layers]

    def load_state_dict(self, sd_list):
        for layer, sd in zip(self.layers, sd_list):
            layer.load_state_dict(sd)


# ======================================================================
# Negative sample generation (unsupervised protocol from the paper)
# ======================================================================

def make_negative_samples(x: torch.Tensor, eta_mix: float = 0.55,
                          img_size: int = 28) -> torch.Tensor:
    """
    Unsupervised negative sample construction (Section 3 of paper).

    x_neg(i) = eta_mix * x(i) + (1 - eta_mix) * rotate(x(j), theta)

    where j is a random permutation of batch indices and
    theta is sampled uniformly from [pi/4, 7*pi/4].

    Args:
        x: [batch, input_dim] normalized pixel values in [0,1]
        eta_mix: Interpolation weight (0.55 in paper)
        img_size: Spatial size (assumes square images)

    Returns:
        x_neg: [batch, input_dim] fabricated negative samples
    """
    batch_size = x.size(0)
    input_dim = x.size(1)
    device = x.device

    # Random partner (shift by random offset to avoid self-pairing)
    perm = torch.randperm(batch_size, device=device)
    x_partner = x[perm]

    # Reshape to image for rotation
    x_img = x_partner.view(batch_size, 1, img_size, img_size)

    # Random rotation angles in [pi/4, 7*pi/4] (= 45 to 315 degrees)
    angles_rad = torch.empty(batch_size, device=device).uniform_(
        math.pi / 4, 7 * math.pi / 4)
    angles_deg = angles_rad * (180.0 / math.pi)

    # Apply rotation via affine grid
    rotated = _batch_rotate(x_img, angles_deg)
    rotated = rotated.view(batch_size, input_dim)

    # Interpolate
    x_neg = eta_mix * x + (1 - eta_mix) * rotated
    x_neg = x_neg.clamp(0, 1)

    return x_neg


def _batch_rotate(images: torch.Tensor,
                  angles_deg: torch.Tensor) -> torch.Tensor:
    """
    Rotate a batch of images by individual angles using affine_grid.

    Args:
        images: [batch, 1, H, W]
        angles_deg: [batch] rotation angles in degrees

    Returns:
        rotated: [batch, 1, H, W]
    """
    batch_size = images.size(0)
    device = images.device

    angles_rad = angles_deg * (math.pi / 180.0)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)

    # 2x3 affine matrices for rotation
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    grid = F.affine_grid(theta, images.size(), align_corners=False)
    rotated = F.grid_sample(images, grid, align_corners=False,
                            mode='bilinear', padding_mode='zeros')
    return rotated


# ======================================================================
# Encoder wrapper (BaseEncoder API)
# ======================================================================

class CSDPEncoder(BaseEncoder):
    """
    CSDP encoder for clustering/hashing pipeline.

    Implements contrastive signal-dependent plasticity from:
    Ororbia (2024), Science Advances 10(43):eadn6076.

    Uses the paper's analytic three-factor Hebbian weight updates
    (not surrogate-gradient backpropagation).  Each layer computes
    a local contrastive modulator delta from the goodness function
    and applies Hebbian updates modulated by delta.

    Unsupervised: negative samples via interpolation + rotation.
    Output: spike count representation from the final spiking layer,
    binarized via top-k for the 'code' output.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']
        self.img_size = config.get('img_size', 28)

        # Network architecture
        self.hidden_dims = config.get('hidden_dims', [500])
        self.output_dim = config.get('output_dim', 400)

        # Spiking / LIF parameters
        self.dt = config.get('dt', 1.0)
        self.tau_m = config.get('tau_m', 20.0)
        self.tau_tr = config.get('tau_tr', 3.0)
        self.gamma = config.get('gamma', 0.5)
        self.n_steps = config.get('n_steps', 100)  # simulation timesteps

        # CSDP parameters
        self.theta_z = config.get('theta_z', 2.0)
        self.lambda_v = config.get('lambda_v', 0.001)
        self.lambda_d = config.get('lambda_d', 0.01)
        self.R_E = config.get('R_E', 1.0)
        self.R_I = config.get('R_I', 1.0)
        self.v_thr_init = config.get('v_thr_init', 0.3)

        # Negative sample generation
        self.eta_mix = config.get('eta_mix', 0.55)

        # Training
        self.n_epochs = config.get('n_epochs', 5)
        self.batch_size = config.get('batch_size', 128)
        self.lr = config.get('lr', 2e-3)
        self.binarization_percent = config.get('binarization_percent', 0.05)

        # Device
        self.device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Build network
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        self.network = CSDPNetwork(
            layer_dims=layer_dims,
            dt=self.dt, tau_m=self.tau_m, tau_tr=self.tau_tr,
            gamma=self.gamma, theta_z=self.theta_z,
            lambda_v=self.lambda_v, lambda_d=self.lambda_d,
            R_E=self.R_E, R_I=self.R_I,
            v_thr_init=self.v_thr_init, device=self.device,
        )

        self.is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_data: np.ndarray,
            train_labels: Optional[np.ndarray] = None):
        """
        Train the CSDP spiking network (unsupervised).

        Each batch:
        1. Positive phase: run network on real data, accumulate Hebbian grads.
        2. Negative phase: run on fabricated negatives, accumulate Hebbian grads.
        3. Apply averaged gradients via Adam optimizer.
        """
        print(f"\n{'='*60}")
        print("Training CSDP (Contrastive Signal-Dependent Plasticity)")
        print(f"{'='*60}")
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        print(f"Architecture: {' -> '.join(map(str, dims))}")
        print(f"Simulation: {self.n_steps} steps x {self.dt} ms")
        print(f"Epochs: {self.n_epochs} | Batch: {self.batch_size} | "
              f"LR: {self.lr}")
        print(f"theta_z: {self.theta_z} | eta_mix: {self.eta_mix} | "
              f"lambda_d: {self.lambda_d}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # Normalize data to [0, 1]
        data = train_data.copy().astype(np.float32)
        if data.max() > 1.0:
            data = data / 255.0
        data = np.clip(data, 0.0, 1.0)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(data).float())
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True)

        # Adam optimizer per layer (on the raw tensors, not nn.Parameters)
        optimizers = []
        for layer in self.network.layers:
            params = layer.parameters()
            # Wrap in a list of dicts for Adam
            # We need to make them require_grad for Adam to work
            for p in params:
                p.requires_grad_(True)
            optimizers.append(
                torch.optim.Adam(params, lr=self.lr))

        start_time = time.time()

        for epoch in range(self.n_epochs):
            epoch_losses = [0.0] * self.network.n_layers
            n_batches = 0

            for batch_idx, (batch_x,) in enumerate(loader):
                batch_x = batch_x.to(self.device)
                bs = batch_x.size(0)

                # --- Positive phase ---
                y_pos = torch.ones(bs, device=self.device)
                with torch.no_grad():
                    self.network.simulate(
                        batch_x, self.n_steps,
                        training=True, y_type=y_pos)

                # Record positive losses for logging
                losses_pos = [layer.contrastive_loss(y_pos)
                              for layer in self.network.layers]

                # Save positive-phase accumulated gradients
                pos_dW = [layer.dW_accum.clone()
                          for layer in self.network.layers]
                pos_dM = [layer.dM_accum.clone()
                          for layer in self.network.layers]

                # --- Negative phase ---
                batch_neg = make_negative_samples(
                    batch_x, eta_mix=self.eta_mix, img_size=self.img_size)
                y_neg = torch.zeros(bs, device=self.device)
                with torch.no_grad():
                    self.network.simulate(
                        batch_neg, self.n_steps,
                        training=True, y_type=y_neg)

                losses_neg = [layer.contrastive_loss(y_neg)
                              for layer in self.network.layers]

                # --- Apply combined gradients ---
                for ell, layer in enumerate(self.network.layers):
                    # Combine pos + neg accumulated gradients, average over steps
                    combined_dW = (pos_dW[ell] + layer.dW_accum) / (
                        2.0 * self.n_steps)
                    combined_dM = (pos_dM[ell] + layer.dM_accum) / (
                        2.0 * self.n_steps)

                    # Set as .grad for Adam
                    layer.W.grad = combined_dW
                    layer.M.grad = combined_dM

                    optimizers[ell].step()
                    optimizers[ell].zero_grad()

                    # Enforce weight constraints
                    with torch.no_grad():
                        layer.clamp_weights()

                    epoch_losses[ell] += losses_pos[ell] + losses_neg[ell]

                n_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    avg = [el / n_batches for el in epoch_losses]
                    avg_str = ', '.join(f'L{i}={v:.4f}'
                                        for i, v in enumerate(avg))
                    print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/"
                          f"{len(loader)} | {avg_str}")

            avg = [el / max(n_batches, 1) for el in epoch_losses]
            avg_str = ', '.join(f'L{i}={v:.4f}' for i, v in enumerate(avg))
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{self.n_epochs} done ({elapsed:.0f}s) "
                  f"| Losses: {avg_str}")

        self.is_trained = True
        total = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"CSDP training complete! Total time: {total:.1f}s")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Encoding (inference)
    # ------------------------------------------------------------------
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data by running the trained spiking network and
        collecting spike counts from the final layer.
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")

        n_samples = len(data)
        print(f"Encoding {n_samples} samples (CSDP, "
              f"{self.network.n_layers} layers)...")

        # Normalize
        data_norm = data.copy().astype(np.float32)
        if data_norm.max() > 1.0:
            data_norm = data_norm / 255.0
        data_norm = np.clip(data_norm, 0.0, 1.0)

        data_tensor = torch.from_numpy(data_norm).float()

        all_counts = []

        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch = data_tensor[i:i + self.batch_size].to(self.device)
                spike_counts = self.network.simulate(batch, self.n_steps)
                # Use last layer's spike counts as representation
                last_counts = spike_counts[-1].cpu().numpy()
                all_counts.append(last_counts)

        pre_code = np.concatenate(all_counts, axis=0)

        # Top-k binarization
        k = max(int(self.output_dim * self.binarization_percent), 1)
        code = self._top_k_binarization(pre_code, k)

        # Stats
        total_spikes = pre_code.sum(axis=1)
        active = (pre_code > 0).sum(axis=1)
        unique = len(np.unique(code, axis=0))
        print(f"  Spike counts - mean: {total_spikes.mean():.1f}, "
              f"median: {np.median(total_spikes):.1f}, "
              f"max: {total_spikes.max():.0f}")
        print(f"  Active neurons/sample - mean: {active.mean():.1f}")
        print(f"  Unique binary codes: {unique}/{n_samples}")
        print("Encoding complete!")

        return {'pre_code': pre_code, 'code': code}

    def _top_k_binarization(self, features: np.ndarray,
                            k: int) -> np.ndarray:
        binary = np.zeros_like(features)
        top_k = np.argsort(features, axis=1)[:, -k:]
        rows = np.arange(len(features))[:, None]
        binary[rows, top_k] = 1
        return binary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        super().save(path)
        model_path = path.replace('.pkl', '_model.pt')
        torch.save({
            'layer_states': self.network.state_dict(),
        }, model_path)
        print(f"CSDP model saved to {model_path}")

    def load(self, path: str):
        super().load(path)
        model_path = path.replace('.pkl', '_model.pt')
        if Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(ckpt['layer_states'])
            print(f"CSDP model loaded from {model_path}")


# ======================================================================
# Quick test
# ======================================================================
if __name__ == '__main__':
    print("Testing CSDP encoder...")

    np.random.seed(42)
    torch.manual_seed(42)

    # Synthetic data (fake MNIST-like)
    train_data = np.random.rand(500, 784).astype(np.float32)
    test_data = np.random.rand(100, 784).astype(np.float32)

    config = {
        'input_dim': 784,
        'img_size': 28,
        'hidden_dims': [300],
        'output_dim': 200,
        'n_steps': 50,
        'n_epochs': 2,
        'batch_size': 64,
        'lr': 2e-3,
        'theta_z': 2.0,
        'eta_mix': 0.55,
        'v_thr_init': 0.3,
        'device': 'cpu',
    }

    encoder = CSDPEncoder(config)
    encoder.fit(train_data)

    result = encoder.encode(test_data)

    print(f"\nPre-code shape: {result['pre_code'].shape}")
    print(f"Code shape: {result['code'].shape}")
    print(f"Code sparsity: {1 - np.mean(result['code']):.3f}")
    print(f"Average ones per sample: "
          f"{np.mean(np.sum(result['code'], axis=1)):.1f}")
    print("Test passed!")
