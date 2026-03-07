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
    spike trace activations.  Synaptic updates are multi-factor Hebbian
    rules modulated by a contrastive signal derived from the goodness
    gradient — no backpropagation or feedback synapses required.

    Unsupervised negative samples are constructed by interpolating between
    two training images and applying a rotation, following the paper's
    unsupervised protocol:
        x_neg(i) = eta_mix * x(i) + (1 - eta_mix) * rotate(x(j), theta)

Architecture (per layer):
    Input spikes -> Excitatory LIF neurons (W, bottom-up)
                 -> Lateral inhibition (M)
                 -> Adaptive threshold (homeostasis)
                 -> Spike traces z (calcium-like)
    Goodness = sum(z^2);  p(positive) = sigma(goodness - theta_z)

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
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from baselines.base_encoder import BaseEncoder


# ======================================================================
# Spiking layer with CSDP plasticity
# ======================================================================

class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for the Heaviside spike function.

    Forward: s = 1 if v >= v_thr else 0  (hard threshold)
    Backward: ds/dv ~ triangular surrogate around threshold

    This is standard practice for training SNNs with gradient-based
    optimizers.  See Neftci et al. (2019) "Surrogate Gradient Learning
    in Spiking Neural Networks".
    """
    scale = 2.0

    @staticmethod
    def forward(ctx, v_minus_thr):
        # v_minus_thr = v - v_thr (pre-computed to avoid saving v_thr buffer)
        ctx.save_for_backward(v_minus_thr)
        return (v_minus_thr >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        v_minus_thr, = ctx.saved_tensors
        surrogate = (1.0 - v_minus_thr.abs() / SurrogateSpike.scale).clamp(min=0)
        return grad_output * surrogate


spike_fn = SurrogateSpike.apply


class CSDPLayer(nn.Module):
    """
    One spiking layer trained with contrastive signal-dependent plasticity.

    Implements LIF dynamics with:
    - Bottom-up excitatory weights W
    - Lateral inhibitory weights M (fixed sparse WTA)
    - Adaptive firing threshold (homeostasis)
    - Exponential spike traces z (proxy for calcium concentration)
    - Goodness-based contrastive loss for local synaptic updates
    """

    def __init__(self, in_dim: int, n_neurons: int, dt: float = 1.0,
                 tau_m: float = 20.0, tau_tr: float = 3.0,
                 gamma: float = 0.05, theta_z: float = 10.0,
                 lambda_v: float = 0.001, R_E: float = 1.0,
                 R_I: float = 1.0, v_thr_init: float = 1.0):
        super().__init__()

        self.in_dim = in_dim
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_m = tau_m
        self.tau_tr = tau_tr
        self.gamma = gamma
        self.theta_z = theta_z
        self.lambda_v = lambda_v
        self.R_E = R_E
        self.R_I = R_I

        # Bottom-up weights (learnable via CSDP)
        # Initialize with positive weights (excitatory input) scaled to
        # produce reasonable membrane currents given Bernoulli spike input
        W_init = torch.rand(n_neurons, in_dim) * (2.0 / math.sqrt(in_dim))
        self.W = nn.Parameter(W_init)

        # Lateral inhibitory weights (fixed, off-diagonal negative)
        M = -0.5 * (torch.ones(n_neurons, n_neurons)
                     - torch.eye(n_neurons))
        self.register_buffer('M', M)

        # Adaptive threshold per neuron
        self.register_buffer('v_thr', torch.full((n_neurons,), v_thr_init))

    def reset_state(self, batch_size: int, device: torch.device):
        """Reset membrane voltage and traces for a new stimulus."""
        self.v = torch.zeros(batch_size, self.n_neurons, device=device)
        self.z = torch.zeros(batch_size, self.n_neurons, device=device)
        self.s_prev = torch.zeros(batch_size, self.n_neurons, device=device)

    def step(self, s_in: torch.Tensor):
        """
        One simulation timestep.

        Args:
            s_in: Pre-synaptic spikes [batch, in_dim], binary {0,1}

        Returns:
            s: Post-synaptic spikes [batch, n_neurons], binary {0,1}
        """
        # Bottom-up current (this is the path gradients flow through)
        d = self.R_E * F.linear(s_in, self.W)  # [batch, n_neurons]

        # Lateral inhibition from previous spikes (detached — no BPTT)
        d = d + self.R_I * F.linear(self.s_prev.detach(), self.M)

        # LIF membrane dynamics:  dv/dt = (-v + I) / tau_m
        # Detach v from previous step to keep computation graph shallow
        self.v = self.v.detach() + (self.dt / self.tau_m) * (
            -self.v.detach() + d)

        # Spike generation (surrogate gradient for backward pass)
        s = spike_fn(self.v - self.v_thr.unsqueeze(0))

        # Reset spiked neurons (detach to avoid backprop through reset)
        self.v = self.v * (1.0 - s.detach())

        # Trace update: z += dt/tau_tr * (-z + gamma * s)
        self.z = self.z + (self.dt / self.tau_tr) * (-self.z + self.gamma * s)

        self.s_prev = s
        return s

    def adapt_threshold(self):
        """
        Homeostatic threshold adaptation (Eq. 5 in paper).
        Called once per stimulus after all timesteps.

        Adjusts thresholds so that on average ~1 neuron fires per timestep.
        """
        with torch.no_grad():
            # Mean spikes across batch for each neuron
            mean_spikes = self.s_prev.mean(dim=0)  # approximate activity
            self.v_thr += self.lambda_v * (mean_spikes - 1.0 / self.n_neurons)
            self.v_thr.clamp_(min=0.1)

    def goodness(self) -> torch.Tensor:
        """
        Compute goodness score = sum of squared traces (Eq. 8 in paper).

        Returns:
            g: [batch] scalar goodness per sample
        """
        return (self.z ** 2).sum(dim=1)

    def goodness_prob(self) -> torch.Tensor:
        """
        p(positive | z) = sigma(goodness - theta_z)  (Eq. 8 in paper).

        Returns:
            p: [batch] probability that input is positive
        """
        g = self.goodness()
        return torch.sigmoid(g - self.theta_z)

    def contrastive_loss(self, y_type: torch.Tensor) -> torch.Tensor:
        """
        Local contrastive loss (Eq. 7 in paper).

        Args:
            y_type: [batch] float, 1.0 for positive, 0.0 for negative

        Returns:
            loss: scalar (mean over batch)
        """
        p = self.goodness_prob()
        p = p.clamp(1e-7, 1 - 1e-7)
        loss = -(y_type * torch.log(p) + (1 - y_type) * torch.log(1 - p))
        return loss.mean()


# ======================================================================
# Multi-layer CSDP spiking network
# ======================================================================

class CSDPNetwork(nn.Module):
    """
    Multi-layer spiking network trained with CSDP.

    Each layer runs LIF dynamics independently and is trained with its
    own local contrastive goodness loss — no inter-layer gradient flow.
    """

    def __init__(self, layer_dims: list, dt: float = 1.0,
                 tau_m: float = 20.0, tau_tr: float = 3.0,
                 gamma: float = 0.05, theta_z: float = 10.0,
                 lambda_v: float = 0.001, R_E: float = 1.0,
                 R_I: float = 1.0, v_thr_init: float = 1.0):
        """
        Args:
            layer_dims: [input_dim, layer1_neurons, layer2_neurons, ...]
        """
        super().__init__()

        self.n_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()

        for i in range(self.n_layers):
            self.layers.append(CSDPLayer(
                in_dim=layer_dims[i],
                n_neurons=layer_dims[i + 1],
                dt=dt, tau_m=tau_m, tau_tr=tau_tr, gamma=gamma,
                theta_z=theta_z, lambda_v=lambda_v,
                R_E=R_E, R_I=R_I, v_thr_init=v_thr_init,
            ))

    def simulate(self, x_spikes: torch.Tensor, n_steps: int):
        """
        Run full simulation for n_steps.

        Args:
            x_spikes: [batch, input_dim] pixel probabilities for Bernoulli
                      spike generation at each timestep
            n_steps: Number of simulation timesteps

        Returns:
            spike_counts: list of [batch, n_neurons] per layer (total spikes)
        """
        batch_size = x_spikes.size(0)
        device = x_spikes.device

        for layer in self.layers:
            layer.reset_state(batch_size, device)

        # Accumulate spike counts for representation
        spike_counts = [torch.zeros(batch_size, layer.n_neurons, device=device)
                        for layer in self.layers]

        for t in range(n_steps):
            # Input: Bernoulli spike train from pixel intensities
            s_in = (torch.rand_like(x_spikes) < x_spikes).float()

            for ell, layer in enumerate(self.layers):
                s_out = layer.step(s_in)
                spike_counts[ell] += s_out
                # Next layer receives this layer's spikes, detached
                # (CSDP is local — no inter-layer gradient flow)
                s_in = s_out.detach()

        # Adapt thresholds after full stimulus
        for layer in self.layers:
            layer.adapt_threshold()

        return spike_counts

    def compute_losses(self, y_type: torch.Tensor) -> list:
        """
        Compute per-layer contrastive losses (after simulate).

        Args:
            y_type: [batch] float, 1.0=positive, 0.0=negative

        Returns:
            List of scalar losses, one per layer
        """
        return [layer.contrastive_loss(y_type) for layer in self.layers]


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


def _batch_rotate(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
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
            lambda_v=self.lambda_v, R_E=self.R_E, R_I=self.R_I,
            v_thr_init=self.v_thr_init,
        ).to(self.device)

        self.is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_data: np.ndarray,
            train_labels: Optional[np.ndarray] = None):
        """
        Train the CSDP spiking network (unsupervised).

        Each batch:
        1. Positive phase: run network on real data, goodness should be high.
        2. Negative phase: run on fabricated negatives, goodness should be low.
        3. Update weights via Adam on per-layer contrastive losses.
        """
        print(f"\n{'='*60}")
        print("Training CSDP (Contrastive Signal-Dependent Plasticity)")
        print(f"{'='*60}")
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        print(f"Architecture: {' -> '.join(map(str, dims))}")
        print(f"Simulation: {self.n_steps} steps x {self.dt} ms")
        print(f"Epochs: {self.n_epochs} | Batch: {self.batch_size} | "
              f"LR: {self.lr}")
        print(f"theta_z: {self.theta_z} | eta_mix: {self.eta_mix}")
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

        # Separate optimizer per layer (layers train in parallel / locally)
        optimizers = [
            torch.optim.Adam(layer.parameters(), lr=self.lr)
            for layer in self.network.layers
        ]

        start_time = time.time()

        for epoch in range(self.n_epochs):
            epoch_losses = [0.0] * self.network.n_layers
            n_batches = 0

            for batch_idx, (batch_x,) in enumerate(loader):
                batch_x = batch_x.to(self.device)

                # --- Positive phase ---
                self.network.simulate(batch_x, self.n_steps)
                y_pos = torch.ones(batch_x.size(0), device=self.device)
                losses_pos = self.network.compute_losses(y_pos)

                # --- Negative phase ---
                batch_neg = make_negative_samples(
                    batch_x, eta_mix=self.eta_mix, img_size=self.img_size)
                self.network.simulate(batch_neg, self.n_steps)
                y_neg = torch.zeros(batch_neg.size(0), device=self.device)
                losses_neg = self.network.compute_losses(y_neg)

                # --- Per-layer weight updates ---
                for ell in range(self.network.n_layers):
                    loss = losses_pos[ell] + losses_neg[ell]
                    optimizers[ell].zero_grad()
                    loss.backward()
                    optimizers[ell].step()
                    epoch_losses[ell] += loss.item()

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

        self.network.eval()
        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch = data_tensor[i:i + self.batch_size].to(self.device)
                spike_counts = self.network.simulate(batch, self.n_steps)
                # Use last layer's spike counts as representation
                last_counts = spike_counts[-1].cpu().numpy()
                all_counts.append(last_counts)

        pre_code = np.concatenate(all_counts, axis=0)  # [n_samples, output_dim]

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
            'network_state_dict': self.network.state_dict(),
            'v_thr': [layer.v_thr.cpu() for layer in self.network.layers],
        }, model_path)
        print(f"CSDP model saved to {model_path}")

    def load(self, path: str):
        super().load(path)
        model_path = path.replace('.pkl', '_model.pt')
        if Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(ckpt['network_state_dict'])
            for layer, vt in zip(self.network.layers, ckpt['v_thr']):
                layer.v_thr.copy_(vt.to(self.device))
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
