"""
Locally Connected SNN (LC-SNN) encoder.

Reference: "Locally Connected Spiking Neural Networks for Unsupervised
            Feature Learning"
           Saunders, Patel, Hazan, Siegelmann & Kozma,
           Neural Networks 2019

Key difference from Diehl & Cook (2015):
    - Instead of a single fully-connected Input->Excitatory layer,
      the input image is divided into (possibly overlapping) local patches.
    - Each patch connects to its own group of excitatory neurons via STDP.
    - This is analogous to a convolutional architecture WITHOUT weight
      sharing - each patch learns its own set of filters.
    - Benefits: drastically fewer synapses, spatially-structured features,
      and faster training.

Architecture:
    Input patches (Poisson) -> Per-patch Excitatory groups (STDP)
    Within each group: local WTA via lateral inhibition
    Feature vector: concatenation of spike counts from all groups
"""

import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from baselines.base_encoder import BaseEncoder


class LCSNNEncoder(BaseEncoder):
    """
    Locally Connected SNN encoder.

    The input image (assumed square, e.g. 28x28 for MNIST) is split into
    non-overlapping or overlapping patches.  Each patch has its own group
    of excitatory neurons trained with STDP + local WTA inhibition.  The
    output code is the concatenation of spike counts across all groups.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']

        # Image geometry (assume square)
        self.image_side = config.get('image_side', int(np.sqrt(self.input_dim)))

        # Patch parameters
        self.patch_size = config.get('patch_size', 14)    # pixels per patch side
        self.patch_stride = config.get('patch_stride', 14) # stride (non-overlapping default)

        # Neurons per patch
        self.neurons_per_patch = config.get('neurons_per_patch', 100)

        # Compute patch grid
        self.n_patches_side = (self.image_side - self.patch_size) // self.patch_stride + 1
        self.n_patches = self.n_patches_side ** 2
        self.patch_dim = self.patch_size ** 2
        self.n_neurons = self.n_patches * self.neurons_per_patch

        # Simulation
        self.simulation_time = config.get('simulation_time', 350)
        self.dt = config.get('dt', 1.0)

        # LIF parameters
        self.thresh = config.get('thresh', -52.0)
        self.rest = config.get('rest', -65.0)
        self.refrac = config.get('refrac', 5)

        # STDP
        self.nu = config.get('nu', (1e-4, 1e-2))
        if isinstance(self.nu, list):
            self.nu = tuple(self.nu)

        # Training
        self.n_train_samples = config.get('n_train_samples', None)
        self.intensity = config.get('intensity', 128.0)
        self.binarization_percent = config.get('binarization_percent', 0.05)

        # Device
        try:
            import torch
            self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            self.device = 'cpu'

        self.networks = []  # one SNN per patch
        self.is_trained = False

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------
    def _extract_patches(self, images: np.ndarray) -> np.ndarray:
        """
        Extract patches from flattened images.

        Args:
            images: (n_samples, input_dim)

        Returns:
            patches: (n_samples, n_patches, patch_dim)
        """
        n = len(images)
        imgs = images.reshape(n, self.image_side, self.image_side)
        patches = np.zeros((n, self.n_patches, self.patch_dim))

        idx = 0
        for r in range(self.n_patches_side):
            for c in range(self.n_patches_side):
                r0 = r * self.patch_stride
                c0 = c * self.patch_stride
                patch = imgs[:, r0:r0+self.patch_size, c0:c0+self.patch_size]
                patches[:, idx] = patch.reshape(n, -1)
                idx += 1

        return patches

    # ------------------------------------------------------------------
    # Build one per-patch SNN
    # ------------------------------------------------------------------
    def _build_patch_network(self, patch_idx: int):
        """Build a small SNN for a single patch."""
        import torch
        from bindsnet.network import Network
        from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
        from bindsnet.network.topology import Connection
        from bindsnet.learning import PostPre
        from bindsnet.network.monitors import Monitor

        network = Network(dt=self.dt)

        inp = Input(n=self.patch_dim, traces=True)
        exc = DiehlAndCookNodes(
            n=self.neurons_per_patch,
            traces=True,
            rest=self.rest, reset=-65.0, thresh=self.thresh,
            refrac=self.refrac, tc_decay=100.0, trace_tc=5e-2,
            theta_plus=0.05, tc_theta_decay=1e7,
        )
        inh = LIFNodes(
            n=self.neurons_per_patch, traces=False,
            rest=-60.0, reset=-45.0, thresh=-40.0,
            tc_decay=10.0, refrac=2, trace_tc=5e-2,
        )

        network.add_layer(inp, name="Input")
        network.add_layer(exc, name="Excitatory")
        network.add_layer(inh, name="Inhibitory")

        # Input -> Excitatory with STDP
        # Norm is scaled proportionally: 78.4 * (patch_dim / 784)
        norm_val = 78.4 * (self.patch_dim / 784.0)
        w = 0.3 * torch.rand(self.patch_dim, self.neurons_per_patch)
        network.add_connection(
            Connection(source=inp, target=exc, w=w,
                       update_rule=PostPre, nu=self.nu,
                       reduction=None, wmin=0.0, wmax=1.0,
                       norm=norm_val),
            source="Input", target="Excitatory")

        # Exc -> Inh (one-to-one)
        w_ei = 22.5 * torch.eye(self.neurons_per_patch)
        network.add_connection(
            Connection(source=exc, target=inh, w=w_ei, wmin=0, wmax=22.5),
            source="Excitatory", target="Inhibitory")

        # Inh -> Exc (all-to-all minus diagonal, standard WTA)
        w_ie = -17.5 * (torch.ones(self.neurons_per_patch, self.neurons_per_patch)
                        - torch.diag(torch.ones(self.neurons_per_patch)))
        network.add_connection(
            Connection(source=inh, target=exc, w=w_ie, wmin=-120.0, wmax=0.0),
            source="Inhibitory", target="Excitatory")

        # Monitor
        network.add_monitor(
            Monitor(exc, state_vars=["s"]),
            name="ExcitatoryMonitor")

        return network

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        import torch
        from bindsnet.encoding import poisson

        print(f"\n{'='*70}")
        print("Training LC-SNN (Locally Connected SNN)")
        print(f"{'='*70}")
        print(f"Image: {self.image_side}x{self.image_side}, "
              f"Patch: {self.patch_size}x{self.patch_size}, "
              f"Stride: {self.patch_stride}")
        print(f"Patches: {self.n_patches} ({self.n_patches_side}x{self.n_patches_side}), "
              f"Neurons/patch: {self.neurons_per_patch}, "
              f"Total neurons: {self.n_neurons}")
        print(f"Simulation: {self.simulation_time} ms | Device: {self.device}")
        print(f"{'='*70}\n")

        # Build per-patch networks
        self.networks = []
        for p in range(self.n_patches):
            net = self._build_patch_network(p)
            net.to(self.device)
            self.networks.append(net)
        print(f"Built {self.n_patches} patch networks.")

        # Data
        n_samples = len(train_data)
        if self.n_train_samples is not None and self.n_train_samples < n_samples:
            train_data = train_data[:self.n_train_samples]
            if train_labels is not None:
                train_labels = train_labels[:self.n_train_samples]
            n_samples = self.n_train_samples

        data_norm = train_data.copy()
        if data_norm.max() > 1.0:
            data_norm = data_norm / 255.0
        data_norm = np.clip(data_norm, 0.0, 1.0) * self.intensity

        patches_all = self._extract_patches(data_norm)  # (N, n_patches, patch_dim)

        print(f"Training on {n_samples} samples...")
        start_time = time.time()
        last_print_time = start_time

        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                now = time.time()
                elapsed = now - start_time
                speed = 100.0 / (now - last_print_time) if (now - last_print_time) > 0 else 0
                eta = (n_samples - (i + 1)) / speed / 60 if speed > 0 else 0
                label_str = f" (label={train_labels[i]})" if train_labels is not None else ""
                print(f"  Sample {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%){label_str}"
                      f" | Elapsed: {elapsed/60:.1f}m | Speed: {speed:.1f} s/s"
                      f" | ETA: {eta:.1f}m")
                last_print_time = now

            for p in range(self.n_patches):
                patch = patches_all[i, p]
                patch_tensor = torch.from_numpy(patch).float()
                encoded = poisson(datum=patch_tensor,
                                  time=int(self.simulation_time), dt=self.dt)
                if isinstance(encoded, torch.Tensor):
                    encoded = encoded.to(self.device)
                self.networks[p].run(inputs={"Input": encoded},
                                     time=int(self.simulation_time))
                self.networks[p].reset_state_variables()

        self.is_trained = True
        total = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training complete! Total time: {total/60:.1f} min")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_trained or not self.networks:
            raise RuntimeError("Encoder must be fitted before encoding.")

        import torch
        from bindsnet.encoding import poisson

        n_samples = len(data)
        print(f"Encoding {n_samples} samples (LC-SNN)...")

        for net in self.networks:
            net.train(mode=False)
            net.to(self.device)

        data_norm = data.copy()
        if data_norm.max() > 1.0:
            data_norm = data_norm / 255.0
        data_norm = np.clip(data_norm, 0.0, 1.0) * self.intensity

        patches_all = self._extract_patches(data_norm)

        # Concatenated spike counts: (n_samples, n_patches * neurons_per_patch)
        spike_counts = np.zeros((n_samples, self.n_neurons))

        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  Sample {i+1}/{n_samples}")

            for p in range(self.n_patches):
                patch = patches_all[i, p]
                patch_tensor = torch.from_numpy(patch).float()
                encoded = poisson(datum=patch_tensor,
                                  time=int(self.simulation_time), dt=self.dt)
                if isinstance(encoded, torch.Tensor):
                    encoded = encoded.to(self.device)
                self.networks[p].run(inputs={"Input": encoded},
                                     time=int(self.simulation_time))

                spikes = self.networks[p].monitors["ExcitatoryMonitor"].get("s")
                counts = torch.sum(spikes, dim=0).cpu().numpy().flatten()
                offset = p * self.neurons_per_patch
                spike_counts[i, offset:offset+self.neurons_per_patch] = counts[:self.neurons_per_patch]

                self.networks[p].reset_state_variables()

        pre_code = spike_counts
        k = max(int(self.n_neurons * self.binarization_percent), 1)
        code = self._top_k_binarization(pre_code, k)

        total = pre_code.sum(axis=1)
        print(f"  Spike stats - mean: {total.mean():.1f}, median: {np.median(total):.1f}, "
              f"max: {total.max():.0f}, zero-samples: {(total==0).sum()}/{n_samples}")
        active = (pre_code > 0).sum(axis=1)
        print(f"  Active neurons/sample - mean: {active.mean():.1f}, median: {np.median(active):.0f}")
        unique = len(np.unique(code, axis=0))
        print(f"  Unique binary codes: {unique}/{n_samples}")
        print("Encoding complete!")

        return {'pre_code': pre_code, 'code': code}

    def _top_k_binarization(self, features: np.ndarray, k: int) -> np.ndarray:
        binary = np.zeros_like(features)
        top_k = np.argsort(features, axis=1)[:, -k:]
        rows = np.arange(len(features))[:, None]
        binary[rows, top_k] = 1
        return binary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        if not self.is_trained or not self.networks:
            raise RuntimeError("Cannot save untrained model.")
        import torch
        super().save(path)
        model_path = path.replace('.pkl', '_networks.pt')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        state_dicts = [net.state_dict() for net in self.networks]
        torch.save({'network_state_dicts': state_dicts,
                     'device': self.device,
                     'n_patches': self.n_patches}, model_path)
        print(f"Network weights saved to {model_path}")

    def load(self, path: str):
        super().load(path)
        import torch
        model_path = path.replace('.pkl', '_networks.pt')
        if not Path(model_path).exists():
            print(f"Warning: Network weights not found: {model_path}")
            return
        ckpt = torch.load(model_path, map_location=self.device)
        self.networks = []
        for p in range(ckpt['n_patches']):
            net = self._build_patch_network(p)
            try:
                net.load_state_dict(ckpt['network_state_dicts'][p], strict=True)
            except RuntimeError:
                net.load_state_dict(ckpt['network_state_dicts'][p], strict=False)
            net.to(self.device)
            self.networks.append(net)
        if 'device' in ckpt:
            self.device = ckpt['device']
        print(f"Loaded {len(self.networks)} patch networks from {model_path}")
