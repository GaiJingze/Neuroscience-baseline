"""
Lattice Map SNN (LM-SNN) encoder.

Reference: "Unsupervised Learning with Self-Organizing Spiking Neural Networks"
           Hazan, Saunders et al., IJCNN 2018 / Annals of Math & AI 2020

Key difference from Diehl & Cook (2015):
    - Neurons are arranged on a 2D grid (lattice).
    - Inhibition is topological: nearby neurons on the grid inhibit each other
      more strongly than distant neurons, mimicking Self-Organizing Map (SOM)
      dynamics.
    - This encourages spatially-organized receptive fields and faster
      convergence of diverse filters compared to global WTA inhibition.

Architecture:
    Input (Poisson) -> Excitatory (STDP + adaptive threshold) -> Inhibitory
    The inhibitory -> excitatory connection uses distance-dependent weights
    on a 2D grid instead of uniform all-to-all inhibition.
"""

import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from baselines.base_encoder import BaseEncoder


class LMSNNEncoder(BaseEncoder):
    """
    Lattice Map SNN encoder with topological (SOM-inspired) inhibition.

    Instead of uniform lateral inhibition (Diehl & Cook), excitatory neurons
    are placed on a 2D grid and the inhibitory weights decrease with grid
    distance.  This produces self-organizing feature maps in the spiking
    domain.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']
        self.n_neurons = config.get('n_neurons', 400)
        self.simulation_time = config.get('simulation_time', 350)
        self.dt = config.get('dt', 1.0)

        # LIF parameters
        self.thresh = config.get('thresh', -52.0)
        self.rest = config.get('rest', -65.0)
        self.refrac = config.get('refrac', 5)

        # STDP parameters
        self.nu = config.get('nu', (1e-4, 1e-2))
        if isinstance(self.nu, list):
            self.nu = tuple(self.nu)

        # Training
        self.n_train_samples = config.get('n_train_samples', None)
        self.intensity = config.get('intensity', 128.0)
        self.binarization_percent = config.get('binarization_percent', 0.05)

        # LM-SNN specific: grid and inhibition parameters
        self.grid_side = config.get('grid_side', int(np.sqrt(self.n_neurons)))
        # Ensure n_neurons matches grid
        self.n_neurons = self.grid_side ** 2

        # SOM-style inhibition neighbourhood radius (initial)
        self.inh_sigma = config.get('inh_sigma', self.grid_side / 4.0)
        # Base inhibition strength
        self.inh_strength = config.get('inh_strength', -17.5)

        # Device
        try:
            import torch
            self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            self.device = 'cpu'

        self.network = None
        self.is_trained = False

    # ------------------------------------------------------------------
    # Grid utilities
    # ------------------------------------------------------------------
    def _grid_positions(self):
        """Return (n_neurons, 2) array of 2D grid coordinates."""
        positions = np.zeros((self.n_neurons, 2))
        for idx in range(self.n_neurons):
            positions[idx] = [idx // self.grid_side, idx % self.grid_side]
        return positions

    def _distance_matrix(self):
        """Euclidean distance between every pair of neurons on the grid."""
        pos = self._grid_positions()
        diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 2)
        return np.sqrt((diff ** 2).sum(axis=-1))   # (N, N)

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    def _build_network(self):
        """Build SNN with topological inhibition using BindsNET."""
        import torch
        from bindsnet.network import Network
        from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
        from bindsnet.network.topology import Connection
        from bindsnet.learning import PostPre
        from bindsnet.network.monitors import Monitor

        network = Network(dt=self.dt)

        # --- Layers ---
        input_layer = Input(n=self.input_dim, traces=True)
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=self.rest,
            reset=-65.0,
            thresh=self.thresh,
            refrac=self.refrac,
            tc_decay=100.0,
            trace_tc=5e-2,
            theta_plus=0.05,
            tc_theta_decay=1e7,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            trace_tc=5e-2,
        )

        network.add_layer(input_layer, name="Input")
        network.add_layer(exc_layer, name="Excitatory")
        network.add_layer(inh_layer, name="Inhibitory")

        # --- Input -> Excitatory (STDP) ---
        w_inp = 0.3 * torch.rand(self.input_dim, self.n_neurons)
        inp_exc = Connection(
            source=input_layer,
            target=exc_layer,
            w=w_inp,
            update_rule=PostPre,
            nu=self.nu,
            reduction=None,
            wmin=0.0,
            wmax=1.0,
            norm=78.4,
        )
        network.add_connection(inp_exc, source="Input", target="Excitatory")

        # --- Excitatory -> Inhibitory (one-to-one) ---
        w_ei = 22.5 * torch.eye(self.n_neurons)
        exc_inh = Connection(
            source=exc_layer, target=inh_layer,
            w=w_ei, wmin=0, wmax=22.5,
        )
        network.add_connection(exc_inh, source="Excitatory", target="Inhibitory")

        # --- Inhibitory -> Excitatory (TOPOLOGICAL, distance-dependent) ---
        dist = self._distance_matrix()  # (N, N)
        # Gaussian falloff: strength * exp(-d^2 / (2*sigma^2)), zero on diagonal
        sigma = self.inh_sigma
        gauss = np.exp(- dist ** 2 / (2.0 * sigma ** 2))
        np.fill_diagonal(gauss, 0.0)
        w_ie = torch.tensor(self.inh_strength * gauss, dtype=torch.float32)

        inh_exc = Connection(
            source=inh_layer, target=exc_layer,
            w=w_ie,
            wmin=-120.0, wmax=0.0,
        )
        network.add_connection(inh_exc, source="Inhibitory", target="Excitatory")

        # --- Monitor ---
        exc_monitor = Monitor(exc_layer, state_vars=["s"])
        network.add_monitor(exc_monitor, name="ExcitatoryMonitor")

        return network

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        import torch
        from bindsnet.encoding import poisson

        print(f"\n{'='*70}")
        print("Training LM-SNN (Lattice Map SNN)")
        print(f"{'='*70}")
        print(f"Architecture: {self.input_dim} -> {self.n_neurons} neurons "
              f"({self.grid_side}x{self.grid_side} grid)")
        print(f"Inhibition sigma: {self.inh_sigma:.1f}, strength: {self.inh_strength}")
        print(f"Simulation time: {self.simulation_time} ms | Device: {self.device}")
        print(f"{'='*70}\n")

        self.network = self._build_network()
        self.network.to(self.device)

        n_samples = len(train_data)
        if self.n_train_samples is not None and self.n_train_samples < n_samples:
            train_data = train_data[:self.n_train_samples]
            if train_labels is not None:
                train_labels = train_labels[:self.n_train_samples]
            n_samples = self.n_train_samples
            print(f"Subsampling to {n_samples} training samples")

        print(f"Training on {n_samples} samples...")

        data_norm = train_data.copy()
        if data_norm.max() > 1.0:
            data_norm = data_norm / 255.0
        data_norm = np.clip(data_norm, 0.0, 1.0) * self.intensity

        start_time = time.time()
        last_print_time = start_time

        for i, image in enumerate(data_norm):
            if (i + 1) % 100 == 0:
                now = time.time()
                elapsed = now - start_time
                speed = 100.0 / (now - last_print_time) if (now - last_print_time) > 0 else 0
                eta = (n_samples - (i + 1)) / speed if speed > 0 else 0
                label_str = f" (label={train_labels[i]})" if train_labels is not None else ""
                print(f"  Sample {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%){label_str}"
                      f" | Elapsed: {elapsed/60:.1f}m | Speed: {speed:.1f} s/s"
                      f" | ETA: {eta/60:.1f}m")
                last_print_time = now

            image_tensor = torch.from_numpy(image).float()
            encoded = poisson(datum=image_tensor, time=int(self.simulation_time), dt=self.dt)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(self.device)
            self.network.run(inputs={"Input": encoded}, time=int(self.simulation_time))
            self.network.reset_state_variables()

        self.is_trained = True
        total = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training complete! Total time: {total/60:.1f} min")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_trained or self.network is None:
            raise RuntimeError("Encoder must be fitted before encoding.")

        import torch
        from bindsnet.encoding import poisson

        print(f"Encoding {len(data)} samples (LM-SNN)...")

        self.network.train(mode=False)
        self.network.to(self.device)

        n_samples = len(data)
        spike_counts = np.zeros((n_samples, self.n_neurons))

        data_norm = data.copy()
        if data_norm.max() > 1.0:
            data_norm = data_norm / 255.0
        data_norm = np.clip(data_norm, 0.0, 1.0) * self.intensity

        for i, image in enumerate(data_norm):
            if (i + 1) % 100 == 0:
                print(f"  Sample {i+1}/{n_samples}")

            image_tensor = torch.from_numpy(image).float()
            encoded = poisson(datum=image_tensor, time=int(self.simulation_time), dt=self.dt)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(self.device)

            self.network.run(inputs={"Input": encoded}, time=int(self.simulation_time))

            spikes = self.network.monitors["ExcitatoryMonitor"].get("s")
            counts = torch.sum(spikes, dim=0).cpu().numpy().flatten()
            spike_counts[i] = counts[:self.n_neurons]
            self.network.reset_state_variables()

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
        if not self.is_trained or self.network is None:
            raise RuntimeError("Cannot save untrained model.")
        import torch
        super().save(path)
        model_path = path.replace('.pkl', '_network.pt')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'network_state_dict': self.network.state_dict(),
                     'device': self.device}, model_path)
        print(f"Network weights saved to {model_path}")

    def load(self, path: str):
        super().load(path)
        import torch
        model_path = path.replace('.pkl', '_network.pt')
        if not Path(model_path).exists():
            print(f"Warning: Network weights not found: {model_path}")
            return
        self.network = self._build_network()
        ckpt = torch.load(model_path, map_location=self.device)
        try:
            self.network.load_state_dict(ckpt['network_state_dict'], strict=True)
        except RuntimeError:
            self.network.load_state_dict(ckpt['network_state_dict'], strict=False)
        self.network.to(self.device)
        if 'device' in ckpt:
            self.device = ckpt['device']
        print(f"Network weights loaded from {model_path}")
