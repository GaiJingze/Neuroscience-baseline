"""
Deep STDP encoder for unsupervised clustering.

Reference: "Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity"
           Sen Lu & Abhronil Sengupta, Neuromorphic Computing and Engineering, 2024
           arXiv: 2307.04054

Key idea:
    Alternates between two phases (inspired by DeepCluster):
    1. STDP Feature Learning: A spiking WTA network learns features from
       input data via STDP (same as Diehl & Cook single layer).
    2. Pseudo-label Generation: K-means clustering on the learned spike
       count representations produces pseudo-labels.
    3. These pseudo-labels are used to re-organize / re-initialize the
       next round of STDP learning, encouraging the network to refine
       cluster boundaries.

    This is repeated for multiple rounds, progressively improving the
    alignment between STDP-learned features and cluster structure.

    The paper reports NMI = 0.8597 on Tiny ImageNet (10 classes).

Architecture:
    Layer 1: Input (Poisson) -> Excitatory_1 (STDP + WTA) -> features_1
    Clustering: K-means on features_1 -> pseudo-labels
    Layer 2: features_1 (Poisson) -> Excitatory_2 (STDP + WTA) -> features_2
    Final: K-means on features_2 -> cluster assignments
"""

import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from baselines.base_encoder import BaseEncoder


class DeepSTDPEncoder(BaseEncoder):
    """
    Deep STDP encoder with alternating STDP learning and clustering.

    Multi-layer STDP-WTA network where each layer's features are used
    as input to the next layer, with k-means pseudo-label guidance
    between rounds.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.input_dim = config['input_dim']

        # Layer configuration: list of neuron counts for each STDP layer
        self.layer_sizes = config.get('layer_sizes', [400, 200])
        self.n_layers = len(self.layer_sizes)

        # Number of alternating STDP-clustering rounds per layer
        self.n_rounds = config.get('n_rounds', 2)

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

        # Clustering
        self.n_clusters = config.get('n_clusters', 10)

        # Device
        try:
            import torch
            self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            self.device = 'cpu'

        self.layer_networks = []   # one SNN per layer
        self.is_trained = False

    # ------------------------------------------------------------------
    # Build a single STDP-WTA layer
    # ------------------------------------------------------------------
    def _build_layer_network(self, in_dim: int, n_neurons: int):
        """Build one STDP-WTA spiking layer."""
        import torch
        from bindsnet.network import Network
        from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
        from bindsnet.network.topology import Connection
        from bindsnet.learning import PostPre
        from bindsnet.network.monitors import Monitor

        network = Network(dt=self.dt)

        inp = Input(n=in_dim, traces=True)
        exc = DiehlAndCookNodes(
            n=n_neurons, traces=True,
            rest=self.rest, reset=-65.0, thresh=self.thresh,
            refrac=self.refrac, tc_decay=100.0, trace_tc=5e-2,
            theta_plus=0.05, tc_theta_decay=1e7,
        )
        inh = LIFNodes(
            n=n_neurons, traces=False,
            rest=-60.0, reset=-45.0, thresh=-40.0,
            tc_decay=10.0, refrac=2, trace_tc=5e-2,
        )

        network.add_layer(inp, name="Input")
        network.add_layer(exc, name="Excitatory")
        network.add_layer(inh, name="Inhibitory")

        # Input -> Excitatory (STDP)
        norm_val = 78.4 * (in_dim / 784.0)
        w = 0.3 * torch.rand(in_dim, n_neurons)
        network.add_connection(
            Connection(source=inp, target=exc, w=w,
                       update_rule=PostPre, nu=self.nu,
                       reduction=None, wmin=0.0, wmax=1.0,
                       norm=norm_val),
            source="Input", target="Excitatory")

        # Exc -> Inh (one-to-one)
        w_ei = 22.5 * torch.eye(n_neurons)
        network.add_connection(
            Connection(source=exc, target=inh, w=w_ei, wmin=0, wmax=22.5),
            source="Excitatory", target="Inhibitory")

        # Inh -> Exc (lateral WTA)
        w_ie = -17.5 * (torch.ones(n_neurons, n_neurons)
                        - torch.diag(torch.ones(n_neurons)))
        network.add_connection(
            Connection(source=inh, target=exc, w=w_ie, wmin=-120.0, wmax=0.0),
            source="Inhibitory", target="Excitatory")

        # Monitor
        network.add_monitor(Monitor(exc, state_vars=["s"]), name="ExcitatoryMonitor")

        return network

    # ------------------------------------------------------------------
    # Run one layer of STDP on data, return spike counts
    # ------------------------------------------------------------------
    def _run_layer(self, network, data: np.ndarray, n_neurons: int,
                   training: bool = True) -> np.ndarray:
        """
        Run data through a single STDP-WTA layer.

        Args:
            network: BindsNET Network
            data: (n_samples, in_dim) - already scaled to Hz
            n_neurons: number of excitatory neurons
            training: whether STDP is enabled

        Returns:
            spike_counts: (n_samples, n_neurons)
        """
        import torch
        from bindsnet.encoding import poisson

        network.train(mode=training)
        network.to(self.device)

        n_samples = len(data)
        spike_counts = np.zeros((n_samples, n_neurons))

        for i in range(n_samples):
            if (i + 1) % 500 == 0:
                print(f"    Sample {i+1}/{n_samples}")

            image_tensor = torch.from_numpy(data[i]).float()
            encoded = poisson(datum=image_tensor,
                              time=int(self.simulation_time), dt=self.dt)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(self.device)

            network.run(inputs={"Input": encoded}, time=int(self.simulation_time))

            spikes = network.monitors["ExcitatoryMonitor"].get("s")
            counts = torch.sum(spikes, dim=0).cpu().numpy().flatten()
            spike_counts[i] = counts[:n_neurons]

            network.reset_state_variables()

        return spike_counts

    # ------------------------------------------------------------------
    # K-means pseudo-label generation
    # ------------------------------------------------------------------
    def _kmeans_pseudo_labels(self, features: np.ndarray) -> np.ndarray:
        """Run k-means and return pseudo-labels."""
        from sklearn.cluster import KMeans

        # Normalize features for clustering
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feat_norm = features / norms

        km = KMeans(n_clusters=self.n_clusters, n_init=3, max_iter=100,
                    random_state=42)
        labels = km.fit_predict(feat_norm)
        return labels

    # ------------------------------------------------------------------
    # Prepare features as Poisson input for next layer
    # ------------------------------------------------------------------
    def _prepare_layer_input(self, spike_counts: np.ndarray) -> np.ndarray:
        """
        Normalize spike counts to [0, intensity] range for Poisson encoding
        in the next layer.
        """
        sc = spike_counts.copy()
        max_val = sc.max()
        if max_val > 0:
            sc = sc / max_val  # [0, 1]
        sc = sc * self.intensity
        return sc

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        import torch

        print(f"\n{'='*70}")
        print("Training Deep-STDP (Multi-layer STDP + Clustering)")
        print(f"{'='*70}")
        print(f"Layers: {[self.input_dim] + self.layer_sizes}")
        print(f"Rounds per layer: {self.n_rounds}")
        print(f"K-means clusters: {self.n_clusters}")
        print(f"Simulation: {self.simulation_time} ms | Device: {self.device}")
        print(f"{'='*70}\n")

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

        self.layer_networks = []
        current_input = data_norm
        current_dim = self.input_dim

        start_time = time.time()

        for layer_idx, n_neurons in enumerate(self.layer_sizes):
            print(f"\n--- Layer {layer_idx + 1}/{self.n_layers}: "
                  f"{current_dim} -> {n_neurons} neurons ---")

            for round_idx in range(self.n_rounds):
                print(f"\n  Round {round_idx + 1}/{self.n_rounds}:")

                # Build (or rebuild) the layer network
                network = self._build_layer_network(current_dim, n_neurons)

                # If not the first round, use pseudo-labels to reorder training
                # data by cluster (so each neuron sees similar inputs in sequence,
                # encouraging specialization).
                train_input = current_input
                if round_idx > 0:
                    print("    Reordering data by pseudo-labels...")
                    pseudo_labels = self._kmeans_pseudo_labels(prev_features)
                    sort_idx = np.argsort(pseudo_labels)
                    train_input = current_input[sort_idx]

                # Run STDP training
                print(f"    STDP training on {len(train_input)} samples...")
                prev_features = self._run_layer(
                    network, train_input, n_neurons, training=True)

                # Report stats
                total_spikes = prev_features.sum(axis=1)
                print(f"    Spike stats - mean: {total_spikes.mean():.1f}, "
                      f"zero-samples: {(total_spikes == 0).sum()}/{len(train_input)}")

                # Clustering quality check (if true labels available)
                if train_labels is not None and round_idx == self.n_rounds - 1:
                    pseudo = self._kmeans_pseudo_labels(prev_features)
                    # Quick NMI check
                    try:
                        from sklearn.metrics import normalized_mutual_info_score
                        # For reordered data, un-reorder labels
                        if round_idx > 0:
                            reordered_labels = train_labels[sort_idx]
                        else:
                            reordered_labels = train_labels
                        nmi = normalized_mutual_info_score(reordered_labels, pseudo)
                        print(f"    Layer {layer_idx+1} NMI (train): {nmi:.4f}")
                    except ImportError:
                        pass

            # Save the final network for this layer
            self.layer_networks.append(network)

            # Prepare input for next layer: run inference to get features
            # (use original ordering)
            print(f"  Generating features for next layer...")
            layer_features = self._run_layer(
                network, current_input, n_neurons, training=False)

            # Update for next layer
            current_input = self._prepare_layer_input(layer_features)
            current_dim = n_neurons

        self.is_trained = True
        total = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Deep-STDP training complete! Total time: {total/60:.1f} min")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Encoding (inference through all layers)
    # ------------------------------------------------------------------
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_trained or not self.layer_networks:
            raise RuntimeError("Encoder must be fitted before encoding.")

        n_samples = len(data)
        print(f"Encoding {n_samples} samples (Deep-STDP, {self.n_layers} layers)...")

        # Normalize input
        data_norm = data.copy()
        if data_norm.max() > 1.0:
            data_norm = data_norm / 255.0
        data_norm = np.clip(data_norm, 0.0, 1.0) * self.intensity

        current_input = data_norm

        for layer_idx, (network, n_neurons) in enumerate(
                zip(self.layer_networks, self.layer_sizes)):
            print(f"  Layer {layer_idx + 1}/{self.n_layers}...")
            features = self._run_layer(
                network, current_input, n_neurons, training=False)

            if layer_idx < self.n_layers - 1:
                # Prepare for next layer
                current_input = self._prepare_layer_input(features)

        # Final layer features
        pre_code = features
        last_neurons = self.layer_sizes[-1]
        k = max(int(last_neurons * self.binarization_percent), 1)
        code = self._top_k_binarization(pre_code, k)

        total = pre_code.sum(axis=1)
        print(f"  Spike stats - mean: {total.mean():.1f}, median: {np.median(total):.1f}, "
              f"max: {total.max():.0f}, zero-samples: {(total==0).sum()}/{n_samples}")
        active = (pre_code > 0).sum(axis=1)
        print(f"  Active neurons/sample - mean: {active.mean():.1f}, "
              f"median: {np.median(active):.0f}")
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
        if not self.is_trained or not self.layer_networks:
            raise RuntimeError("Cannot save untrained model.")
        import torch
        super().save(path)
        model_path = path.replace('.pkl', '_layers.pt')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        state_dicts = [net.state_dict() for net in self.layer_networks]
        torch.save({
            'layer_state_dicts': state_dicts,
            'layer_sizes': self.layer_sizes,
            'device': self.device,
        }, model_path)
        print(f"Layer weights saved to {model_path}")

    def load(self, path: str):
        super().load(path)
        import torch
        model_path = path.replace('.pkl', '_layers.pt')
        if not Path(model_path).exists():
            print(f"Warning: Layer weights not found: {model_path}")
            return
        ckpt = torch.load(model_path, map_location=self.device)
        dims = [self.input_dim] + ckpt['layer_sizes']
        self.layer_networks = []
        for i, sd in enumerate(ckpt['layer_state_dicts']):
            net = self._build_layer_network(dims[i], dims[i+1])
            try:
                net.load_state_dict(sd, strict=True)
            except RuntimeError:
                net.load_state_dict(sd, strict=False)
            net.to(self.device)
            self.layer_networks.append(net)
        if 'device' in ckpt:
            self.device = ckpt['device']
        print(f"Loaded {len(self.layer_networks)} layer networks from {model_path}")
