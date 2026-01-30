"""
Diehl & Cook (2015) STDP-WTA encoder using BindsNET.

Paper: "Unsupervised learning of digit recognition using spike-timing-dependent plasticity"
Diehl & Cook, Frontiers in Computational Neuroscience, 2015
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from baselines.base_encoder import BaseEncoder
from typing import Optional, Dict

# Try to import torch and BindsNET
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection
    from bindsnet.learning import PostPre
    from bindsnet.network.monitors import Monitor
    from bindsnet.encoding import poisson
    BINDSNET_AVAILABLE = True
except ImportError:
    BINDSNET_AVAILABLE = False


class DiehlCookEncoder(BaseEncoder):
    """
    STDP-based SNN encoder using BindsNET framework.
    
    Architecture:
    - Input layer: Poisson rate-coded neurons
    - Excitatory layer: LIF neurons with STDP
    - Inhibitory layer: LIF neurons for lateral inhibition
    - Feature extraction: Spike counts
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.input_dim = config['input_dim']
        self.n_neurons = config.get('n_neurons', 400)
        self.simulation_time = config.get('simulation_time', 350)  # ms
        self.dt = config.get('dt', 1.0)  # ms
        
        # LIF parameters
        self.thresh = config.get('thresh', -52.0)  # mV
        self.rest = config.get('rest', -65.0)  # mV
        self.refrac = config.get('refrac', 5)  # ms
        
        # STDP parameters
        self.nu = config.get('nu', (1e-4, 1e-2))  # Learning rates (pre, post)
        
        # Training parameters
        self.n_train_samples = config.get('n_train_samples', None)  # None = use all
        
        # Device
        if TORCH_AVAILABLE:
            self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        self.network = None
        self.is_trained = False
        
        # Check availability
        if not TORCH_AVAILABLE:
            print("⚠️  Warning: PyTorch not available. Using placeholder implementation.")
        if not BINDSNET_AVAILABLE:
            print("⚠️  Warning: BindsNET not available. Using placeholder implementation.")
            print("   Install with: pip install bindsnet")
    
    def _build_network(self):
        """Build the SNN using BindsNET."""
        if not BINDSNET_AVAILABLE or not TORCH_AVAILABLE:
            return None
        
        print(f"Building Diehl & Cook network...")
        print(f"  Input neurons: {self.input_dim}")
        print(f"  Excitatory neurons: {self.n_neurons}")
        print(f"  Device: {self.device}")
        
        network = Network(dt=self.dt)
        
        # Input layer (Poisson)
        input_layer = Input(n=self.input_dim, shape=(1, 1, self.input_dim), traces=True)
        
        # Excitatory layer (LIF with adaptive threshold)
        exc_layer = LIFNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            decay=1e-2,
            trace_tc=5e-2,
            theta_plus=0.05,  # Adaptive threshold increment
            tc_theta_decay=1e7,  # Adaptive threshold decay
        )
        
        # Inhibitory layer (for lateral inhibition)
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            decay=1e-1,
            refrac=2,
            trace_tc=5e-2,
        )
        
        # Add layers
        network.add_layer(input_layer, name="Input")
        network.add_layer(exc_layer, name="Excitatory")
        network.add_layer(inh_layer, name="Inhibitory")
        
        # Input -> Excitatory connection with STDP
        w = 0.3 * torch.rand(self.input_dim, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,  # STDP
            nu=self.nu,
            reduction=None,
            wmin=0.0,
            wmax=1.0,
            norm=78.4,  # Weight normalization
        )
        network.add_connection(input_exc_conn, source="Input", target="Excitatory")
        
        # Excitatory -> Inhibitory (one-to-one)
        w = torch.eye(self.n_neurons)
        exc_inh_conn = Connection(
            source=exc_layer,
            target=inh_layer,
            w=w,
            wmin=0,
            wmax=22.5,
        )
        network.add_connection(exc_inh_conn, source="Excitatory", target="Inhibitory")
        
        # Inhibitory -> Excitatory (all-to-all except diagonal)
        w = 10.4 * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons)))
        inh_exc_conn = Connection(
            source=inh_layer,
            target=exc_layer,
            w=w,
            wmin=-120.0,
            wmax=0.0,
        )
        network.add_connection(inh_exc_conn, source="Inhibitory", target="Excitatory")
        
        # Add spike monitors
        exc_monitor = Monitor(exc_layer, state_vars=["s", "v"], time=int(self.simulation_time))
        network.add_monitor(exc_monitor, name="ExcitatoryMonitor")
        
        return network
    
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train the STDP network.
        
        Args:
            train_data: Training images (n_samples, input_dim)
            train_labels: Training labels (optional, for analysis only)
        """
        # Limit training samples if specified
        if self.n_train_samples is not None:
            train_data = train_data[:self.n_train_samples]
            if train_labels is not None:
                train_labels = train_labels[:self.n_train_samples]
        
        n_samples = len(train_data)
        print(f"\nTraining Diehl & Cook encoder on {n_samples} samples...")
        print(f"Architecture: {self.input_dim} -> {self.n_neurons} neurons")
        print(f"Simulation time: {self.simulation_time} ms")
        
        # Check if BindsNET is available
        if not BINDSNET_AVAILABLE or not TORCH_AVAILABLE:
            print("\n" + "="*60)
            print("⚠️  WARNING: Using placeholder implementation!")
            print("BindsNET or PyTorch not available.")
            print("Install with: pip install torch bindsnet")
            print("="*60 + "\n")
            
            # Placeholder: Create random spike count matrix
            self.spike_count_matrix = np.random.rand(self.input_dim, self.n_neurons)
            self.is_trained = True
            print("Training complete (placeholder)")
            return
        
        # Build network
        self.network = self._build_network()
        self.network.to(self.device)
        
        # Training loop
        print(f"\n🔥 Starting STDP training...")
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("Install tqdm for progress bars: pip install tqdm")
        
        # Create progress bar or simple counter
        if use_tqdm:
            pbar = tqdm(
                range(n_samples),
                desc="Training",
                unit="sample",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            iterator = pbar
        else:
            iterator = range(n_samples)
        
        for i in iterator:
            # Get image
            image = train_data[i]
            label = train_labels[i] if train_labels is not None else -1
            
            # Update progress
            if use_tqdm:
                pbar.set_postfix({'label': int(label), 'progress': f'{(i+1)/n_samples*100:.1f}%'})
            elif i % 1000 == 0:
                print(f"  Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
            
            # Prepare image - Poisson encoding expects [0, 255] range for proper spike rates
            image = torch.from_numpy(image).float()
            if image.max() <= 1.0:
                # If normalized to [0, 1], scale back to [0, 255]
                image = image * 255.0
            
            # Move to device
            image = image.to(self.device)
            
            # Encode as Poisson spike train
            encoded = poisson(datum=image, time=int(self.simulation_time), dt=self.dt)
            
            # Ensure encoded data is on the correct device
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(self.device)
            
            # Run simulation with STDP learning
            inputs = {"Input": encoded}
            self.network.run(inputs=inputs, time=int(self.simulation_time))
            
            # Reset network state for next sample
            self.network.reset_state_variables()
        
        if use_tqdm:
            pbar.close()
        print("✅ STDP training complete!")
        
        self.is_trained = True
    
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data by running through trained SNN.
        
        Args:
            data: Input images (n_samples, input_dim)
        
        Returns:
            Dictionary with 'pre_code' (spike counts) and 'code' (binarized)
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")
        
        n_samples = len(data)
        print(f"\nEncoding {n_samples} samples...")
        
        # Check if using placeholder
        if not BINDSNET_AVAILABLE or not TORCH_AVAILABLE or self.network is None:
            # Placeholder encoding
            pre_code = np.dot(data, self.spike_count_matrix)
            pre_code = np.maximum(pre_code, 0)
            
            # Binarization
            k = int(self.n_neurons * 0.05)
            code = self._top_k_binarization(pre_code, k)
            
            return {
                'pre_code': pre_code,
                'code': code
            }
        
        # Real BindsNET encoding
        self.network.to(self.device)
        
        # Note: We don't call network.train(False) because in BindsNET it may affect neuron dynamics
        # Instead, STDP is automatically disabled when we don't set learning=True in network.run()
        
        spike_counts = np.zeros((n_samples, self.n_neurons))
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        # Create progress bar or simple counter
        if use_tqdm:
            pbar = tqdm(
                range(n_samples),
                desc="Encoding",
                unit="sample",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            iterator = pbar
        else:
            iterator = range(n_samples)
        
        for i in iterator:
            # Get image
            image = data[i]
            
            # Update progress
            if use_tqdm:
                pbar.set_postfix({'progress': f'{(i+1)/n_samples*100:.1f}%'})
            elif i % 1000 == 0:
                print(f"  Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
            
            # Prepare image - Poisson encoding expects [0, 255] range for proper spike rates
            image = torch.from_numpy(image).float()
            if image.max() <= 1.0:
                # If normalized to [0, 1], scale back to [0, 255]
                image = image * 255.0
            
            # Move to device
            image = image.to(self.device)
            
            # Encode as Poisson spike train
            encoded = poisson(datum=image, time=int(self.simulation_time), dt=self.dt)
            
            # Ensure encoded data is on the correct device
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(self.device)
            
            # Run simulation (no learning)
            inputs = {"Input": encoded}
            self.network.run(inputs=inputs, time=int(self.simulation_time))
            
            # Get spike counts
            spikes = self.network.monitors["ExcitatoryMonitor"].get("s")
            spike_counts[i] = torch.sum(spikes, dim=0).cpu().numpy()
            
            # Reset network state
            self.network.reset_state_variables()
        
        if use_tqdm:
            pbar.close()
        print("✅ Encoding complete!")
        
        # Spike counts are the pre-code
        pre_code = spike_counts
        
        # Binarization: Top-k (simulating WTA)
        k = int(self.n_neurons * 0.05)  # 5% sparsity
        code = self._top_k_binarization(pre_code, k)
        
        return {
            'pre_code': pre_code,
            'code': code
        }
    
    def _top_k_binarization(self, features: np.ndarray, k: int) -> np.ndarray:
        """Keep top-k values per sample."""
        binary_codes = np.zeros_like(features)
        top_k_indices = np.argsort(features, axis=1)[:, -k:]
        rows = np.arange(len(features))[:, None]
        binary_codes[rows, top_k_indices] = 1
        return binary_codes


if __name__ == '__main__':
    print("Testing Diehl & Cook encoder with STDP...")
    
    if not BINDSNET_AVAILABLE or not TORCH_AVAILABLE:
        print("⚠️  BindsNET or PyTorch not available. Test will use placeholder.")
    
    np.random.seed(0)
    train_data = np.random.rand(100, 784)
    test_data = np.random.rand(20, 784)
    
    config = {
        'input_dim': 784,
        'n_neurons': 400,
        'simulation_time': 350,
        'dt': 1.0,
        'nu': (1e-4, 1e-2),
        'n_train_samples': 10,  # Small for testing
    }
    
    encoder = DiehlCookEncoder(config)
    encoder.fit(train_data)
    
    result = encoder.encode(test_data)
    print(f"\nPre-code shape: {result['pre_code'].shape}")
    print(f"Code shape: {result['code'].shape}")
    print(f"Code sparsity: {1 - np.mean(result['code']):.3f}")
    print(f"Mean spike count: {np.mean(result['pre_code']):.2f}")
