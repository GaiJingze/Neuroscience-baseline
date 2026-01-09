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


class DiehlCookEncoder(BaseEncoder):
    """
    STDP-based SNN encoder using BindsNET framework.
    
    Architecture:
    - Input layer: Poisson rate-coded neurons
    - Excitatory layer: LIF neurons with STDP
    - Lateral inhibition (Winner-Take-All)
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
        
        self.network = None
        self.is_trained = False
    
    def _build_network(self):
        """Build the SNN using BindsNET."""
        try:
            from bindsnet.network import Network
            from bindsnet.network.nodes import Input, LIFNodes
            from bindsnet.network.topology import Connection
            from bindsnet.learning import PostPre
            from bindsnet.network.monitors import Monitor
        except ImportError:
            raise ImportError(
                "BindsNET is required for Diehl & Cook encoder. "
                "Install with: pip install bindsnet"
            )
        
        # Create network
        network = Network(dt=self.dt)
        
        # Input layer (Poisson)
        input_layer = Input(n=self.input_dim, shape=(1, 1, self.input_dim))
        
        # Excitatory layer (LIF with adaptive threshold)
        exc_layer = LIFNodes(
            n=self.n_neurons,
            thresh=self.thresh,
            rest=self.rest,
            refrac=self.refrac,
            decay=1e-2,  # Membrane time constant
            theta_plus=0.05,  # Adaptive threshold increment
            tc_theta_decay=1e7,  # Adaptive threshold decay
        )
        
        # Inhibitory layer (for lateral inhibition)
        # Note: Can be implemented as direct connections or separate layer
        # Here we use a simplified approach
        
        # Add layers
        network.add_layer(input_layer, name='input')
        network.add_layer(exc_layer, name='excitatory')
        
        # Input -> Excitatory connection with STDP
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=0.3 * torch.rand(self.input_dim, self.n_neurons),  # Random init
            update_rule=PostPre,  # STDP
            nu=self.nu,
            wmin=0.0,
            wmax=1.0,
        )
        network.add_connection(input_exc_conn, source='input', target='excitatory')
        
        # Lateral inhibition (excitatory -> excitatory)
        # All-to-all except self-connections, with negative weights
        w_inh = -100.0 * (torch.ones(self.n_neurons, self.n_neurons) - torch.eye(self.n_neurons))
        exc_exc_conn = Connection(
            source=exc_layer,
            target=exc_layer,
            w=w_inh,
            wmin=-100.0,
            wmax=0.0,
        )
        network.add_connection(exc_exc_conn, source='excitatory', target='excitatory')
        
        # Add monitors
        network.add_monitor(
            Monitor(exc_layer, ['s']),  # Spike monitor
            name='excitatory_spikes'
        )
        
        return network
    
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train the STDP network.
        
        Note: This is a simplified placeholder. Full implementation requires:
        1. Building the network with BindsNET
        2. Converting images to Poisson spike trains
        3. Running simulation for each training sample
        4. Recording learned weights
        """
        print(f"Training Diehl & Cook encoder on {len(train_data)} samples...")
        print(f"Architecture: {self.input_dim} -> {self.n_neurons} neurons")
        print(f"Simulation time: {self.simulation_time} ms")
        
        # NOTE: Full implementation would go here
        # For now, we'll create a placeholder that can be filled in
        
        print("\n" + "="*60)
        print("WARNING: This is a skeleton implementation!")
        print("Full BindsNET training needs to be implemented.")
        print("See baselines/diehl_cook/train.py for detailed implementation.")
        print("="*60 + "\n")
        
        # Placeholder: Create random spike count matrix (for testing)
        # In real implementation, this would be learned through STDP
        self.spike_count_matrix = np.random.rand(self.input_dim, self.n_neurons)
        
        self.is_trained = True
        print("Training complete (placeholder)")
    
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data by running through trained SNN.
        
        Returns spike counts as features.
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")
        
        print(f"Encoding {len(data)} samples...")
        
        # NOTE: Full implementation would:
        # 1. Convert data to Poisson spike trains
        # 2. Run network simulation
        # 3. Record spike counts per neuron
        
        # Placeholder: Simple projection + ReLU
        pre_code = np.dot(data, self.spike_count_matrix)
        pre_code = np.maximum(pre_code, 0)  # ReLU (spike counts are positive)
        
        # Binarization: Top-k (simulating WTA)
        k = int(self.n_neurons * 0.05)  # 5% sparsity
        code = self._top_k_binarization(pre_code, k)
        
        return {
            'pre_code': pre_code,
            'code': code
        }
    
    def _top_k_binarization(self, features: np.ndarray, k: int) -> np.ndarray:
        """Keep top-k values."""
        binary_codes = np.zeros_like(features)
        top_k_indices = np.argsort(features, axis=1)[:, -k:]
        rows = np.arange(len(features))[:, None]
        binary_codes[rows, top_k_indices] = 1
        return binary_codes


# Try to import torch for BindsNET
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Diehl & Cook encoder will use placeholder.")


if __name__ == '__main__':
    print("Testing Diehl & Cook encoder (skeleton)...")
    
    np.random.seed(0)
    train_data = np.random.rand(100, 784)
    test_data = np.random.rand(20, 784)
    
    config = {
        'input_dim': 784,
        'n_neurons': 400,
        'simulation_time': 350,
        'dt': 1.0,
        'nu': (1e-4, 1e-2),
    }
    
    encoder = DiehlCookEncoder(config)
    encoder.fit(train_data)
    
    result = encoder.encode(test_data)
    print(f"Pre-code shape: {result['pre_code'].shape}")
    print(f"Code shape: {result['code'].shape}")
    print(f"Code sparsity: {1 - np.mean(result['code']):.3f}")
