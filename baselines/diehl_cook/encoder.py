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
        if isinstance(self.nu, list):
            self.nu = tuple(self.nu)
        
        # Training parameters
        self.n_train_samples = config.get('n_train_samples', None)  # None = use all
        
        # Device support
        try:
            import torch
            self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            self.device = 'cpu'
        
        self.network = None
        self.is_trained = False
    
    def _build_network(self):
        """Build the SNN using BindsNET with full Diehl & Cook architecture."""
        try:
            import torch
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
        
        # Input layer with traces for STDP
        input_layer = Input(n=self.input_dim, shape=(1, 1, self.input_dim), traces=True)
        
        # Excitatory layer with adaptive threshold
        exc_layer = LIFNodes(
            n=self.n_neurons,
            traces=True,
            rest=self.rest,
            reset=-60.0,  # Reset potential
            thresh=self.thresh,
            refrac=self.refrac,
            decay=1e-2,  # Membrane time constant
            trace_tc=5e-2,  # Trace time constant
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
        
        # Input -> Excitatory (with STDP)
        w = 0.3 * torch.rand(self.input_dim, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,  # STDP rule
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
        
        # Inhibitory -> Excitatory (all-to-all except diagonal, lateral inhibition)
        w = 10.4 * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons)))
        inh_exc_conn = Connection(
            source=inh_layer,
            target=exc_layer,
            w=w,
            wmin=-120.0,
            wmax=0.0,
        )
        network.add_connection(inh_exc_conn, source="Inhibitory", target="Excitatory")
        
        # Add spike monitors for Excitatory layer
        exc_monitor = Monitor(exc_layer, state_vars=["s", "v"], time=int(self.simulation_time * 1.5))
        network.add_monitor(exc_monitor, name="ExcitatoryMonitor")
        
        return network
    
    def fit(self, train_data: np.ndarray, train_labels: Optional[np.ndarray] = None):
        """
        Train the STDP network using BindsNET.
        
        Args:
            train_data: Training images [n_samples, input_dim]
            train_labels: Optional labels (not used in training, only for analysis)
        """
        # Check BindsNET availability
        try:
            import torch
            from bindsnet.encoding import poisson
        except ImportError:
            raise ImportError(
                "BindsNET is required for Diehl & Cook encoder. "
                "Install with: pip install bindsnet"
            )
        
        print(f"\n{'='*70}")
        print("Training Diehl & Cook STDP Network")
        print(f"{'='*70}")
        print(f"Architecture: {self.input_dim} -> {self.n_neurons} neurons")
        print(f"Simulation time: {self.simulation_time} ms")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Build network
        self.network = self._build_network()
        self.network.to(self.device)
        
        # Handle data subsampling
        n_samples = len(train_data)
        if self.n_train_samples is not None and self.n_train_samples < n_samples:
            train_data = train_data[:self.n_train_samples]
            if train_labels is not None:
                train_labels = train_labels[:self.n_train_samples]
            n_samples = self.n_train_samples
            print(f"Subsampling to {n_samples} training samples")
        
        print(f"Training on {n_samples} samples...")
        
        # Normalize input data to [0, 1] range
        train_data_normalized = train_data.copy()
        if train_data_normalized.max() > 1.0:
            train_data_normalized = train_data_normalized / 255.0
        train_data_normalized = np.clip(train_data_normalized, 0.0, 1.0)
        
        # Time tracking
        import time
        start_time = time.time()
        last_print_time = start_time
        
        # Training loop
        for i, image in enumerate(train_data_normalized):
            if (i + 1) % 100 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                elapsed_since_last = current_time - last_print_time
                samples_per_sec = 100.0 / elapsed_since_last if elapsed_since_last > 0 else 0
                
                progress = (i + 1) / n_samples * 100
                remaining_samples = n_samples - (i + 1)
                eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                
                label_str = f" (label={train_labels[i]})" if train_labels is not None else ""
                time_str = f" | Elapsed: {elapsed/60:.1f}m | Speed: {samples_per_sec:.1f} samples/s"
                if eta_hours >= 1:
                    eta_str = f" | ETA: {eta_hours:.1f}h"
                elif eta_minutes >= 1:
                    eta_str = f" | ETA: {eta_minutes:.1f}m"
                else:
                    eta_str = f" | ETA: {eta_seconds:.0f}s"
                
                print(f"  Sample {i+1}/{n_samples} ({progress:.1f}%){label_str}{time_str}{eta_str}")
                last_print_time = current_time
            
            # Convert to torch tensor and move to device
            image_tensor = torch.from_numpy(image).float().to(self.device)
            
            # Encode as Poisson spike train
            encoded = poisson(
                datum=image_tensor,
                time=int(self.simulation_time),
                dt=self.dt
            )
            # Ensure encoded spikes are on the correct device
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(self.device)
            elif isinstance(encoded, dict):
                encoded = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in encoded.items()}
            
            # Run network simulation (STDP updates happen automatically)
            inputs = {"Input": encoded}
            self.network.run(inputs=inputs, time=int(self.simulation_time))
            
            # Reset network state for next sample
            self.network.reset_state_variables()
        
        self.is_trained = True
        print(f"\n{'='*70}")
        print("✅ Training complete!")
        print(f"{'='*70}\n")
    
    def encode(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode data by running through trained SNN and extracting spike counts.
        
        Args:
            data: Input images [n_samples, input_dim]
        
        Returns:
            Dictionary with:
                - 'pre_code': Spike counts [n_samples, n_neurons]
                - 'code': Binary codes [n_samples, n_neurons] (top-k binarized)
        """
        if not self.is_trained:
            raise RuntimeError("Encoder must be fitted before encoding.")
        
        if self.network is None:
            raise RuntimeError("Network not initialized. Please call fit() first.")
        
        try:
            import torch
            from bindsnet.encoding import poisson
        except ImportError:
            raise ImportError(
                "BindsNET is required for Diehl & Cook encoder. "
                "Install with: pip install bindsnet"
            )
        
        print(f"Encoding {len(data)} samples...")
        
        # Set network to evaluation mode (disable learning)
        self.network.train(mode=False)
        self.network.to(self.device)
        
        n_samples = len(data)
        spike_counts = np.zeros((n_samples, self.n_neurons))
        
        # Normalize input data to [0, 1] range
        data_normalized = data.copy()
        if data_normalized.max() > 1.0:
            data_normalized = data_normalized / 255.0
        data_normalized = np.clip(data_normalized, 0.0, 1.0)
        
        # Extract spike counts for each sample
        for i, image in enumerate(data_normalized):
            if (i + 1) % 100 == 0:
                print(f"  Sample {i+1}/{n_samples}")
            
            # Convert to torch tensor and move to device
            image_tensor = torch.from_numpy(image).float().to(self.device)
            
            # Encode as Poisson spike train
            encoded = poisson(
                datum=image_tensor,
                time=int(self.simulation_time),
                dt=self.dt
            )
            # Ensure encoded spikes are on the correct device
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.to(self.device)
            elif isinstance(encoded, dict):
                encoded = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in encoded.items()}
            
            # Run simulation
            inputs = {"Input": encoded}
            self.network.run(inputs=inputs, time=int(self.simulation_time))
            
            # Extract spike counts from monitor
            spikes = self.network.monitors["ExcitatoryMonitor"].get("s")
            spike_counts[i] = torch.sum(spikes, dim=0).cpu().numpy()
            
            # Reset network state for next sample
            self.network.reset_state_variables()
        
        # Pre-code is the spike counts
        pre_code = spike_counts
        
        # Binarization: Top-k (5% sparsity by default)
        k = max(int(self.n_neurons * 0.05), 1)
        code = self._top_k_binarization(pre_code, k)
        
        print(f"✅ Encoding complete!")
        
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
    
    def save(self, path: str):
        """
        Save trained model to disk.
        
        Args:
            path: Path to save file
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        
        if self.network is None:
            raise RuntimeError("Network not initialized. Cannot save.")
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for saving Diehl & Cook model.")
        
        # Save base encoder state
        super().save(path)
        
        # Save network weights
        model_path = path.replace('.pkl', '_network.pt')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'device': self.device,
        }, model_path)
        
        print(f"Network weights saved to {model_path}")
    
    def load(self, path: str):
        """
        Load trained model from disk.
        
        Args:
            path: Path to saved file
        """
        # Load base encoder state
        super().load(path)
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for loading Diehl & Cook model.")
        
        # Load network weights
        model_path = path.replace('.pkl', '_network.pt')
        if not Path(model_path).exists():
            print(f"Warning: Network weights file not found: {model_path}")
            print("Model will need to be retrained.")
            return
        
        # Rebuild network architecture
        self.network = self._build_network()
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.to(self.device)
        
        # Restore device if saved
        if 'device' in checkpoint:
            self.device = checkpoint['device']
        
        print(f"Network weights loaded from {model_path}")


# Try to import torch for BindsNET
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Diehl & Cook encoder will use placeholder.")


if __name__ == '__main__':
    print("Testing Diehl & Cook encoder (full implementation)...")
    
    np.random.seed(0)
    # Use smaller dataset for quick test
    train_data = np.random.rand(50, 784)
    test_data = np.random.rand(10, 784)
    
    config = {
        'input_dim': 784,
        'n_neurons': 400,
        'simulation_time': 350,
        'dt': 1.0,
        'nu': (1e-4, 1e-2),
        'device': 'cpu',  # Use CPU for testing
    }
    
    try:
        encoder = DiehlCookEncoder(config)
        encoder.fit(train_data)
        
        result = encoder.encode(test_data)
        print(f"\n{'='*70}")
        print("Test Results:")
        print(f"{'='*70}")
        print(f"Pre-code shape: {result['pre_code'].shape}")
        print(f"Code shape: {result['code'].shape}")
        print(f"Code sparsity: {1 - np.mean(result['code']):.3f}")
        print(f"Mean spike counts per sample: {np.mean(result['pre_code']):.2f}")
        print(f"{'='*70}\n")
    except ImportError as e:
        print(f"\n⚠️  BindsNET not available: {e}")
        print("This is expected if BindsNET is not installed.")
        print("The encoder will work when BindsNET is installed.")
