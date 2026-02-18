#!/usr/bin/env python
"""
Training script for Diehl & Cook (2015) STDP baseline using BindsNET.

This is a more complete implementation that can be used to train the network.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline.datasets import load_dataset
from pipeline.utils import set_seed

# Try to import BindsNET
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
    print("WARNING: BindsNET not installed. Please install with: pip install bindsnet")


def build_diehl_cook_network(n_input=784, n_neurons=400, dt=1.0, nu=(1e-4, 1e-2)):
    """
    Build Diehl & Cook network using BindsNET.
    
    Args:
        n_input: Number of input neurons
        n_neurons: Number of excitatory neurons
        dt: Simulation timestep (ms)
        nu: STDP learning rates (pre, post)
    
    Returns:
        BindsNET Network object
    """
    if not BINDSNET_AVAILABLE:
        raise ImportError("BindsNET is required. Install with: pip install bindsnet")
    
    network = Network(dt=dt)
    
    # Input layer (flat shape — do NOT use shape=(1,1,N) which creates 4-D
    # state variables and breaks state_dict loading)
    input_layer = Input(n=n_input, traces=True)
    
    # Excitatory layer with adaptive threshold
    exc_layer = LIFNodes(
        n=n_neurons,
        traces=True,
        rest=-65.0,
        reset=-65.0,  # Must equal rest per Diehl & Cook 2015
        thresh=-52.0,
        refrac=5,
        tc_decay=100.0,  # Membrane time constant (ms)
        trace_tc=5e-2,
        theta_plus=0.05,  # Adaptive threshold increment
        tc_theta_decay=1e7,  # Adaptive threshold decay
    )

    # Inhibitory layer (for lateral inhibition)
    inh_layer = LIFNodes(
        n=n_neurons,
        traces=False,
        rest=-60.0,
        reset=-45.0,
        thresh=-40.0,
        tc_decay=10.0,  # Membrane time constant (ms)
        refrac=2,
        trace_tc=5e-2,
    )
    
    # Add layers
    network.add_layer(input_layer, name="Input")
    network.add_layer(exc_layer, name="Excitatory")
    network.add_layer(inh_layer, name="Inhibitory")
    
    # Input -> Excitatory (with STDP)
    w = 0.3 * torch.rand(n_input, n_neurons)
    input_exc_conn = Connection(
        source=input_layer,
        target=exc_layer,
        w=w,
        update_rule=PostPre,
        nu=nu,
        reduction=None,
        wmin=0.0,
        wmax=1.0,
        norm=78.4,  # Weight normalization
    )
    network.add_connection(input_exc_conn, source="Input", target="Excitatory")
    
    # Excitatory -> Inhibitory (one-to-one, weight=22.5)
    w = 22.5 * torch.eye(n_neurons)
    exc_inh_conn = Connection(
        source=exc_layer,
        target=inh_layer,
        w=w,
        wmin=0,
        wmax=22.5,
    )
    network.add_connection(exc_inh_conn, source="Excitatory", target="Inhibitory")

    # Inhibitory -> Excitatory (all-to-all except diagonal, lateral inhibition)
    # Weights must be NEGATIVE for inhibition (official default: -17.5)
    w = -17.5 * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
    inh_exc_conn = Connection(
        source=inh_layer,
        target=exc_layer,
        w=w,
        wmin=-120.0,
        wmax=0.0,
    )
    network.add_connection(inh_exc_conn, source="Inhibitory", target="Excitatory")
    
    # Add spike monitors (time=None avoids sliding-window empty-list bug)
    exc_monitor = Monitor(exc_layer, state_vars=["s"])
    network.add_monitor(exc_monitor, name="ExcitatoryMonitor")
    
    return network


def train_network(network, train_data, train_labels, n_epochs=1, time=350,
                   device="cpu", intensity=128.0):
    """
    Train the Diehl & Cook network on data.

    Args:
        network: BindsNET Network
        train_data: Training images (n_samples, 784)
        train_labels: Training labels (for analysis only)
        n_epochs: Number of training epochs
        time: Simulation time per sample (ms)
        device: 'cpu' or 'cuda'
        intensity: Poisson encoding intensity (Hz scaling factor)
    """
    network.to(device)

    n_samples = len(train_data)
    print(f"Training on {n_samples} samples for {n_epochs} epoch(s)...")

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")

        for i, (image, label) in enumerate(zip(train_data, train_labels)):
            if (i + 1) % 100 == 0:
                print(f"  Sample {i+1}/{n_samples} (label={label})")

            # Normalise to [0, 1] then scale by intensity so BindsNET's
            # poisson() sees Hz-level firing rates (not sub-Hz).
            image = torch.from_numpy(image).float()
            if image.max() > 1.0:
                image = image / 255.0
            image = image * intensity

            # Encode as Poisson spike train
            encoded = poisson(datum=image, time=int(time), dt=network.dt)
            
            # Clamp input and run simulation
            inputs = {"Input": encoded}
            network.run(inputs=inputs, time=int(time))
            
            # Reset network state
            network.reset_state_variables()
        
        print(f"Epoch {epoch+1} complete!")
    
    return network


def extract_spike_counts(network, test_data, time=350, device="cpu",
                          intensity=128.0):
    """
    Extract spike counts from trained network.

    Args:
        network: Trained BindsNET Network
        test_data: Test images (n_samples, 784)
        time: Simulation time per sample (ms)
        device: 'cpu' or 'cuda'
        intensity: Poisson encoding intensity (Hz scaling factor)

    Returns:
        Spike count matrix (n_samples, n_neurons)
    """
    network.to(device)
    network.train(mode=False)  # Disable learning

    n_samples = len(test_data)
    n_neurons = network.layers["Excitatory"].n
    spike_counts = np.zeros((n_samples, n_neurons))

    print(f"Extracting spike counts from {n_samples} samples...")

    for i, image in enumerate(test_data):
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}")

        # Prepare image — same normalisation + intensity scaling as training
        image = torch.from_numpy(image).float()
        if image.max() > 1.0:
            image = image / 255.0
        image = image * intensity

        # Encode
        encoded = poisson(datum=image, time=int(time), dt=network.dt)

        # Run simulation
        inputs = {"Input": encoded}
        network.run(inputs=inputs, time=int(time))

        # Get spike counts
        spikes = network.monitors["ExcitatoryMonitor"].get("s")
        spike_counts[i] = torch.sum(spikes, dim=0).cpu().numpy().flatten()[:n_neurons]

        # Reset
        network.reset_state_variables()

    return spike_counts


def main(args):
    if not BINDSNET_AVAILABLE:
        print("ERROR: BindsNET is not installed!")
        print("Please install with: pip install bindsnet")
        return
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    dataset = load_dataset(args.dataset, root=args.data_root)
    
    # Subsample if needed
    if args.n_train is not None:
        train_data = dataset['train_data'][:args.n_train]
        train_labels = dataset['train_labels'][:args.n_train]
    else:
        train_data = dataset['train_data']
        train_labels = dataset['train_labels']
    
    test_data = dataset['test_data']
    test_labels = dataset['test_labels']
    
    print(f"Train: {len(train_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Build network
    print(f"\nBuilding Diehl & Cook network...")
    print(f"  Input neurons: {args.input_dim}")
    print(f"  Excitatory neurons: {args.n_neurons}")
    print(f"  Simulation time: {args.time} ms")
    
    network = build_diehl_cook_network(
        n_input=args.input_dim,
        n_neurons=args.n_neurons,
        dt=args.dt,
        nu=tuple(args.nu)
    )
    
    # Train
    if args.train:
        network = train_network(
            network, 
            train_data, 
            train_labels,
            n_epochs=args.n_epochs,
            time=args.time,
            device=device
        )
        
        # Save network
        if args.save:
            save_path = Path(args.output_dir) / "models" / f"diehl_cook_seed{args.seed}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(network.state_dict(), save_path)
            print(f"\nModel saved to {save_path}")
    
    # Extract spike counts
    if args.extract:
        print("\nExtracting spike counts...")
        spike_counts = extract_spike_counts(network, test_data, time=args.time, device=device)
        
        # Save
        save_path = Path(args.output_dir) / "codes" / "diehl_cook" / args.dataset
        save_path.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path / f"pre_code_seed{args.seed}.npy", spike_counts)
        print(f"Spike counts saved to {save_path}")
        
        # Statistics
        print(f"\nSpike count statistics:")
        print(f"  Mean: {np.mean(spike_counts):.2f}")
        print(f"  Std: {np.std(spike_counts):.2f}")
        print(f"  Min: {np.min(spike_counts):.0f}")
        print(f"  Max: {np.max(spike_counts):.0f}")
        print(f"  Sparsity: {np.mean(spike_counts == 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diehl & Cook STDP network")
    
    # Data
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--n_train", type=int, default=None, help="Number of training samples (None=all)")
    
    # Architecture
    parser.add_argument("--input_dim", type=int, default=784, help="Input dimension")
    parser.add_argument("--n_neurons", type=int, default=400, help="Number of excitatory neurons")
    parser.add_argument("--time", type=int, default=350, help="Simulation time (ms)")
    parser.add_argument("--dt", type=float, default=1.0, help="Timestep (ms)")
    parser.add_argument("--nu", type=float, nargs=2, default=[1e-4, 1e-2], help="STDP learning rates")
    
    # Training
    parser.add_argument("--train", action="store_true", help="Train the network")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--extract", action="store_true", help="Extract spike counts")
    
    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save", action="store_true", help="Save trained model")
    
    args = parser.parse_args()
    main(args)
