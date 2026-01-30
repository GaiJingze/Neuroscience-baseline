#!/usr/bin/env python3
"""Diagnose Diehl & Cook encoder to find why it's failing."""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from baselines.diehl_cook.encoder import DiehlCookEncoder
from pipeline.datasets import load_dataset

def diagnose_encoder():
    print("=" * 80)
    print("DIEHL & COOK ENCODER DIAGNOSIS")
    print("=" * 80)
    print()
    
    # Load a small subset of MNIST
    print("[1/6] Loading MNIST subset...")
    dataset = load_dataset('mnist', root='./data')
    train_data = dataset['train_data']
    train_labels = dataset['train_labels']
    test_data = dataset['test_data']
    test_labels = dataset['test_labels']
    
    # Use only 100 samples for quick diagnosis
    n_train = 100
    n_test = 50
    train_data = train_data[:n_train]
    train_labels = train_labels[:n_train]
    test_data = test_data[:n_test]
    test_labels = test_labels[:n_test]
    
    print(f"  Train: {n_train} samples")
    print(f"  Test: {n_test} samples")
    print(f"  Data range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    print()
    
    # Create encoder config
    print("[2/6] Creating encoder...")
    config = {
        'input_dim': 784,
        'n_neurons': 100,  # Smaller for faster diagnosis
        'simulation_time': 350,
        'dt': 1.0,
        'thresh': -52.0,
        'rest': -65.0,
        'refrac': 5,
        'nu': (1e-4, 1e-2),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    encoder = DiehlCookEncoder(config)
    print(f"  Encoder created")
    print(f"  Device: {encoder.device}")
    print()
    
    # Train encoder
    print("[3/6] Training encoder...")
    print("=" * 60)
    encoder.fit(train_data, train_labels)
    print("=" * 60)
    print()
    
    # Check if network was actually built
    print("[4/6] Checking network structure...")
    if encoder.network is None:
        print("  ❌ ERROR: Network was not built!")
        print("  Likely cause: BindsNET not available")
        return
    else:
        print("  ✅ Network exists")
        print(f"  Layers: {list(encoder.network.layers.keys())}")
        print(f"  Connections: {list(encoder.network.connections.keys())}")
        
        # Check weights
        input_exc_conn = encoder.network.connections[("Input", "Excitatory")]
        weights = input_exc_conn.w.cpu().numpy()
        print(f"\n  Input->Excitatory weights:")
        print(f"    Shape: {weights.shape}")
        print(f"    Mean: {weights.mean():.6f}")
        print(f"    Std: {weights.std():.6f}")
        print(f"    Min: {weights.min():.6f}")
        print(f"    Max: {weights.max():.6f}")
        print(f"    Zeros: {(weights == 0).sum()} / {weights.size}")
        
        # Check if weights are all the same
        weight_diversity = len(np.unique(np.round(weights, decimals=6)))
        print(f"    Unique values (rounded): {weight_diversity}")
        
        if weight_diversity < 100:
            print("    ⚠️  WARNING: Very low weight diversity!")
    print()
    
    # Encode test data
    print("[5/6] Encoding test data...")
    result = encoder.encode(test_data)
    pre_code = result['pre_code']
    code = result['code']
    
    print(f"\n  Pre-code (spike counts):")
    print(f"    Shape: {pre_code.shape}")
    print(f"    Mean: {pre_code.mean():.6f}")
    print(f"    Std: {pre_code.std():.6f}")
    print(f"    Min: {pre_code.min():.6f}")
    print(f"    Max: {pre_code.max():.6f}")
    print(f"    Zeros: {(pre_code == 0).sum()} / {pre_code.size} ({(pre_code == 0).sum() / pre_code.size * 100:.1f}%)")
    
    # Check spike count distribution
    total_spikes_per_sample = pre_code.sum(axis=1)
    print(f"\n  Spikes per sample:")
    print(f"    Mean: {total_spikes_per_sample.mean():.2f}")
    print(f"    Std: {total_spikes_per_sample.std():.2f}")
    print(f"    Min: {total_spikes_per_sample.min():.2f}")
    print(f"    Max: {total_spikes_per_sample.max():.2f}")
    
    # Check if all samples have the same code
    unique_codes = len(np.unique(code, axis=0))
    print(f"\n  Binary codes:")
    print(f"    Shape: {code.shape}")
    print(f"    Sparsity: {1 - code.mean():.4f}")
    print(f"    Unique codes: {unique_codes} / {len(code)}")
    
    if unique_codes == 1:
        print("    ❌ CRITICAL: All samples have identical codes!")
        print("    This explains NMI=0, ARI=0")
    elif unique_codes < len(code) * 0.1:
        print(f"    ⚠️  WARNING: Very low code diversity ({unique_codes / len(code) * 100:.1f}%)")
    else:
        print(f"    ✅ Code diversity seems OK ({unique_codes / len(code) * 100:.1f}%)")
    
    print()
    
    # Check per-neuron activity
    print("[6/6] Analyzing neuron activity...")
    spikes_per_neuron = pre_code.sum(axis=0)
    active_neurons = (spikes_per_neuron > 0).sum()
    
    print(f"  Active neurons: {active_neurons} / {encoder.n_neurons} ({active_neurons / encoder.n_neurons * 100:.1f}%)")
    print(f"  Spikes per neuron:")
    print(f"    Mean: {spikes_per_neuron.mean():.2f}")
    print(f"    Std: {spikes_per_neuron.std():.2f}")
    print(f"    Min: {spikes_per_neuron.min():.2f}")
    print(f"    Max: {spikes_per_neuron.max():.2f}")
    
    if active_neurons == 0:
        print("    ❌ CRITICAL: NO neurons are active!")
        print("    Possible causes:")
        print("      1. Threshold too high (neurons can't fire)")
        print("      2. Poisson encoding not generating spikes")
        print("      3. Input scaling issue")
    elif active_neurons < encoder.n_neurons * 0.1:
        print("    ⚠️  WARNING: Less than 10% of neurons are active")
    else:
        print("    ✅ Neuron activity seems reasonable")
    
    # Check if codes are diverse by label
    print("\n  Code diversity by label:")
    for label in range(10):
        label_mask = test_labels == label
        if label_mask.sum() > 0:
            label_codes = code[label_mask]
            unique_label_codes = len(np.unique(label_codes, axis=0))
            print(f"    Label {label}: {unique_label_codes} unique codes ({label_mask.sum()} samples)")
    
    print()
    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print()
    
    # Summary
    print("SUMMARY:")
    if unique_codes == 1:
        print("  🔴 CRITICAL ISSUE: All codes are identical")
        print("     → NMI and ARI will be 0")
        print("     → Likely cause: All neurons fire the same way OR no neurons fire at all")
    elif active_neurons == 0:
        print("  🔴 CRITICAL ISSUE: No neuron activity")
        print("     → Check threshold, input scaling, or Poisson encoding")
    elif unique_codes < len(code) * 0.1:
        print("  🟡 MODERATE ISSUE: Low code diversity")
        print("     → STDP learning may not be working properly")
    else:
        print("  🟢 Encoding seems to work")
        print("     → Problem might be elsewhere")
    
    print()

if __name__ == '__main__':
    diagnose_encoder()
