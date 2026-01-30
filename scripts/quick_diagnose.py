#!/usr/bin/env python3
"""Quick diagnosis of Diehl & Cook encoder."""

import numpy as np
import json
from pathlib import Path

# Load a result file
result_file = Path('./outputs/results/diehl_cook_mnist_seed0.json')

print("=" * 80)
print("QUICK DIAGNOSIS: Diehl & Cook MNIST Seed 0")
print("=" * 80)
print()

with open(result_file, 'r') as f:
    data = json.load(f)

print("Result file contents:")
print(f"  NMI: {data['clustering']['kmeans']['nmi']}")
print(f"  ARI: {data['clustering']['kmeans']['ari']}")
print(f"  ACC: {data['clustering']['kmeans']['acc']}")
print(f"  Silhouette: {data['clustering']['kmeans']['silhouette']}")
print()

# Try to load the actual codes
code_file = Path('./outputs/codes/diehl_cook/mnist/code_seed0.npy')

if code_file.exists():
    print("Loading saved codes...")
    codes = np.load(code_file)
    
    print(f"\nCode statistics:")
    print(f"  Shape: {codes.shape}")
    print(f"  Mean: {codes.mean():.6f}")
    print(f"  Std: {codes.std():.6f}")
    print(f"  Min: {codes.min():.6f}")
    print(f"  Max: {codes.max():.6f}")
    print(f"  Sparsity: {1 - codes.mean():.4f}")
    
    # Check unique codes
    unique_codes = len(np.unique(codes, axis=0))
    print(f"  Unique codes: {unique_codes} / {len(codes)}")
    
    if unique_codes == 1:
        print("\n  ❌ CRITICAL: All codes are IDENTICAL!")
        print("     This is why NMI and ARI are 0")
        
        # Show the single unique code
        print(f"\n  The single unique code:")
        unique_code = np.unique(codes, axis=0)[0]
        print(f"    Non-zero positions: {np.where(unique_code > 0)[0][:20]}")  # Show first 20
        print(f"    Number of non-zero: {(unique_code > 0).sum()}")
    elif unique_codes < 10:
        print(f"\n  ⚠️  WARNING: Only {unique_codes} unique codes for 10,000 samples!")
        print("     This is why clustering performance is terrible")
    
    # Check per-sample activity
    activity_per_sample = codes.sum(axis=1)
    print(f"\n  Activity per sample:")
    print(f"    Mean: {activity_per_sample.mean():.2f}")
    print(f"    Std: {activity_per_sample.std():.2f}")
    print(f"    Min: {activity_per_sample.min():.2f}")
    print(f"    Max: {activity_per_sample.max():.2f}")
    
    if activity_per_sample.std() == 0:
        print("    ❌ All samples have the SAME number of active neurons!")
    
    # Check per-neuron activity
    activity_per_neuron = codes.sum(axis=0)
    active_neurons = (activity_per_neuron > 0).sum()
    print(f"\n  Neuron activity:")
    print(f"    Active neurons: {active_neurons} / {codes.shape[1]}")
    print(f"    Activity per neuron (mean): {activity_per_neuron.mean():.2f}")
    print(f"    Activity per neuron (std): {activity_per_neuron.std():.2f}")
    
    if active_neurons == 0:
        print("    ❌ NO neurons are active!")
    elif active_neurons < codes.shape[1] * 0.1:
        print(f"    ⚠️  Only {active_neurons / codes.shape[1] * 100:.1f}% of neurons are active")
    
    # Show a few example codes
    print(f"\n  Sample codes (first 5 samples):")
    for i in range(min(5, len(codes))):
        active_positions = np.where(codes[i] > 0)[0]
        print(f"    Sample {i}: {len(active_positions)} active neurons at positions {active_positions[:10]}")
        
else:
    print(f"Code file not found: {code_file}")
    print("Codes were not saved or the experiment didn't run.")

print()
print("=" * 80)
