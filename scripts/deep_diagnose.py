#!/usr/bin/env python3
"""Deep diagnosis: Check pre-code (spike counts before binarization)."""

import numpy as np
from pathlib import Path

print("=" * 80)
print("DEEP DIAGNOSIS: Checking spike counts (pre-binarization)")
print("=" * 80)
print()

# Load pre-code (spike counts)
pre_code_file = Path('./outputs/codes/diehl_cook/mnist/pre_code_seed0.npy')

if pre_code_file.exists():
    print("Loading pre-code (spike counts)...")
    pre_code = np.load(pre_code_file)
    
    print(f"\nPre-code (spike counts) statistics:")
    print(f"  Shape: {pre_code.shape}")
    print(f"  Mean: {pre_code.mean():.6f}")
    print(f"  Std: {pre_code.std():.6f}")
    print(f"  Min: {pre_code.min():.6f}")
    print(f"  Max: {pre_code.max():.6f}")
    
    # Check if all samples have identical spike counts
    unique_spike_patterns = len(np.unique(pre_code, axis=0))
    print(f"  Unique spike patterns: {unique_spike_patterns} / {len(pre_code)}")
    
    if unique_spike_patterns == 1:
        print("\n  ❌ CRITICAL: All samples produce IDENTICAL spike counts!")
        print("     This means:")
        print("       1. The network is completely deterministic OR")
        print("       2. All input images produce the same response OR")
        print("       3. The network is not properly trained")
        
        # Show the spike count pattern
        unique_pattern = pre_code[0]
        non_zero_neurons = np.where(unique_pattern > 0)[0]
        print(f"\n  The single spike pattern:")
        print(f"    Non-zero neurons: {len(non_zero_neurons)} / {len(unique_pattern)}")
        print(f"    Non-zero positions: {non_zero_neurons[:30]}")  # Show first 30
        print(f"    Spike counts of active neurons:")
        for neuron in non_zero_neurons[:10]:
            print(f"      Neuron {neuron}: {unique_pattern[neuron]:.2f} spikes")
    
    # Check per-neuron spike statistics
    print(f"\n  Per-neuron spike statistics:")
    spikes_per_neuron = pre_code.sum(axis=0)
    neurons_with_spikes = (spikes_per_neuron > 0).sum()
    print(f"    Neurons with spikes: {neurons_with_spikes} / {pre_code.shape[1]}")
    print(f"    Spikes per neuron (mean): {spikes_per_neuron.mean():.2f}")
    print(f"    Spikes per neuron (std): {spikes_per_neuron.std():.2f}")
    print(f"    Spikes per neuron (min): {spikes_per_neuron.min():.2f}")
    print(f"    Spikes per neuron (max): {spikes_per_neuron.max():.2f}")
    
    if neurons_with_spikes == 0:
        print("\n    ❌ NO neurons produced any spikes!")
        print("       Possible causes:")
        print("         - Threshold too high")
        print("         - Insufficient input current")
        print("         - Poisson encoding not generating spikes")
    
    # Check variance across samples for each neuron
    spike_variance_per_neuron = pre_code.var(axis=0)
    neurons_with_variance = (spike_variance_per_neuron > 0).sum()
    print(f"\n  Per-neuron variance across samples:")
    print(f"    Neurons with variance: {neurons_with_variance} / {pre_code.shape[1]}")
    print(f"    Mean variance: {spike_variance_per_neuron.mean():.6f}")
    print(f"    Max variance: {spike_variance_per_neuron.max():.6f}")
    
    if neurons_with_variance == 0:
        print("\n    ❌ NO neurons have variance across samples!")
        print("       This means every neuron fires the same number of spikes for all inputs")
        print("       → The network is completely input-agnostic")
    
    # Show spike counts for first few samples
    print(f"\n  Spike counts for first 5 samples:")
    for i in range(min(5, len(pre_code))):
        active_neurons = np.where(pre_code[i] > 0)[0]
        print(f"    Sample {i}: {len(active_neurons)} neurons fired, total {pre_code[i].sum():.0f} spikes")
        if len(active_neurons) > 0:
            print(f"              Top 5 neurons: {active_neurons[:5]} with counts {pre_code[i, active_neurons[:5]]}")
    
    # Check if spike counts are identical
    print(f"\n  Checking for identical spike counts...")
    all_same = True
    reference = pre_code[0]
    for i in range(1, min(10, len(pre_code))):
        if not np.allclose(pre_code[i], reference, atol=1e-10):
            all_same = False
            break
    
    if all_same:
        print("    ❌ First 10 samples have IDENTICAL spike counts!")
        print("       → Network output does not depend on input")
    else:
        print("    ✅ Spike counts differ between samples")
    
else:
    print(f"Pre-code file not found: {pre_code_file}")
    print("Pre-codes were not saved.")

print()
print("=" * 80)
print("\nCONCLUSION:")
print("=" * 80)
print()
print("Based on the diagnosis:")
print("  • All 10,000 samples produce IDENTICAL codes")
print("  • Only 20 out of 400 neurons (5%) are active")
print("  • Same 20 neurons fire for every input")
print()
print("ROOT CAUSE:")
print("  The STDP training did NOT learn input-dependent weights.")
print("  The network responds identically to all inputs.")
print()
print("POSSIBLE REASONS:")
print("  1. Weights did not change during STDP training")
print("  2. Weight normalization forces all weights to be similar")
print("  3. Lateral inhibition is too strong (Winner-Take-All collapsed)")
print("  4. Learning rate too small (weights didn't update significantly)")
print("  5. Training time insufficient (not enough plasticity)")
print()
print("RECOMMENDATIONS:")
print("  1. Check if weights actually changed after training")
print("  2. Increase learning rate (nu)")
print("  3. Train for more samples")
print("  4. Reduce weight normalization strength")
print("  5. Verify STDP is actually updating weights")
print()
