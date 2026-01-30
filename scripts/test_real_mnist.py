#!/usr/bin/env python3
"""Test with real MNIST data"""

import numpy as np
import sys
sys.path.insert(0, '/hy-tmp/clustering')

from baselines.diehl_cook.encoder import DiehlCookEncoder
from pipeline.datasets import load_dataset

print("Loading MNIST dataset...")
dataset = load_dataset('mnist', './data')

print("\nCreating encoder...")
config = {
    'input_dim': 784,
    'n_neurons': 100,
    'simulation_time': 350,
    'dt': 1.0,
    'n_train_samples': 100  # Train on 100 samples only
}
encoder = DiehlCookEncoder(config)

print("\nTraining on 100 MNIST samples...")
encoder.fit(dataset['train_data'][:100], dataset['train_labels'][:100])

print("\nEncoding 50 test samples...")
result = encoder.encode(dataset['test_data'][:50])

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nPre-code (spike counts) shape: {result['pre_code'].shape}")
print(f"Pre-code statistics:")
print(f"  Min: {result['pre_code'].min():.2f}")
print(f"  Max: {result['pre_code'].max():.2f}")
print(f"  Mean: {result['pre_code'].mean():.2f}")
print(f"  Std: {result['pre_code'].std():.2f}")

# Check diversity
unique_codes = np.unique(result['code'], axis=0)
print(f"\nCode diversity:")
print(f"  Unique codes: {len(unique_codes)} / {len(result['code'])}")
print(f"  Diversity: {len(unique_codes)/len(result['code'])*100:.1f}%")

# Check per-neuron activity
per_neuron_spikes = result['pre_code'].sum(axis=0)
active_neurons = np.sum(per_neuron_spikes > 0)
print(f"\nNeuron activity:")
print(f"  Active neurons: {active_neurons} / {result['pre_code'].shape[1]}")
print(f"  Mean spikes per neuron: {per_neuron_spikes.mean():.2f}")
print(f"  Std spikes per neuron: {per_neuron_spikes.std():.2f}")

if len(unique_codes) > len(result['code']) * 0.5:
    print("\n✅ Codes look diverse! Clustering should work better now.")
else:
    print(f"\n⚠️  Code diversity is still low ({len(unique_codes)/len(result['code'])*100:.1f}%)")
    print("   May need more training samples or parameter tuning.")
