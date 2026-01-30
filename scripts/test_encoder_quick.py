#!/usr/bin/env python3
"""Quick test to see if encoder produces any spikes"""

import numpy as np
import sys
sys.path.insert(0, '/hy-tmp/clustering')

from baselines.diehl_cook.encoder import DiehlCookEncoder

print("Creating encoder...")
config = {
    'input_dim': 784,
    'n_neurons': 100,
    'simulation_time': 350,
    'dt': 1.0,
    'n_train_samples': 10
}
encoder = DiehlCookEncoder(config)

print("\nGenerating dummy data...")
# Create 10 random "images"
train_data = np.random.rand(10, 784)
test_data = np.random.rand(5, 784)

print("\nTraining...")
encoder.fit(train_data)

print("\nEncoding...")
result = encoder.encode(test_data)

print("\nResults:")
print(f"  Pre-code shape: {result['pre_code'].shape}")
print(f"  Pre-code stats:")
print(f"    Min: {result['pre_code'].min()}")
print(f"    Max: {result['pre_code'].max()}")
print(f"    Mean: {result['pre_code'].mean()}")
print(f"    Std: {result['pre_code'].std()}")
print(f"  Zero spike samples: {np.sum(result['pre_code'].sum(axis=1) == 0)}/{len(test_data)}")
