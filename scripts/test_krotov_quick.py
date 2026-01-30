#!/usr/bin/env python3
"""
Quick test of Krotov encoder implementation.
"""

import numpy as np
import sys
sys.path.insert(0, '/hy-tmp/clustering')

from baselines.krotov.encoder import KrotovEncoder

print("\n" + "="*70)
print("🧪 Testing Krotov Encoder Implementation")
print("="*70)

# Create small test dataset
print("\n1. Creating test data...")
n_train = 1000
n_test = 100
input_dim = 784

train_data = np.random.rand(n_train, input_dim)
test_data = np.random.rand(n_test, input_dim)

# Initialize encoder
print("\n2. Initializing encoder...")
config = {
    'input_dim': 784,
    'n_neurons': 100,
    'n_epochs': 20,  # Fewer epochs for quick test
    'batch_size': 100,
    'lr': 0.02,
    'delta': 0.4,
    'p': 2.0,
    'k': 2,
    'verbose': True
}

encoder = KrotovEncoder(config)
print(f"   {encoder}")

# Train
print("\n3. Training...")
encoder.fit(train_data)

# Encode
print("\n4. Encoding test data...")
result = encoder.encode(test_data)

# Check results
print("\n5. Checking results...")
print(f"   Pre-code shape: {result['pre_code'].shape}")
print(f"   Code shape: {result['code'].shape}")
print(f"   Pre-code stats:")
print(f"     Min: {result['pre_code'].min():.4f}")
print(f"     Max: {result['pre_code'].max():.4f}")
print(f"     Mean: {result['pre_code'].mean():.4f}")
print(f"     Std: {result['pre_code'].std():.4f}")
print(f"   Code sparsity: {1 - result['code'].mean():.4f}")

# Check diversity
unique_codes = np.unique(result['code'], axis=0)
diversity = len(unique_codes) / len(result['code']) * 100
print(f"\n   Code diversity: {len(unique_codes)}/{len(result['code'])} ({diversity:.1f}%)")

# Assessment
print("\n" + "="*70)
if diversity > 50:
    print("✅ Test passed! Code diversity looks good.")
elif diversity > 10:
    print("⚠️  Test passed with warning: Code diversity is moderate.")
else:
    print("❌ Test failed: Code diversity is too low.")
print("="*70 + "\n")
