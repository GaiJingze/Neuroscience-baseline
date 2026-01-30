#!/usr/bin/env python3
"""Detailed debugging of Diehl & Cook encoder"""

import numpy as np
import torch
import sys
sys.path.insert(0, '/hy-tmp/clustering')

from baselines.diehl_cook.encoder import DiehlCookEncoder
from bindsnet.encoding import poisson

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
train_data = np.random.rand(10, 784)
test_data = np.random.rand(2, 784)

print("\nTraining...")
encoder.fit(train_data)

print("\n" + "="*80)
print("DEBUGGING NETWORK STATE")
print("="*80)

# Check weights
print("\n1. Checking trained weights...")
input_exc_conn = encoder.network.connections[("Input", "Excitatory")]
print(f"   Weight shape: {input_exc_conn.w.shape}")
print(f"   Weight stats:")
print(f"     Min: {input_exc_conn.w.min().item():.4f}")
print(f"     Max: {input_exc_conn.w.max().item():.4f}")
print(f"     Mean: {input_exc_conn.w.mean().item():.4f}")
print(f"     Std: {input_exc_conn.w.std().item():.4f}")

# Check if weights are all zeros
if input_exc_conn.w.abs().sum() < 1e-6:
    print(f"   ⚠️  WARNING: All weights are nearly zero!")
else:
    print(f"   ✓ Weights look reasonable")

# Manually test encoding one sample
print("\n2. Manually testing encoding of one sample...")
image = torch.from_numpy(test_data[0]).float()
if image.max() > 1.0:
    image = image / 255.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
encoder.network.to(device)

# Encode as Poisson spikes
encoded = poisson(datum=image, time=int(encoder.simulation_time), dt=encoder.dt)
if isinstance(encoded, torch.Tensor):
    encoded = encoded.to(device)

print(f"   Input spike shape: {encoded.shape}")
print(f"   Input spike stats:")
print(f"     Total spikes: {encoded.sum().item()}")
print(f"     Mean spikes per neuron: {encoded.sum(dim=0).float().mean().item():.2f}")
print(f"     Neurons with spikes: {(encoded.sum(dim=0) > 0).sum().item()}/{encoded.shape[1]}")

if encoded.sum() < 1:
    print(f"   ⚠️  WARNING: No input spikes generated!")
else:
    print(f"   ✓ Input spikes look reasonable")

# Run simulation
print("\n3. Running simulation...")
inputs = {"Input": encoded}

# Check excitatory layer state before
exc_layer = encoder.network.layers["Excitatory"]
print(f"   Before simulation:")
print(f"     Exc voltage: mean={exc_layer.v.mean().item():.2f}, min={exc_layer.v.min().item():.2f}, max={exc_layer.v.max().item():.2f}")

encoder.network.run(inputs=inputs, time=int(encoder.simulation_time))

# Check excitatory layer state after
print(f"   After simulation:")
print(f"     Exc voltage: mean={exc_layer.v.mean().item():.2f}, min={exc_layer.v.min().item():.2f}, max={exc_layer.v.max().item():.2f}")

# Get spikes
spikes = encoder.network.monitors["ExcitatoryMonitor"].get("s")
print(f"\n4. Checking output spikes...")
print(f"   Spike tensor shape: {spikes.shape}")
print(f"   Total spikes: {spikes.sum().item()}")
print(f"   Neurons that spiked: {(spikes.sum(dim=0) > 0).sum().item()}/{spikes.shape[1]}")

if spikes.sum() < 1:
    print(f"   ❌ PROBLEM: No output spikes!")
    print(f"\n5. Investigating why no spikes...")
    
    # Check if neurons can spike at all
    print(f"   Neuron thresholds: {exc_layer.thresh.mean().item():.2f}")
    print(f"   Neuron rest potential: {exc_layer.rest.mean().item():.2f}")
    print(f"   Max voltage reached: {exc_layer.v.max().item():.2f}")
    
    if exc_layer.v.max() < exc_layer.thresh.min():
        print(f"   ❌ Voltages never reached threshold!")
        print(f"      This could be because:")
        print(f"      - Input spike rate too low")
        print(f"      - Weights too small")
        print(f"      - Threshold too high")
else:
    print(f"   ✓ Output spikes detected!")

encoder.network.reset_state_variables()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
