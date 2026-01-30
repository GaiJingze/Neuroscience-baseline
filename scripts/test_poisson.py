#!/usr/bin/env python3
"""Test Poisson encoding parameters"""

import torch
from bindsnet.encoding import poisson

# Test with different input intensities
print("Testing Poisson encoding with different input intensities:")
print("="*80)

for intensity in [0.1, 0.5, 1.0, 10.0, 100.0, 255.0]:
    data = torch.ones(784) * intensity
    encoded = poisson(datum=data, time=350, dt=1.0)
    
    total_spikes = encoded.sum().item()
    spikes_per_neuron = total_spikes / 784
    
    print(f"\nIntensity: {intensity:6.1f}")
    print(f"  Total spikes: {total_spikes:.0f}")
    print(f"  Spikes per neuron: {spikes_per_neuron:.2f}")
    print(f"  Neurons active: {(encoded.sum(dim=0) > 0).sum().item()}/784")

print("\n" + "="*80)
print("Recommended: Input data should be scaled to 0-255 range for Poisson encoding")
print("="*80)
