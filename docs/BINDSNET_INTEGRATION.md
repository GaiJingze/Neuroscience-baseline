# BindsNET Integration Guide

This document explains how BindsNET is integrated into the clustering pipeline.

## Overview

**BindsNET** is a PyTorch-based Spiking Neural Network (SNN) simulation library used to implement the Diehl & Cook (2015) STDP baseline.

- **Version**: >= 0.3.1
- **GitHub**: https://github.com/BindsNET/bindsnet
- **Documentation**: https://bindsnet-docs.readthedocs.io/
- **Paper**: Hazan et al., "BindsNET: A machine learning-oriented spiking neural networks library in Python", Frontiers in Neuroinformatics, 2018

## Installation

### Quick Install

```bash
pip install bindsnet
```

### Verify Installation

```bash
python -c "import bindsnet; print(bindsnet.__version__)"
```

Expected output: `0.3.1` or higher

## Integration Architecture

### 1. Encoder Interface

BindsNET is integrated through the `DiehlCookEncoder` class:

```
clustering/
├── baselines/
│   └── diehl_cook/
│       ├── encoder.py          # Encoder interface (inherits BaseEncoder)
│       ├── train.py            # Full BindsNET training script
│       └── README.md           # Detailed documentation
```

### 2. Two-Level Integration

We provide **two ways** to use BindsNET:

#### Level 1: Main Pipeline (Simplified)

For quick baseline evaluation:

```bash
python scripts/run_baseline.py --config configs/diehl_cook.yaml
```

This uses `encoder.py` which provides:
- ✅ Simplified interface
- ✅ Compatible with pipeline
- ⚠️ Placeholder training (for now)

#### Level 2: Full Training Script

For complete STDP training:

```bash
python baselines/diehl_cook/train.py \
    --train --extract \
    --n_train 5000 \
    --n_epochs 1 \
    --device cuda
```

This uses `train.py` which provides:
- ✅ Full BindsNET network construction
- ✅ STDP learning with adaptive thresholds
- ✅ Lateral inhibition (Winner-Take-All)
- ✅ Spike count extraction
- ✅ Model saving/loading

## Network Architecture

The Diehl & Cook network implemented with BindsNET:

```
Input Layer (784 neurons, Poisson encoding)
    ↓
Excitatory Layer (400 LIF neurons, STDP learning)
    ↓ ↑
Inhibitory Layer (400 LIF neurons, lateral inhibition)
```

### Key Components

1. **Input Layer**:
   - Type: `Input` nodes
   - Encoding: Poisson rate coding
   - Function: Convert pixel intensities to spike trains

2. **Excitatory Layer**:
   - Type: `LIFNodes` (Leaky Integrate-and-Fire)
   - Count: 400 neurons
   - Features:
     - STDP learning
     - Adaptive threshold (homeostasis)
     - Spike traces for learning

3. **Inhibitory Layer**:
   - Type: `LIFNodes`
   - Count: 400 neurons
   - Function: Winner-Take-All competition

4. **Connections**:
   - Input → Excitatory: STDP learning (`PostPre` rule)
   - Excitatory → Inhibitory: One-to-one excitation
   - Inhibitory → Excitatory: All-to-all inhibition (except self)

## Usage Examples

### Example 1: Quick Test

```bash
# Test the encoder interface
cd clustering
python baselines/diehl_cook/encoder.py
```

### Example 2: Train on Subset

```bash
# Train on 1000 samples for quick validation
python baselines/diehl_cook/train.py \
    --train \
    --n_train 1000 \
    --n_epochs 1 \
    --device cpu
```

### Example 3: Full Training

```bash
# Full training (requires GPU, takes hours)
python baselines/diehl_cook/train.py \
    --train --extract --save \
    --dataset mnist \
    --n_neurons 400 \
    --n_epochs 1 \
    --time 350 \
    --device cuda \
    --output_dir ./outputs
```

### Example 4: Extract Features Only

```bash
# Load pre-trained model and extract spike counts
python baselines/diehl_cook/train.py \
    --extract \
    --load outputs/models/diehl_cook_seed0.pt \
    --device cuda
```

### Example 5: Pipeline Integration

```bash
# Run through main evaluation pipeline
python scripts/run_baseline.py \
    --config configs/diehl_cook.yaml \
    --seed 0
```

## Parameters

### Network Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_neurons` | 400 | 100-1000 | Excitatory neurons |
| `simulation_time` | 350 ms | 100-500 | Time per sample |
| `dt` | 1.0 ms | 0.1-1.0 | Simulation timestep |
| `thresh` | -52.0 mV | -60 to -50 | LIF threshold |
| `nu[0]` | 1e-4 | 1e-5 to 1e-3 | Pre-synaptic learning rate |
| `nu[1]` | 1e-2 | 1e-3 to 1e-1 | Post-synaptic learning rate |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 1 | Training epochs |
| `n_train` | All | Training samples (None=all) |
| `device` | cuda | 'cuda' or 'cpu' |

### Tuning Tips

1. **For faster training**:
   - Reduce `simulation_time` (e.g., 250 ms)
   - Reduce `n_neurons` (e.g., 200)
   - Train on subset: `--n_train 5000`

2. **For better performance**:
   - Increase `n_neurons` (e.g., 800)
   - Increase `simulation_time` (e.g., 500 ms)
   - Multiple epochs: `--n_epochs 3`

3. **For debugging**:
   - Use `--n_train 100` for quick test
   - Use CPU: `--device cpu`
   - Check spike statistics in output

## Output Format

### Spike Counts

The network outputs spike count vectors:

```python
# Shape: (n_samples, n_neurons)
spike_counts = np.array([
    [12, 5, 0, 8, ...],  # Sample 1: spike counts per neuron
    [3, 15, 2, 0, ...],  # Sample 2
    ...
])
```

These are saved as:
- `outputs/codes/diehl_cook/mnist/pre_code_seed0.npy`

### Binarization

For clustering/hashing, spike counts are binarized:

```python
# Top-5% neurons with highest spike counts
code = top_k_percent_binarization(spike_counts, percent=0.05)
```

Saved as:
- `outputs/codes/diehl_cook/mnist/code_seed0.npy`

## Performance Expectations

### Original Paper (Diehl & Cook 2015)

- **Classification accuracy**: ~95% (with linear SVM)
- **Training time**: Several hours on CPU
- **Dataset**: MNIST

### Our Implementation

**Goals** (as baseline):
- Achieve ~85-90% of paper performance
- Extract meaningful features for clustering
- Demonstrate bio-plausible learning

**Expected Results**:
- NMI: 0.6-0.7 (clustering on MNIST)
- Training time: 2-4 hours (GPU), 8-12 hours (CPU)

## Troubleshooting

### Issue 1: Import Error

```
ImportError: No module named 'bindsnet'
```

**Solution**:
```bash
pip install bindsnet
```

### Issue 2: CUDA Error

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce `n_neurons`: `--n_neurons 200`
- Use CPU: `--device cpu`
- Reduce batch size in code

### Issue 3: Slow Training

BindsNET simulations can be slow.

**Solutions**:
- Use GPU: `--device cuda`
- Train on subset: `--n_train 5000`
- Reduce simulation time: `--time 250`
- Reduce neurons: `--n_neurons 200`

### Issue 4: Silent Network (No Spikes)

Check spike statistics in output. If all zeros:

**Solutions**:
- Check input encoding (should be in [0, 1])
- Increase `simulation_time`
- Check threshold value
- Verify network construction

### Issue 5: Version Conflicts

```
ERROR: incompatible versions
```

**Solution**:
```bash
# Uninstall
pip uninstall torch bindsnet -y

# Reinstall in order
pip install torch==2.0.1
pip install bindsnet
```

## Comparison with Other SNN Frameworks

| Feature | BindsNET | SpikingJelly | Norse |
|---------|----------|--------------|-------|
| Backend | PyTorch | PyTorch | PyTorch |
| STDP Support | ✅ Built-in | ⚠️ Manual | ⚠️ Manual |
| Documentation | ✅ Excellent | ✅ Good | ⚠️ Limited |
| Examples | ✅ Many | ✅ Many | ⚠️ Few |
| Maintenance | ✅ Active | ✅ Active | ⚠️ Less active |
| **Ease of Use** | 🏆 Best for STDP | Good | Complex |

**Why BindsNET for Diehl & Cook?**
- Built-in STDP learning rules
- Good documentation and examples
- Active community
- Designed for neuroscience-inspired learning

## Further Reading

1. **BindsNET Documentation**: https://bindsnet-docs.readthedocs.io/
2. **BindsNET Examples**: https://github.com/BindsNET/bindsnet/tree/master/examples
3. **Diehl & Cook Paper**: https://doi.org/10.3389/fncom.2015.00099
4. **BindsNET Paper**: Hazan et al., Frontiers in Neuroinformatics, 2018

## Contributing

If you improve the BindsNET integration:
1. Update `baselines/diehl_cook/train.py`
2. Update this documentation
3. Add tests if applicable
4. Submit changes to the team

---

**Status**: ✅ BindsNET integrated and ready to use  
**Last Updated**: 2026-01-09  
**Maintainer**: Jingze Gai
