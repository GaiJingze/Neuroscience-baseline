# Diehl & Cook (2015) STDP Baseline

## Overview

Implementation of the STDP-based unsupervised learning method from:

> **"Unsupervised learning of digit recognition using spike-timing-dependent plasticity"**  
> Peter U. Diehl and Matthew Cook  
> Frontiers in Computational Neuroscience, 2015  
> DOI: [10.3389/fncom.2015.00099](https://doi.org/10.3389/fncom.2015.00099)

## Architecture

- **Input layer**: Poisson rate-coded neurons (784 for MNIST)
- **Excitatory layer**: LIF neurons (400) with STDP learning
- **Inhibitory layer**: Lateral inhibition (Winner-Take-All)
- **Feature extraction**: Spike count vectors

## Implementation Status

✅ **Framework**: BindsNET  
🔨 **Status**: Partial implementation (skeleton ready, full STDP training in progress)

## Installation

```bash
# Install BindsNET
pip install bindsnet

# Verify installation
python -c "import bindsnet; print(bindsnet.__version__)"
```

## Usage

### Quick Test

```bash
# Test the encoder skeleton
python baselines/diehl_cook/encoder.py
```

### Full Training (TODO)

```bash
# Train on MNIST
python baselines/diehl_cook/train.py --dataset mnist --n_neurons 400 --n_epochs 1

# Encode test data
python baselines/diehl_cook/extract_features.py --model saved_models/diehl_cook.pt
```

### Integration with Pipeline

```bash
# Run through main pipeline
python scripts/run_baseline.py --config configs/diehl_cook.yaml
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neurons` | 400 | Number of excitatory neurons |
| `simulation_time` | 350 ms | Simulation time per sample |
| `dt` | 1.0 ms | Time step |
| `thresh` | -52.0 mV | LIF threshold voltage |
| `nu` | (1e-4, 1e-2) | STDP learning rates (pre, post) |

## Implementation Notes

### Current Status

The current implementation provides:
- ✅ Encoder interface compatible with pipeline
- ✅ Basic BindsNET network structure
- ⚠️ Simplified training (placeholder)
- ⚠️ Spike count feature extraction (simplified)

### Next Steps

1. **Complete BindsNET Integration**:
   - Implement full STDP learning loop
   - Add adaptive threshold mechanism
   - Implement proper lateral inhibition

2. **Feature Extraction**:
   - Record spike counts per neuron
   - Implement proper spike train encoding

3. **Training Loop**:
   - Iterate over MNIST training samples
   - Update weights via STDP
   - Save learned weights

### Reference Implementation

BindsNET provides examples in their repository:
- `examples/mnist/` - Basic MNIST examples
- Check: https://github.com/BindsNET/bindsnet/tree/master/examples

## Expected Results

From the original paper:
- **Classification accuracy**: ~95% (with linear SVM on spike counts)
- **Training time**: Several hours on CPU

Our goal (for baseline):
- Achieve ~85-90% of paper performance
- Extract meaningful spike count features for clustering

## Troubleshooting

### Import Error

```python
ImportError: No module named 'bindsnet'
```

**Solution**: Install BindsNET: `pip install bindsnet`

### CUDA Error

```
RuntimeError: CUDA out of memory
```

**Solution**: 
- Reduce batch size
- Use CPU: Set `device: "cpu"` in config
- Reduce `n_neurons` parameter

### Slow Training

BindsNET training can be slow. Tips:
- Use GPU if available
- Reduce simulation time
- Train on subset of data first

## References

1. Diehl & Cook (2015) - Original paper
2. BindsNET Documentation: https://bindsnet-docs.readthedocs.io/
3. BindsNET GitHub: https://github.com/BindsNET/bindsnet

## TODO

- [ ] Implement full STDP training loop
- [ ] Add weight visualization
- [ ] Optimize training speed
- [ ] Validate against paper results
- [ ] Add pre-trained weights (optional)
