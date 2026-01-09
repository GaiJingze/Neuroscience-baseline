# Troubleshooting Guide

Common issues and solutions for the clustering pipeline.

## Installation Issues

### Issue 1: BindsNET Import Error - NumPy Version

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
ModuleNotFoundError: No module named 'torch._six'
```

**Cause**: BindsNET is not compatible with NumPy 2.x

**Solution**:
```bash
# Downgrade NumPy to 1.x
pip install "numpy<2.0.0" --force-reinstall

# Reinstall BindsNET
pip install bindsnet --no-cache-dir

# Verify
python -c "import bindsnet; print('OK')"
```

### Issue 2: PyTorch CUDA Mismatch

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**:
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip uninstall torch torchvision -y
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: Scikit-learn-extra Not Found

```
ModuleNotFoundError: No module named 'sklearn_extra'
```

**Solution**:
```bash
pip install scikit-learn-extra
```

## Runtime Issues

### Issue 4: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions**:
1. **Use CPU instead**:
   ```yaml
   # In config file
   device: "cpu"
   ```

2. **Reduce model size**:
   ```yaml
   encoder_config:
     n_neurons: 200  # Instead of 400
   ```

3. **Train on subset**:
   ```bash
   python train.py --n_train 1000  # Instead of all data
   ```

### Issue 5: Slow Training

BindsNET simulations can be very slow.

**Solutions**:
1. **Use GPU**: Make sure CUDA is available
2. **Reduce simulation time**: `simulation_time: 250` instead of 350
3. **Fewer neurons**: `n_neurons: 200` instead of 400
4. **Train on subset**: `--n_train 5000`

### Issue 6: Dataset Download Fails

```
HTTPError: 403 Forbidden
```

**Solution**:
```bash
# Manual download for MNIST
mkdir -p data/mnist/MNIST/raw
cd data/mnist/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

### Issue 7: FlyHash Package Conflicts

If using the PyPI FlyHash package:

```
ImportError: cannot import name 'FlyHash'
```

**Solution**: Use our implementation instead:
```yaml
# In config
encoder_config:
  use_package: false  # Use manual implementation
```

## Configuration Issues

### Issue 8: Config File Not Found

```
FileNotFoundError: configs/xxx.yaml
```

**Solution**:
```bash
# Make sure you're in the right directory
cd clustering

# Check config exists
ls configs/

# Use absolute path
python scripts/run_baseline.py --config /full/path/to/config.yaml
```

### Issue 9: Wrong Device Setting

```
AssertionError: Torch not compiled with CUDA enabled
```

**Solution**:
```yaml
# In config file, change to:
device: "cpu"
```

## Data Issues

### Issue 10: SIFT1M Files Corrupt

```
struct.error: unpack requires a buffer of X bytes
```

**Solution**:
```bash
# Re-download
cd data/sift1m
rm -rf *
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -zxvf sift.tar.gz
```

### Issue 11: Data Shape Mismatch

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Solution**: Check input dimensions match config:
```yaml
encoder_config:
  input_dim: 784  # Must match flattened image size
```

## Evaluation Issues

### Issue 12: NaN in Metrics

```
Warning: NaN values in clustering results
```

**Causes**:
- All zeros in feature codes
- Empty clusters
- Division by zero

**Solutions**:
```bash
# Check feature statistics
python -c "
import numpy as np
codes = np.load('outputs/codes/.../code_seed0.npy')
print('Mean:', np.mean(codes))
print('Std:', np.std(codes))
print('Zeros:', np.mean(codes == 0))
"

# If all zeros, check encoder output
```

### Issue 13: Clustering Takes Forever

k-medoids with Hamming distance on large datasets is slow.

**Solutions**:
```python
# Use subset for testing
codes_subset = codes[:1000]

# Or use faster algorithm
clustering_methods: ['kmeans']  # Instead of kmedoids
```

## Permission Issues

### Issue 14: Permission Denied

```
PermissionError: [Errno 13] Permission denied: 'outputs/...'
```

**Solution**:
```bash
# Fix permissions
chmod -R u+w outputs/
mkdir -p outputs/codes outputs/results outputs/logs
```

## Environment Issues

### Issue 15: Wrong Python Version

```
SyntaxError: invalid syntax (using := operator)
```

**Solution**:
```bash
# Check Python version (need 3.9+)
python --version

# Create new environment with correct version
conda create -n clustering_pipeline python=3.9
conda activate clustering_pipeline
pip install -r requirements.txt
```

### Issue 16: Module Not Found After Installation

```
ModuleNotFoundError: No module named 'pipeline'
```

**Solution**:
```bash
# Make sure you're in the project directory
cd clustering

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or run from clustering/ directory
python scripts/run_baseline.py ...
```

## Performance Issues

### Issue 17: Training Doesn't Converge

STDP learning is sensitive to parameters.

**Solutions**:
1. **Adjust learning rates**:
   ```yaml
   nu: [1.0e-5, 5.0e-3]  # Try different values
   ```

2. **Check input normalization**:
   ```python
   # Data should be in [0, 1]
   data = data / 255.0 if data.max() > 1.0 else data
   ```

3. **Increase simulation time**:
   ```yaml
   simulation_time: 500  # Give neurons more time to spike
   ```

### Issue 18: Silent Network (No Spikes)

All spike counts are zero.

**Diagnosis**:
```python
# Check if network is spiking
python baselines/diehl_cook/train.py --train --n_train 10
# Look for spike statistics in output
```

**Solutions**:
- Increase input intensity
- Lower threshold: `thresh: -55.0`
- Check Poisson encoding rate
- Verify network connections

## Debugging Tips

### General Debugging

```bash
# 1. Test with minimal data
python scripts/run_baseline.py --config configs/xxx.yaml --n_train 10

# 2. Use CPU for debugging
device: "cpu"

# 3. Enable verbose output
verbose: true

# 4. Check intermediate outputs
ls outputs/codes/
python -c "import numpy as np; print(np.load('outputs/codes/.../code.npy').shape)"

# 5. Run tests
python tests/test_pipeline.py
```

### Getting Help

If issues persist:

1. Check error message carefully
2. Search this troubleshooting guide
3. Check GitHub issues
4. Review documentation in `docs/`
5. Ask on project discussion forum
6. Contact maintainer: Jingze Gai

## Quick Fixes Summary

| Error | Quick Fix |
|-------|-----------|
| NumPy 2.x | `pip install "numpy<2.0"` |
| No CUDA | Set `device: "cpu"` |
| Out of memory | Reduce `n_neurons` |
| Slow training | Use subset with `--n_train` |
| Module not found | `pip install -r requirements.txt` |
| Permission denied | `chmod -R u+w outputs/` |
| Wrong directory | `cd clustering` |

---

**Last Updated**: 2026-01-09  
**Version**: 1.0
