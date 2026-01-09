# Installation Guide

Complete installation guide for the Clustering/Hashing pipeline.

## Quick Install

```bash
# 1. Create environment
conda create -n clustering_pipeline python=3.9 -y
conda activate clustering_pipeline

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch, bindsnet, sklearn, numpy; print('All dependencies installed!')"
```

---

## Detailed Installation

### Step 1: Python Environment

We recommend using **conda** for environment management:

```bash
# Create environment with Python 3.9
conda create -n clustering_pipeline python=3.9 -y

# Activate
conda activate clustering_pipeline

# Verify Python version
python --version  # Should be 3.9.x
```

### Step 2: PyTorch Installation

Install PyTorch based on your system:

#### With CUDA (GPU available)

```bash
# For CUDA 11.8 (check your CUDA version with: nvidia-smi)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### CPU Only

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Core Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

This installs:
- NumPy, SciPy (numerical computing)
- Scikit-learn (clustering, metrics)
- BindsNET (SNN framework)
- Matplotlib, Seaborn (visualization)
- PyYAML (config files)

### Step 4: Optional Dependencies

#### FAISS (for fast similarity search)

```bash
# CPU version
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

#### FlyHash Package

```bash
# If you want to use the PyPI package instead of our implementation
pip install FlyHash
```

#### Jupyter (for notebooks)

```bash
pip install jupyter ipython
```

---

## Verifying Installation

### 1. Test Core Imports

```bash
python -c "
import numpy as np
import torch
import sklearn
print('✓ Core dependencies OK')
"
```

### 2. Test BindsNET

```bash
python -c "
import bindsnet
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
print(f'✓ BindsNET {bindsnet.__version__} OK')
"
```

### 3. Test Pipeline Modules

```bash
cd clustering
python tests/test_pipeline.py
```

Expected output:
```
Running pipeline tests...
Clustering metrics test passed: NMI=...
Retrieval metrics test passed: mAP=...
...
All tests passed! ✓
```

### 4. Test Dataset Loading

```bash
python pipeline/datasets.py
```

This will download MNIST and test the data loaders.

### 5. Test a Baseline

```bash
# Test FlyHash (quick)
python baselines/flyhash/encoder.py

# Test Diehl & Cook skeleton
python baselines/diehl_cook/encoder.py
```

---

## Troubleshooting

### Issue: BindsNET Import Error

```
ImportError: No module named 'bindsnet'
```

**Solution**:
```bash
pip install bindsnet
```

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Use CPU instead: Set `device: "cpu"` in config files
2. Reduce batch size or number of neurons
3. Free up GPU memory: `nvidia-smi` to check usage

### Issue: PyTorch Version Conflict

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution**:
```bash
# Uninstall conflicting packages
pip uninstall torch torchvision -y

# Reinstall with specific versions
pip install torch==2.0.1 torchvision==0.15.2

# Then install BindsNET
pip install bindsnet
```

### Issue: Scikit-learn-extra Not Found

```
ModuleNotFoundError: No module named 'sklearn_extra'
```

**Solution**:
```bash
pip install scikit-learn-extra
```

### Issue: Permission Denied (Linux/Mac)

```bash
# Use --user flag
pip install --user -r requirements.txt
```

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

May need system packages:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
```

### macOS

May need Xcode command line tools:
```bash
xcode-select --install
```

### Windows

- Use **Anaconda Prompt** instead of Command Prompt
- Some packages may need Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## Docker Installation (Alternative)

If you prefer Docker:

```bash
# Build image
docker build -t clustering_pipeline .

# Run container
docker run -it --gpus all -v $(pwd):/workspace clustering_pipeline bash
```

(Note: Dockerfile not included yet, but can be created if needed)

---

## Development Installation

For development (with testing tools):

```bash
pip install -r requirements.txt
pip install pytest black flake8 ipython
```

---

## GPU Requirements

| Task | Minimum GPU | Recommended GPU | Can Use CPU? |
|------|-------------|-----------------|--------------|
| FlyHash | N/A | N/A | ✅ Yes (CPU only) |
| Diehl & Cook (small) | 8GB | 16GB | ✅ Yes (slow) |
| Diehl & Cook (full) | 16GB | 24GB | ⚠️ Very slow |
| SoftHebb | 16GB | 24GB | ⚠️ Very slow |

---

## Next Steps

After successful installation:

1. **Download datasets**: `bash scripts/download_sift1m.sh` (optional)
2. **Run setup script**: `bash scripts/setup.sh`
3. **Test pipeline**: `python scripts/run_baseline.py --config configs/flyhash.yaml`
4. **Read documentation**: See `README.md` and `docs/`

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review GitHub issues: [Project Issues](link_to_issues)
3. Ask on discussion forum
4. Contact: Jingze Gai

---

**Last Updated**: 2026-01-09
