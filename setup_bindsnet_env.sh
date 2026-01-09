#!/bin/bash
# Setup script for BindsNET environment (isolated)

set -e

echo "=========================================="
echo "BindsNET Environment Setup"
echo "=========================================="
echo ""
echo "This creates an ISOLATED environment for BindsNET"
echo "to avoid version conflicts with the main pipeline."
echo ""

ENV_NAME="bindsnet_env"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create environment
echo "[1/5] Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# Activate environment (note: might need manual activation in some shells)
echo ""
echo "[2/5] Activating environment..."
echo "Note: You may need to manually activate with:"
echo "  conda activate $ENV_NAME"
echo ""

# Install PyTorch 1.13 (compatible with BindsNET)
echo "[3/5] Installing PyTorch 1.13..."
conda run -n $ENV_NAME pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# Install NumPy 1.24
echo ""
echo "[4/5] Installing NumPy 1.24..."
conda run -n $ENV_NAME pip install numpy==1.24.3

# Install BindsNET
echo ""
echo "[5/5] Installing BindsNET..."
conda run -n $ENV_NAME pip install bindsnet

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
conda run -n $ENV_NAME python -c "
import torch
import numpy
import bindsnet
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ BindsNET: {bindsnet.__version__}')
print('')
print('All packages installed successfully!')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use BindsNET:"
echo "  1. Activate environment: conda activate $ENV_NAME"
echo "  2. Run training: python baselines/diehl_cook/train.py --train"
echo "  3. Features will be saved to outputs/codes/"
echo "  4. Switch back to main env for evaluation"
echo ""
echo "For main pipeline (without BindsNET training):"
echo "  Use your main environment"
echo "  The encoder interface will work with placeholder training"
echo ""
