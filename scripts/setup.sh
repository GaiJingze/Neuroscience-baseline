#!/bin/bash
# Setup script for clustering pipeline

set -e

echo "=========================================="
echo "Clustering Pipeline Setup"
echo "=========================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python --version

# Create conda environment (optional)
read -p "Create new conda environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating conda environment 'clustering_pipeline'..."
    conda create -n clustering_pipeline python=3.9 -y
    echo "Activate with: conda activate clustering_pipeline"
    echo "Then re-run this script to install packages."
    exit 0
fi

# Install requirements
echo ""
echo "[2/5] Installing Python packages..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "[3/5] Creating directories..."
mkdir -p data outputs/codes outputs/results outputs/logs notebooks

# Download MNIST (auto-download on first use)
echo ""
echo "[4/5] Testing MNIST download..."
python -c "from torchvision import datasets; datasets.MNIST('./data', download=True)" || echo "MNIST download failed (will retry on first use)"

# Test imports
echo ""
echo "[5/5] Testing imports..."
python -c "
import numpy as np
import torch
import sklearn
from pipeline import load_dataset, set_seed
print('All imports successful!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download SIFT1M (optional): bash scripts/download_sift1m.sh"
echo "2. Run a baseline: python scripts/run_baseline.py --config configs/flyhash.yaml"
echo "3. Check outputs: ls outputs/results/"
