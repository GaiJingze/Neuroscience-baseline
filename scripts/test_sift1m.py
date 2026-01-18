#!/usr/bin/env python
"""
Test SIFT1M dataset loading and basic operations.

Usage:
    python scripts/test_sift1m.py
    python scripts/test_sift1m.py --subset 1000
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.datasets import load_dataset


def test_sift1m_loading(subset_size=10000):
    """Test SIFT1M dataset loading."""
    print("=" * 70)
    print("Testing SIFT1M Dataset Loading")
    print("=" * 70)
    print(f"Subset size: {subset_size}")
    print()
    
    try:
        # Load dataset
        print("Loading SIFT1M dataset...")
        dataset = load_dataset('sift1m', root='./data', subset_size=subset_size)
        
        # Check data
        print("✅ Dataset loaded successfully!")
        print()
        print("Dataset information:")
        print(f"  Train data shape: {dataset['train_data'].shape}")
        print(f"  Test data shape: {dataset['test_data'].shape}")
        print(f"  Train labels: {dataset['train_labels']}")
        print(f"  Test labels: {dataset['test_labels']}")
        
        if dataset.get('query_data') is not None:
            print(f"  Query data shape: {dataset['query_data'].shape}")
        if dataset.get('groundtruth') is not None:
            print(f"  Ground truth shape: {dataset['groundtruth'].shape}")
        
        # Statistics
        train_data = dataset['train_data']
        print()
        print("Data statistics:")
        print(f"  Dimension: {train_data.shape[1]}")
        print(f"  Mean: {train_data.mean():.4f}")
        print(f"  Std: {train_data.std():.4f}")
        print(f"  Min: {train_data.min():.4f}")
        print(f"  Max: {train_data.max():.4f}")
        
        # Check for NaN/Inf
        print()
        print("Data quality:")
        print(f"  Contains NaN: {np.isnan(train_data).any()}")
        print(f"  Contains Inf: {np.isinf(train_data).any()}")
        
        # Sample vectors
        print()
        print("Sample vectors (first 3):")
        for i in range(min(3, len(train_data))):
            vec = train_data[i]
            print(f"  Vector {i}: mean={vec.mean():.2f}, std={vec.std():.2f}, "
                  f"norm={np.linalg.norm(vec):.2f}")
        
        print()
        print("=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        
        return True
        
    except FileNotFoundError as e:
        print()
        print("=" * 70)
        print("❌ SIFT1M dataset not found!")
        print("=" * 70)
        print()
        print("Error:", str(e))
        print()
        print("To download SIFT1M dataset, run:")
        print("  bash scripts/download_sift1m.sh")
        print()
        return False
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Error loading SIFT1M dataset")
        print("=" * 70)
        print()
        print("Error:", str(e))
        print()
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SIFT1M dataset loading")
    parser.add_argument('--subset', type=int, default=10000,
                        help='Subset size for testing (default: 10000)')
    
    args = parser.parse_args()
    
    success = test_sift1m_loading(args.subset)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
