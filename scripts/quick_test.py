#!/usr/bin/env python
"""
Quick test script - runs minimal tests to verify installation.
Can be run directly: python scripts/quick_test.py
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    tests = [
        ("NumPy", "import numpy"),
        ("PyTorch", "import torch"),
        ("Scikit-learn", "import sklearn"),
        ("SciPy", "import scipy"),
        ("Matplotlib", "import matplotlib"),
        ("YAML", "import yaml"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    
    return passed, failed


def test_pipeline_modules():
    """Test pipeline modules."""
    print("\nTesting pipeline modules...")
    
    passed = 0
    failed = 0
    
    # Test imports
    try:
        from pipeline import (
            load_dataset, 
            compute_nmi, 
            compute_ari,
            top_k_binarization,
            set_seed
        )
        print("  ✓ Pipeline imports")
        passed += 1
    except Exception as e:
        print(f"  ✗ Pipeline imports: {e}")
        failed += 1
        return passed, failed
    
    # Test set_seed
    try:
        set_seed(42)
        print("  ✓ set_seed()")
        passed += 1
    except Exception as e:
        print(f"  ✗ set_seed(): {e}")
        failed += 1
    
    # Test binarization
    try:
        import numpy as np
        features = np.random.randn(10, 20)
        binary = top_k_binarization(features, k=5)
        assert binary.shape == (10, 20)
        assert np.all(np.sum(binary, axis=1) == 5)
        print("  ✓ top_k_binarization()")
        passed += 1
    except Exception as e:
        print(f"  ✗ top_k_binarization(): {e}")
        failed += 1
    
    # Test metrics
    try:
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([1, 1, 0, 0, 2, 2])
        nmi = compute_nmi(labels_true, labels_pred)
        ari = compute_ari(labels_true, labels_pred)
        assert 0 <= nmi <= 1
        assert -1 <= ari <= 1
        print("  ✓ Clustering metrics")
        passed += 1
    except Exception as e:
        print(f"  ✗ Clustering metrics: {e}")
        failed += 1
    
    return passed, failed


def test_baselines():
    """Test baseline encoders."""
    print("\nTesting baseline encoders...")
    
    passed = 0
    failed = 0
    
    # Test FlyHash
    try:
        from baselines.flyhash.encoder import FlyHashEncoder
        import numpy as np
        
        config = {
            'input_dim': 100,
            'projection_dim': 250,
            'hash_length': 10,
            'use_package': False
        }
        encoder = FlyHashEncoder(config)
        encoder.fit(np.random.randn(50, 100))
        result = encoder.encode(np.random.randn(10, 100))
        
        assert 'pre_code' in result
        assert 'code' in result
        assert result['code'].shape == (10, 250)
        
        print("  ✓ FlyHash encoder")
        passed += 1
    except Exception as e:
        print(f"  ✗ FlyHash encoder: {e}")
        traceback.print_exc()
        failed += 1
    
    # Test base encoder
    try:
        from baselines.base_encoder import DummyEncoder
        
        config = {'input_dim': 100, 'code_dim': 50}
        encoder = DummyEncoder(config)
        encoder.fit(np.random.randn(20, 100))
        result = encoder.encode(np.random.randn(10, 100))
        
        assert result['code'].shape == (10, 50)
        
        print("  ✓ Base encoder")
        passed += 1
    except Exception as e:
        print(f"  ✗ Base encoder: {e}")
        failed += 1
    
    return passed, failed


def test_data_loading():
    """Test data loading (may download MNIST)."""
    print("\nTesting data loading...")
    
    passed = 0
    failed = 0
    
    try:
        from pipeline.datasets import load_dataset
        
        print("  Downloading MNIST (this may take a moment)...")
        dataset = load_dataset('mnist', root='./data')
        
        assert 'train_data' in dataset
        assert 'test_data' in dataset
        assert dataset['train_data'].shape[1] == 784
        
        print(f"  ✓ MNIST loaded ({len(dataset['test_data'])} test samples)")
        passed += 1
    except Exception as e:
        print(f"  ✗ MNIST loading: {e}")
        failed += 1
    
    return passed, failed


def main():
    """Run all tests."""
    print("="*60)
    print("Clustering Pipeline - Quick Test")
    print("="*60)
    print()
    
    total_passed = 0
    total_failed = 0
    
    # Run test suites
    p, f = test_imports()
    total_passed += p
    total_failed += f
    
    p, f = test_pipeline_modules()
    total_passed += p
    total_failed += f
    
    p, f = test_baselines()
    total_passed += p
    total_failed += f
    
    p, f = test_data_loading()
    total_passed += p
    total_failed += f
    
    # Summary
    print()
    print("="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total tests: {total_passed + total_failed}")
    print(f"✓ Passed: {total_passed}")
    print(f"✗ Failed: {total_failed}")
    print()
    
    if total_failed == 0:
        print("All tests passed! ✓")
        print()
        print("Next steps:")
        print("  1. Run a baseline: python scripts/run_baseline.py --config configs/flyhash.yaml")
        print("  2. Check documentation: cat README.md")
        return 0
    else:
        print("Some tests failed ✗")
        print()
        print("Troubleshooting:")
        print("  1. Check INSTALL.md for installation instructions")
        print("  2. Verify dependencies: pip install -r requirements.txt")
        print("  3. See TROUBLESHOOTING.md for common issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())
