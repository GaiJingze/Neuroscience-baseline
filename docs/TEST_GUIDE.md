# Testing Guide

Complete guide to testing the clustering pipeline.

## Quick Start

### One-Click Testing

```bash
# Option 1: Python script (recommended)
python scripts/quick_test.py

# Option 2: Bash script
bash scripts/run_tests.sh

# Option 3: Makefile
make test
```

All three methods run the same tests and report results.

---

## Test Levels

### Level 1: Quick Test (1-2 minutes)

Verifies basic functionality:

```bash
python scripts/quick_test.py
```

**Tests**:
- ✓ Import all dependencies
- ✓ Pipeline modules work
- ✓ Baseline encoders functional
- ✓ Data loading works

**Use when**:
- After installation
- Quick verification
- Before committing code

### Level 2: Unit Tests (2-5 minutes)

Comprehensive module testing:

```bash
python tests/test_pipeline.py
```

**Tests**:
- Clustering metrics (NMI, ARI, ACC)
- Retrieval metrics (mAP, Recall@K)
- Binarization methods
- Seed reproducibility

**Use when**:
- Developing new features
- Debugging issues
- Before pull requests

### Level 3: Integration Tests (10-30 minutes)

End-to-end pipeline testing:

```bash
# Run a complete baseline
python scripts/run_baseline.py --config configs/flyhash.yaml

# Or test all baselines
bash scripts/run_all_baselines.sh
```

**Tests**:
- Full encode → cluster → evaluate flow
- Multiple seeds
- All datasets

**Use when**:
- Before important milestones
- Performance validation
- Preparing results

---

## Test Commands

### Using Python

```bash
# Quick test
python scripts/quick_test.py

# Unit tests
python tests/test_pipeline.py

# Individual module tests
python pipeline/datasets.py
python pipeline/metrics.py
python baselines/flyhash/encoder.py
```

### Using Bash

```bash
# Full test suite
bash scripts/run_tests.sh

# Run all baselines
bash scripts/run_all_baselines.sh
```

### Using Makefile

```bash
# Show all options
make help

# Run tests
make test           # Full suite
make quick-test     # Quick only

# Run baselines
make run-flyhash    # Single baseline
make run-all        # All baselines

# Check status
make status         # Show what's installed
make check-env      # Check environment
```

### Using Pytest (if installed)

```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/ -v                    # Verbose
pytest tests/ --tb=short            # Short traceback
pytest tests/test_pipeline.py::test_clustering_metrics  # Specific test
```

---

## What Each Test Does

### quick_test.py

```
✓ Test imports (NumPy, PyTorch, Scikit-learn, etc.)
✓ Test pipeline modules (datasets, metrics, utils)
✓ Test baseline encoders (FlyHash, DummyEncoder)
✓ Test data loading (MNIST download)
```

**Runtime**: ~1-2 minutes (includes MNIST download)

### run_tests.sh

```
✓ Python environment check
✓ Core dependencies import
✓ Pipeline modules import
✓ Unit tests (test_pipeline.py)
✓ FlyHash encoder test
✓ Dataset loading test
```

**Runtime**: ~2-3 minutes

### test_pipeline.py

```python
def test_clustering_metrics():
    # Test NMI, ARI, ACC calculations
    
def test_retrieval_metrics():
    # Test mAP, Recall@K
    
def test_binarization():
    # Test top-k, threshold, WTA
    
def test_seed_reproducibility():
    # Test random seed setting
```

**Runtime**: ~10 seconds

---

## Interpreting Results

### Success Output

```
========================================
Clustering Pipeline - Quick Test
========================================

Testing imports...
  ✓ NumPy
  ✓ PyTorch
  ✓ Scikit-learn
  ...

Testing pipeline modules...
  ✓ Pipeline imports
  ✓ set_seed()
  ✓ top_k_binarization()
  ✓ Clustering metrics

...

========================================
Test Summary
========================================
Total tests: 12
✓ Passed: 12
✗ Failed: 0

All tests passed! ✓
```

### Failure Output

```
Testing baseline encoders...
  ✗ FlyHash encoder: ModuleNotFoundError: No module named 'baselines'

========================================
Test Summary
========================================
Total tests: 12
✓ Passed: 10
✗ Failed: 2

Some tests failed ✗

Troubleshooting:
  1. Check INSTALL.md for installation instructions
  2. Verify dependencies: pip install -r requirements.txt
  3. See TROUBLESHOOTING.md for common issues
```

---

## Common Test Failures

### Import Errors

```
ModuleNotFoundError: No module named 'xyz'
```

**Solution**:
```bash
pip install -r requirements.txt
```

### CUDA Errors

```
RuntimeError: CUDA out of memory
```

**Solution**: Tests should run on CPU, but if they try to use CUDA:
```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/quick_test.py
```

### Data Download Fails

```
HTTPError: 403 Forbidden
```

**Solution**: Manually download (see TROUBLESHOOTING.md)

### Permission Errors

```
PermissionError: [Errno 13] Permission denied
```

**Solution**:
```bash
chmod -R u+w outputs/
mkdir -p outputs/codes outputs/results
```

---

## Adding New Tests

### Add to test_pipeline.py

```python
def test_my_new_feature():
    """Test description."""
    # Setup
    data = ...
    
    # Execute
    result = my_function(data)
    
    # Assert
    assert result == expected
    
    print("My new test passed")
```

### Add to quick_test.py

```python
def test_my_module():
    """Test my module."""
    passed = 0
    failed = 0
    
    try:
        # Your test code
        passed += 1
        print("  ✓ My module")
    except Exception as e:
        failed += 1
        print(f"  ✗ My module: {e}")
    
    return passed, failed
```

### Add to run_tests.sh

```bash
echo "[7/7] Testing my new feature..."
total_tests=$((total_tests + 1))
if run_test "My feature" "python my_test_script.py"; then
    passed_tests=$((passed_tests + 1))
fi
```

---

## Continuous Integration

### GitHub Actions (template)

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python scripts/quick_test.py
```

---

## Performance Testing

### Timing Tests

```bash
# Time a baseline run
time python scripts/run_baseline.py --config configs/flyhash.yaml

# Time encoding only
time python baselines/flyhash/encoder.py
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Profile a script
python -m memory_profiler scripts/run_baseline.py --config configs/flyhash.yaml
```

---

## Test Data

### Using Test Subsets

For faster testing, use small subsets:

```python
# In your test
dataset = load_dataset('mnist')
test_data = dataset['test_data'][:100]  # Only 100 samples
```

### Synthetic Data

```python
# Generate synthetic data
import numpy as np

np.random.seed(0)
fake_data = np.random.randn(1000, 784)
fake_labels = np.random.randint(0, 10, 1000)
```

---

## Debugging Failed Tests

### Verbose Output

```bash
# Python with verbose
python -v scripts/quick_test.py

# Pytest with verbose
pytest tests/ -v -s
```

### Interactive Debugging

```bash
# Drop into debugger on failure
python -m pdb scripts/quick_test.py

# Or add breakpoint in code
import pdb; pdb.set_trace()
```

### Check Logs

```bash
# View test output
cat /tmp/test_output.log

# View pipeline logs
cat outputs/logs/*.log
```

---

## Test Coverage

### Install coverage.py

```bash
pip install coverage
```

### Measure Coverage

```bash
# Run with coverage
coverage run -m pytest tests/

# Generate report
coverage report

# HTML report
coverage html
open htmlcov/index.html
```

---

## Best Practices

1. **Run tests before committing**
   ```bash
   make quick-test && git commit
   ```

2. **Test on clean environment**
   ```bash
   conda create -n test_env python=3.9
   conda activate test_env
   pip install -r requirements.txt
   make test
   ```

3. **Test on different seeds**
   ```bash
   for seed in 0 1 2; do
       python scripts/run_baseline.py --config configs/flyhash.yaml --seed $seed
   done
   ```

4. **Keep tests fast**
   - Use small data subsets
   - Cache expensive operations
   - Skip slow tests in quick mode

5. **Write descriptive test names**
   ```python
   # Good
   def test_top_k_binarization_preserves_sparsity():
       pass
   
   # Bad
   def test1():
       pass
   ```

---

## Summary Commands

```bash
# Quick verification (1-2 min)
make quick-test

# Full test suite (2-3 min)
make test

# Integration test (10-30 min)
make run-flyhash

# Check status
make status

# Clean and test
make clean && make test
```

---

## Getting Help

If tests fail:

1. Check error message carefully
2. Review TROUBLESHOOTING.md
3. Try `make check-env` to verify setup
4. Run `make clean` and retry
5. Check documentation in docs/

For persistent issues:
- Review INSTALL.md
- Check GitHub issues
- Contact maintainer

---

**Last Updated**: 2026-01-09  
**Test Coverage**: ~80% of core functionality
