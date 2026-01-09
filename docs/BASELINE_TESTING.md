# Baseline Testing Guide

Complete guide for testing single or multiple baselines.

## Quick Start

### Test Single Baseline

```bash
# Using Python script
python scripts/test_baseline.py flyhash

# Using Makefile
make test-baseline BASELINE=flyhash
```

### Test Multiple Baselines

```bash
# Test specific baselines
python scripts/test_baseline.py flyhash diehl_cook

# Test all baselines
python scripts/test_baseline.py --all
```

### Batch Testing

```bash
# Quick batch test (1 seed, 1 dataset)
bash scripts/batch_test.sh --quick

# Full batch test (3 seeds, 2 datasets)
bash scripts/batch_test.sh --full
```

---

## Test Methods

### Method 1: Python Script (Recommended)

**Advantages**: Flexible, detailed output, easy to customize

```bash
# Test single baseline
python scripts/test_baseline.py flyhash

# Test with options
python scripts/test_baseline.py flyhash --dataset mnist --seeds 0 1 2

# Test multiple baselines
python scripts/test_baseline.py flyhash diehl_cook --seeds 0 1

# Test all
python scripts/test_baseline.py --all
```

**Options**:
- `--dataset NAME`: Dataset to use (mnist, fashion_mnist, etc.)
- `--seeds N1 N2 ...`: Multiple random seeds
- `--force`: Force re-encoding (ignore cache)
- `--quiet`: Less output
- `--list`: List available baselines

### Method 2: Bash Script

**Advantages**: Easy batch testing, parallel execution support

```bash
# Quick mode (1 seed, 1 dataset)
bash scripts/batch_test.sh --quick

# Full mode (3 seeds, 2 datasets)
bash scripts/batch_test.sh --full

# Custom configuration
bash scripts/batch_test.sh \
    --baselines 'flyhash diehl_cook' \
    --datasets 'mnist fashion_mnist' \
    --seeds '0 1 2'
```

**Options**:
- `--baselines LIST`: Space-separated baseline names
- `--datasets LIST`: Space-separated dataset names
- `--seeds LIST`: Space-separated seed values
- `--quick`: Preset for quick testing
- `--full`: Preset for thorough testing

### Method 3: Makefile

**Advantages**: Simplest, memorable commands

```bash
# Test single baseline
make test-baseline BASELINE=flyhash

# Quick test all baselines
make test-baselines-quick

# Full test all baselines
make test-baselines-full

# Batch tests
make batch-test-quick
make batch-test-full
```

---

## Examples

### Example 1: Quick Single Baseline Test

```bash
# Test FlyHash on MNIST with seed 0
python scripts/test_baseline.py flyhash
```

**Output**:
```
======================================================================
  Testing 1 Baseline(s) on mnist
======================================================================

--- Baseline: flyhash ---

[1/1] Testing flyhash with seed=0
Testing: flyhash
  Description: FlyHash - Fruit fly-inspired hashing
  Dataset: mnist
  Seed: 0
  Command: python scripts/run_baseline.py --config configs/flyhash.yaml --seed 0 --dataset mnist
  ✓ SUCCESS (12.3s)

flyhash Summary: 1/1 passed

======================================================================
  Test Summary
======================================================================

✓ flyhash        : 1/1 passed

Total: 1 tests
✓ Passed: 1
✗ Failed: 0
```

### Example 2: Test with Multiple Seeds

```bash
# Test FlyHash with 3 different seeds
python scripts/test_baseline.py flyhash --seeds 0 1 2
```

**Use case**: Verify reproducibility and get mean±std results

### Example 3: Test Multiple Baselines

```bash
# Test both FlyHash and Diehl & Cook
python scripts/test_baseline.py flyhash diehl_cook --seeds 0
```

**Use case**: Compare different methods on same data

### Example 4: Batch Test with Multiple Datasets

```bash
# Test on both MNIST and Fashion-MNIST
bash scripts/batch_test.sh \
    --baselines 'flyhash' \
    --datasets 'mnist fashion_mnist' \
    --seeds '0'
```

**Use case**: Evaluate generalization across datasets

### Example 5: Full Evaluation

```bash
# Test all baselines, all datasets, multiple seeds
bash scripts/batch_test.sh --full
```

**Use case**: Complete baseline evaluation for paper/report

### Example 6: Environment Variables

```bash
# Set via environment
BASELINES='flyhash diehl_cook' \
DATASETS='mnist' \
SEEDS='0 1 2 3 4' \
bash scripts/batch_test.sh
```

**Use case**: Flexible configuration without long command lines

### Example 7: Custom Pipeline Test

```bash
# Force re-encoding to test pipeline changes
python scripts/test_baseline.py flyhash --force --seeds 0
```

**Use case**: After modifying encoding logic

---

## Available Baselines

To see all available baselines:

```bash
# List baselines
python scripts/test_baseline.py --list
```

**Output**:
```
======================================================================
  Available Baselines
======================================================================

✓ flyhash        - FlyHash - Fruit fly-inspired hashing
                   (No training, CPU ok)
                   Config: configs/flyhash.yaml

✓ diehl_cook     - Diehl & Cook - STDP learning
                   (Needs training, GPU recommended)
                   Config: configs/diehl_cook.yaml
```

**Legend**:
- ✓ = Config file exists and baseline is ready
- ✗ = Config missing or baseline not implemented

---

## Test Configurations

### Quick Mode

**Purpose**: Fast verification  
**Settings**:
- 1 dataset (MNIST)
- 1 seed (0)
- ~5-10 minutes per baseline

```bash
bash scripts/batch_test.sh --quick
```

### Standard Mode

**Purpose**: Reliable results  
**Settings**:
- 1 dataset (MNIST)
- 3 seeds (0, 1, 2)
- ~15-30 minutes per baseline

```bash
python scripts/test_baseline.py --all --seeds 0 1 2
```

### Full Mode

**Purpose**: Complete evaluation  
**Settings**:
- 2 datasets (MNIST, Fashion-MNIST)
- 3 seeds (0, 1, 2)
- ~30-60 minutes per baseline

```bash
bash scripts/batch_test.sh --full
```

### Custom Mode

**Purpose**: Specific requirements  
**Settings**: Your choice

```bash
python scripts/test_baseline.py flyhash \
    --dataset fashion_mnist \
    --seeds 0 1 2 3 4 \
    --force
```

---

## Understanding Results

### Success Output

```
[1/3] Testing flyhash with seed=0
  ✓ SUCCESS (12.3s)
```

**Meaning**:
- Encoding completed
- Features saved
- Clustering/retrieval evaluated
- Results saved to outputs/

### Failure Output

```
[2/3] Testing diehl_cook with seed=1
  ✗ FAILED (5.2s)
  Error: CUDA out of memory
```

**Meaning**:
- Something went wrong
- Check error message
- See TROUBLESHOOTING.md

### Summary Format

```
======================================================================
  Test Summary
======================================================================

✓ flyhash        : 3/3 passed
✗ diehl_cook     : 2/3 passed

Total: 6 tests
✓ Passed: 5
✗ Failed: 1
```

**Interpretation**:
- FlyHash: All 3 seeds passed
- Diehl & Cook: 1 seed failed (check logs)
- Overall: 5/6 tests successful

---

## Output Locations

### Feature Codes

```
outputs/codes/{baseline}/{dataset}/
  ├── pre_code_seed0.npy      # Continuous features
  ├── code_seed0.npy           # Binary codes
  ├── pre_code_seed1.npy
  └── code_seed1.npy
```

### Results

```
outputs/results/
  ├── {baseline}_{dataset}_seed0.json
  ├── {baseline}_{dataset}_seed1.json
  └── summary_table.csv
```

### Logs

```
outputs/logs/
  ├── {baseline}_{dataset}_seed0.log
  └── {baseline}_{dataset}_seed1.log
```

### Batch Test Results

```
outputs/batch_results/
  └── batch_test_20260109_143022.txt
```

---

## Advanced Usage

### Parallel Testing

```bash
# Run multiple baselines in parallel (bash)
for baseline in flyhash diehl_cook; do
    python scripts/test_baseline.py $baseline --seeds 0 &
done
wait
echo "All tests complete"
```

### Testing Specific Configurations

```bash
# Test only clustering (skip retrieval)
python scripts/run_baseline.py \
    --config configs/flyhash.yaml \
    --seed 0 \
    # Config file controls eval_clustering: true

# Test only retrieval
# Modify config: eval_retrieval: true
```

### Comparing Results

```bash
# Test same baseline with different configs
for config in configs/flyhash_*.yaml; do
    python scripts/run_baseline.py --config $config --seed 0
done
```

### Automated Testing (CI/CD)

```bash
# In .github/workflows/test.yml
- name: Test baselines
  run: |
    python scripts/test_baseline.py --all --seeds 0
```

---

## Troubleshooting

### Issue: Baseline not found

```
ERROR: Unknown baseline 'xyz'
```

**Solution**:
```bash
# List available baselines
python scripts/test_baseline.py --list

# Check if config exists
ls configs/
```

### Issue: Config file missing

```
ERROR: Config not found: configs/xyz.yaml
```

**Solution**: Create the config file or use existing baseline

### Issue: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solution**:
```bash
# Use CPU instead
# Edit config file: device: "cpu"

# Or reduce batch size/neurons
```

### Issue: Tests take too long

**Solution**:
```bash
# Use quick mode
bash scripts/batch_test.sh --quick

# Or test subset
python scripts/test_baseline.py flyhash --seeds 0
```

### Issue: Inconsistent results across seeds

**Solution**: This is expected for some stochastic methods. Report mean±std.

---

## Best Practices

### 1. Always Test with Multiple Seeds

```bash
# Bad: Single seed (unreliable)
python scripts/test_baseline.py flyhash --seeds 0

# Good: Multiple seeds (robust)
python scripts/test_baseline.py flyhash --seeds 0 1 2
```

### 2. Use Quick Mode for Development

```bash
# During development
bash scripts/batch_test.sh --quick

# For final results
bash scripts/batch_test.sh --full
```

### 3. Cache Features When Possible

```bash
# First run: Slow (encoding)
python scripts/test_baseline.py flyhash --seeds 0

# Second run: Fast (uses cache)
python scripts/test_baseline.py flyhash --seeds 0
```

### 4. Check Logs on Failure

```bash
# Run test
python scripts/test_baseline.py flyhash

# If failed, check log
cat outputs/logs/flyhash_mnist_seed0.log
```

### 5. Clean Before Important Tests

```bash
# Clean old outputs
make clean

# Run fresh test
python scripts/test_baseline.py --all
```

---

## Integration with Workflow

### Development Workflow

```bash
# 1. Modify baseline code
vim baselines/flyhash/encoder.py

# 2. Quick test
python scripts/test_baseline.py flyhash --seeds 0

# 3. Full test before commit
python scripts/test_baseline.py flyhash --seeds 0 1 2

# 4. Commit
git commit -m "Update FlyHash encoder"
```

### Evaluation Workflow

```bash
# 1. Test all baselines
bash scripts/batch_test.sh --full

# 2. Check results
cat outputs/batch_results/batch_test_*.txt

# 3. Generate figures
python scripts/plot_results.py  # (to be implemented)

# 4. Write report
```

---

## Summary Commands

```bash
# Quick reference
python scripts/test_baseline.py --list          # List baselines
python scripts/test_baseline.py flyhash         # Test one
python scripts/test_baseline.py --all           # Test all
bash scripts/batch_test.sh --quick              # Quick batch
bash scripts/batch_test.sh --full               # Full batch

# Makefile shortcuts
make test-baseline BASELINE=flyhash             # Test one
make test-baselines-quick                       # Test all (quick)
make test-baselines-full                        # Test all (full)
make batch-test-quick                           # Batch (quick)
```

---

## Next Steps

After testing baselines:

1. **Analyze results**: Check `outputs/results/`
2. **Compare performance**: Use metrics (NMI, ARI, mAP)
3. **Generate report**: See baseline_report_template.md
4. **Iterate**: Improve baselines based on results

---

**Last Updated**: 2026-01-09  
**Version**: 1.0
