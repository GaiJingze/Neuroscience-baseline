# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Basic installation (CPU only)
pip install -r requirements.txt

# Or with GPU support
pip install -r requirements.txt
pip install faiss-gpu
```

For detailed installation (especially for BindsNET), see [`docs/INSTALL.md`](docs/INSTALL.md)

### Step 2: List Available Baselines

```bash
python run.py --list
```

Expected output:
```
Available Baselines
===================

✅ flyhash
   Status: Ready
   Config: configs/flyhash.yaml
   Description: Fly-inspired locality-sensitive hashing

🟡 diehl_cook
   Status: Interface ready (training needed)
   Config: configs/diehl_cook.yaml
   Description: STDP-based unsupervised SNN
```

### Step 3: Run a Baseline

```bash
# Run FlyHash (fastest, no training needed)
python run.py --baseline flyhash --dataset mnist --seed 0

# Or with config file
python run.py --config configs/flyhash.yaml
```

## 🎯 What Happens When You Run

```
1. Loading dataset...          [MNIST: 60k train, 10k test]
2. Encoding data...             [Using FlyHash encoder]
3. Binarizing codes...          [Top-k WTA]
4. Evaluating clustering...     [K-means, computing NMI/ARI/ACC]
5. Saving results...            [outputs/results/flyhash_mnist_seed0.json]
```

## 📊 View Results

```bash
# View detailed results
cat outputs/results/flyhash_mnist_seed0.json

# Results include:
# - Clustering metrics: NMI, ARI, ACC
# - Silhouette score, Davies-Bouldin index
# - Encoding time, evaluation time
```

Example output:
```json
{
  "experiment_name": "flyhash_mnist",
  "dataset": "mnist",
  "encoder": "flyhash",
  "clustering": {
    "nmi": 0.482,
    "ari": 0.365,
    "acc": 0.542
  },
  "timing": {
    "encoding_time": 1.23,
    "clustering_time": 0.87
  }
}
```

## 🧪 Test Your Setup

```bash
# Quick test (validates all components)
python run.py --test

# Test specific baseline
python scripts/test_baseline.py flyhash
```

## 📚 Next Steps

### Run More Baselines

```bash
# Try different datasets
python run.py --baseline flyhash --dataset fashion_mnist

# Try different seeds
python run.py --baseline flyhash --seed 1

# Run Diehl & Cook (requires BindsNET setup)
# See docs/BINDSNET_INTEGRATION.md first
python run.py --baseline diehl_cook
```

### Batch Processing

```bash
# Run all baselines
make run-all

# Or manually
bash scripts/run_all_baselines.sh
```

### Download More Datasets

```bash
# Download SIFT1M
bash scripts/download_sift1m.sh

# Download GloVe
bash scripts/download_glove.sh
```

## 🛠️ Common Commands

```bash
# Main entry point
python run.py --help              # Show help
python run.py --list              # List baselines
python run.py --test              # Quick test
python run.py --help-test         # Testing help
python run.py --help-config       # Config help

# Using Makefile (shortcuts)
make help                         # Show all commands
make install                      # Install dependencies
make test                         # Run tests
make run-flyhash                  # Run FlyHash
make run-all                      # Run all baselines
make clean                        # Clean outputs
make docs                         # Show doc index

# Testing
make quick-test                   # Quick validation
make test-baseline BASELINE=flyhash  # Test single baseline
make batch-test-quick             # Batch quick tests

# Development
make format                       # Format code (black)
make lint                         # Lint code (flake8)
make check-env                    # Check environment
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `No module named 'bindsnet'`
```bash
# Solution: Install BindsNET
pip install bindsnet>=0.3.1
# Or use dedicated setup script
bash setup_bindsnet_env.sh
```

**Issue**: NumPy version conflicts
```bash
# Solution: Use compatible versions
pip install "numpy>=1.21.0,<2.0.0"
pip install "torch>=1.10.0,<2.3.0"
```

**Issue**: Dataset not found
```bash
# Solution: Datasets are auto-downloaded on first use
# Or manually download:
bash scripts/download_sift1m.sh
```

For more solutions, see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)

## 📖 Documentation

All documentation is organized in `docs/`:

| Document | Purpose |
|----------|---------|
| [`docs/README.md`](docs/README.md) | Documentation index |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Detailed installation |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Problem solving |
| [`docs/TESTING_SUMMARY.md`](docs/TESTING_SUMMARY.md) | Quick test reference |
| [`docs/clustering_hashing_baseline_guide.md`](docs/clustering_hashing_baseline_guide.md) | Complete guide |

View full documentation tree:
```bash
cat docs/README.md
```

## 💡 Tips

1. **Start with FlyHash** - It's the fastest and requires no training
2. **Use quick tests** - Validate your setup before full runs
3. **Check docs/** - Comprehensive documentation for everything
4. **Use Makefile** - Convenient shortcuts for common tasks
5. **Cache is smart** - Results are cached in `outputs/codes/` for speed

## 🎓 Learning Path

### Beginner
1. Read this guide (QUICK_START.md)
2. Run quick test: `python run.py --test`
3. Run FlyHash: `python run.py --baseline flyhash`
4. Check results: `cat outputs/results/flyhash_mnist_seed0.json`

### Intermediate
1. Read implementation guide: `docs/clustering_hashing_baseline_guide.md`
2. Try different datasets and seeds
3. Understand metrics in `pipeline/metrics.py`
4. Modify configs in `configs/`

### Advanced
1. Setup BindsNET: `docs/BINDSNET_INTEGRATION.md`
2. Implement new baseline: follow `baselines/base_encoder.py` interface
3. Add custom evaluation metrics
4. Contribute to the codebase

## 🤝 Getting Help

1. **Check documentation**: `docs/README.md` has full index
2. **Run tests**: Diagnose issues with `python run.py --test`
3. **Read troubleshooting**: `docs/TROUBLESHOOTING.md`
4. **Check structure**: `STRUCTURE.md` explains organization
5. **Contact**: Jingze Gai (project maintainer)

## ✅ Checklist for First Run

- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Environment validated: `python run.py --test`
- [ ] Listed baselines: `python run.py --list`
- [ ] Run first baseline: `python run.py --baseline flyhash`
- [ ] Check results: `cat outputs/results/flyhash_mnist_seed0.json`
- [ ] Read main docs: `cat docs/README.md`

## 🎉 Success!

If you've completed the checklist above, you're ready to go! 

Next steps:
- Explore different baselines
- Try your own datasets
- Read the implementation guide
- Contribute improvements

Happy experimenting! 🚀

---

**Last Updated**: 2026-01-09  
**Difficulty**: Beginner  
**Time to Complete**: 15-30 minutes
