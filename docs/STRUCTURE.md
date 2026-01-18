# Project Structure

## Overview

```
clustering/
├── run.py                      # ⭐ Main entry point
├── README.md                   # Project overview
├── requirements.txt            # Python dependencies
├── Makefile                    # Convenient commands
│
├── docs/                       # 📚 All documentation
│   ├── README.md                          # Documentation index
│   ├── INSTALL.md                         # Installation guide
│   ├── TROUBLESHOOTING.md                 # Problem solving
│   ├── TESTING_SUMMARY.md                 # Quick test reference
│   ├── BASELINE_TESTING.md                # Complete test guide
│   ├── TEST_GUIDE.md                      # General testing
│   ├── BINDSNET_INTEGRATION.md            # BindsNET details
│   ├── BINDSNET_INTEGRATION_SUMMARY.md    # BindsNET summary
│   ├── bindsnet_status.md                 # Current status
│   ├── clustering_hashing_baseline_guide.md  # Implementation guide
│   └── baseline_code_availability_report.md  # Code survey
│
├── pipeline/                   # 🔧 Core pipeline modules
│   ├── __init__.py
│   ├── datasets.py             # Data loading
│   ├── metrics.py              # Evaluation metrics
│   ├── binarization.py         # Code binarization
│   ├── clustering.py           # Clustering algorithms
│   ├── retrieval.py            # Retrieval evaluation
│   └── utils.py                # Utilities
│
├── baselines/                  # 🧠 Baseline implementations
│   ├── base_encoder.py         # Abstract interface
│   ├── flyhash/
│   │   └── encoder.py
│   ├── diehl_cook/
│   │   ├── encoder.py
│   │   ├── train.py
│   │   └── README.md
│   └── softhebb/
│       └── encoder.py          # (to be implemented)
│
├── configs/                    # ⚙️ Configuration files
│   ├── default.yaml
│   ├── flyhash.yaml
│   └── diehl_cook.yaml
│
├── scripts/                    # 🛠️ Utility scripts
│   ├── run_baseline.py         # Backend runner
│   ├── test_baseline.py        # Baseline testing
│   ├── batch_test.sh           # Batch testing
│   ├── quick_test.py           # Quick tests
│   ├── run_tests.sh            # Test suite
│   ├── run_all_baselines.sh    # Batch runner
│   ├── download_sift1m.sh      # Data download
│   ├── download_glove.sh       # Data download
│   └── setup.sh                # Environment setup
│
├── tests/                      # ✅ Unit tests
│   └── test_pipeline.py
│
├── data/                       # 💾 Datasets (gitignored)
│   ├── mnist/
│   ├── fashion_mnist/
│   └── sift1m/
│
├── outputs/                    # 📊 Results (gitignored)
│   ├── codes/                  # Feature codes
│   ├── results/                # Evaluation results
│   ├── logs/                   # Training logs
│   └── batch_results/          # Batch test results
│
└── notebooks/                  # 📓 Analysis notebooks
    └── (to be added)
```

## Key Files

### Main Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| **run.py** | Main entry point | `python run.py --baseline flyhash` |
| **Makefile** | Convenient shortcuts | `make run-flyhash` |

### Documentation

All documentation is in `docs/`:

| File | Purpose |
|------|---------|
| **docs/README.md** | Documentation index |
| **docs/INSTALL.md** | Installation instructions |
| **docs/TROUBLESHOOTING.md** | Problem solving |
| **docs/TESTING_SUMMARY.md** | Quick test reference |
| **docs/BASELINE_TESTING.md** | Complete test guide |

### Core Modules

| Module | Purpose |
|--------|---------|
| **pipeline/datasets.py** | Load MNIST, Fashion-MNIST, SIFT1M, GloVe |
| **pipeline/metrics.py** | NMI, ARI, ACC, mAP, Recall@K |
| **pipeline/binarization.py** | Top-k, WTA, threshold |
| **pipeline/clustering.py** | K-means, K-medoids, spectral |
| **pipeline/retrieval.py** | ANN search and evaluation |

### Baselines

| Baseline | Location | Status |
|----------|----------|--------|
| **FlyHash** | baselines/flyhash/ | ✅ Ready |
| **Diehl & Cook** | baselines/diehl_cook/ | 🟡 Interface ready |
| **SoftHebb** | baselines/softhebb/ | ⚪ To be implemented |

### Scripts

| Script | Purpose |
|--------|---------|
| **scripts/run_baseline.py** | Backend for running baselines |
| **scripts/test_baseline.py** | Test single/multiple baselines |
| **scripts/batch_test.sh** | Batch testing with configs |
| **scripts/quick_test.py** | Quick validation tests |

## Usage Patterns

### Running Baselines

```bash
# Method 1: Main entry point (recommended)
python run.py --baseline flyhash --dataset mnist --seed 0

# Method 2: With config
python run.py --config configs/flyhash.yaml

# Method 3: Makefile
make run-flyhash

# Method 4: Direct script
python scripts/run_baseline.py --config configs/flyhash.yaml
```

### Testing

```bash
# Quick test
python run.py --test

# Test single baseline
python scripts/test_baseline.py flyhash

# Batch test
bash scripts/batch_test.sh --quick
```

### Documentation

```bash
# View documentation index
cat docs/README.md

# View specific doc
cat docs/INSTALL.md

# List all docs
make docs
```

## Output Structure

```
outputs/
├── codes/                              # Cached feature codes
│   └── {baseline}/{dataset}/
│       ├── pre_code_seed0.npy         # Continuous features
│       └── code_seed0.npy             # Binary codes
│
├── results/                            # Evaluation results
│   └── {baseline}_{dataset}_seed0.json
│
├── logs/                               # Training logs
│   └── {baseline}_{dataset}_seed0.log
│
└── batch_results/                      # Batch test summaries
    └── batch_test_20260109.txt
```

## Configuration Structure

```yaml
# configs/baseline.yaml

experiment_name: "baseline_mnist"
seed: 0
dataset: "mnist"

encoder: "baseline_name"
encoder_config:
  # Baseline-specific parameters
  input_dim: 784
  ...

eval_clustering: true
eval_retrieval: false
n_clusters: 10
```

## Design Principles

### 1. Single Entry Point

- **run.py** is the main interface
- All features accessible through simple commands
- Consistent API across all baselines

### 2. Organized Documentation

- All .md files (except README) in `docs/`
- Clear hierarchy: Getting Started → Implementation → Testing
- Cross-referenced with index

### 3. Modular Architecture

- **pipeline/** for core functionality
- **baselines/** for method implementations
- **scripts/** for utilities
- Clear separation of concerns

### 4. Caching Strategy

- Features cached in `outputs/codes/`
- Avoid re-encoding unless forced
- Fast iteration during development

### 5. Flexible Testing

- Multiple test levels (quick, standard, full)
- Support for single and batch testing
- Easy to add new baselines

## Adding New Components

### Adding a Baseline

1. Create directory: `baselines/new_baseline/`
2. Implement encoder: `baselines/new_baseline/encoder.py`
3. Create config: `configs/new_baseline.yaml`
4. Add to `AVAILABLE_BASELINES` in `scripts/test_baseline.py`
5. Document in `docs/`

### Adding Documentation

1. Create file in `docs/`
2. Add entry to `docs/README.md`
3. Update main `README.md` if needed
4. Cross-reference related docs

### Adding Tests

1. Add test function to `tests/test_pipeline.py`
2. Or create new test file in `tests/`
3. Update `scripts/run_tests.sh` if needed

## Migration Notes

### What Changed

**Before** (old structure):
```
clustering/
├── INSTALL.md
├── TROUBLESHOOTING.md
├── BINDSNET_INTEGRATION.md
├── ...
└── scripts/run_baseline.py  # Main entry
```

**After** (new structure):
```
clustering/
├── run.py                    # ⭐ New main entry
├── docs/                     # 📚 All docs here
│   ├── INSTALL.md
│   ├── TROUBLESHOOTING.md
│   └── ...
└── scripts/run_baseline.py  # Backend
```

### Migration Benefits

1. **Cleaner root directory**
2. **Single entry point** (`run.py`)
3. **Organized documentation**
4. **Easier to navigate**
5. **More professional structure**

## Quick Reference

```bash
# Main commands
python run.py --list          # List baselines
python run.py --test          # Quick test
python run.py --baseline NAME # Run baseline

# Documentation
cat docs/README.md            # Doc index
ls docs/                      # List all docs

# Testing
python scripts/test_baseline.py --list  # List testable baselines
make test                                # Run tests

# Makefile shortcuts
make help                     # Show all commands
make run-flyhash              # Run baseline
make docs                     # Show doc index
```

---

**Last Updated**: 2026-01-09  
**Version**: 2.0 (Reorganized)  
**Total Files**: 50+ across all directories
