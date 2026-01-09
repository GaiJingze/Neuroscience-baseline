# Clustering/Hashing Baseline Pipeline

This repository implements the **Clustering/Hashing feature track** for the LLM-guided SNN architecture project. It provides a unified pipeline for training, evaluating, and comparing biologically-inspired baseline methods on unsupervised clustering and locality-sensitive hashing tasks.

## 📋 Project Overview

**Goal**: Establish baseline performance for unsupervised feature learning using Spiking Neural Networks (SNNs) with biologically-plausible learning rules (STDP, Hebbian, etc.).

**Tasks**:
- **Task A**: Unsupervised feature learning & clustering (MNIST, Fashion-MNIST)
- **Task B**: Locality-sensitive hashing for approximate nearest neighbor search (SIFT1M)

**Key Features**:
- Unified evaluation pipeline for all baselines
- Standardized metrics (NMI, ARI, ACC for clustering; mAP, Recall@K for retrieval)
- SNN-specific metrics (spike sparsity, temporal dynamics)
- Modular baseline implementations
- Feature caching for fast iteration

## 🏗️ Repository Structure

```
clustering/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── data/                         # Datasets (auto-downloaded or manual)
│   ├── mnist/
│   ├── fashion_mnist/
│   └── sift1m/
│
├── pipeline/                     # Core evaluation pipeline
│   ├── __init__.py
│   ├── datasets.py               # Unified data loaders
│   ├── binarization.py           # WTA, top-k, threshold methods
│   ├── clustering.py             # Clustering algorithms & evaluation
│   ├── retrieval.py              # Retrieval (ANN) evaluation
│   ├── metrics.py                # NMI, ARI, ACC, mAP, Recall@K
│   └── utils.py                  # Utilities (seed, logging, etc.)
│
├── baselines/                    # Baseline implementations
│   ├── base_encoder.py           # Abstract base class
│   ├── flyhash/                  # FlyHash (Dasgupta et al., 2017)
│   │   └── encoder.py
│   ├── diehl_cook/               # STDP-WTA (Diehl & Cook, 2015)
│   │   └── encoder.py
│   └── softhebb/                 # SoftHebb (Moraitis et al., 2022)
│       └── encoder.py            # (to be implemented)
│
├── configs/                      # Experiment configurations
│   ├── default.yaml
│   ├── flyhash.yaml
│   └── diehl_cook.yaml
│
├── scripts/                      # Utility scripts
│   ├── run_baseline.py           # Main evaluation script
│   ├── download_sift1m.sh        # Download SIFT1M dataset
│   └── download_glove.sh         # Download GloVe embeddings
│
├── outputs/                      # Experiment outputs (gitignored)
│   ├── codes/                    # Cached feature codes
│   ├── results/                  # Evaluation results (JSON)
│   └── logs/                     # Training logs
│
├── notebooks/                    # Analysis notebooks
│   └── (to be added)
│
└── docs/                         # Documentation
    └── clustering_hashing_baseline_guide.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository (or create new directory)
cd clustering

# Create conda environment
conda create -n clustering_pipeline python=3.9
conda activate clustering_pipeline

# Install dependencies
pip install -r requirements.txt

# Verify BindsNET installation (for SNN baselines)
python -c "import bindsnet; print(f'BindsNET {bindsnet.__version__} installed successfully')"
```

**⚠️ Installation Issues?**

If you encounter dependency problems:
- 📖 **Quick fixes**: See `docs/INSTALLATION_QUICK_FIXES.md`
- 📊 **Version status**: See `docs/VERSION_STATUS.md`
- 🔧 **Full guide**: See `docs/INSTALL.md`

Common issues:
- BindsNET version not found → Use `bindsnet>=0.2.7` (PyPI latest)
- `torch._six` error → Install from GitHub or downgrade PyTorch
- NumPy conflicts → Use `numpy<2.0.0`

### 2. Download Datasets

```bash
# MNIST & Fashion-MNIST (auto-download via torchvision)
python -c "from torchvision import datasets; datasets.MNIST('./data', download=True); datasets.FashionMNIST('./data', download=True)"

# SIFT1M (manual download, ~400MB)
bash scripts/download_sift1m.sh

# GloVe (optional, ~860MB)
bash scripts/download_glove.sh
```

### 3. Run a Baseline

```bash
# Method 1: Using main entry point (recommended)
python run.py --baseline flyhash --dataset mnist --seed 0

# Method 2: With config file
python run.py --config configs/flyhash.yaml

# Method 3: Direct script call
python scripts/run_baseline.py --config configs/flyhash.yaml

# Run with different seed
python run.py --baseline flyhash --seed 1

# Run Diehl & Cook (requires BindsNET, GPU recommended)
python run.py --baseline diehl_cook

# Full training with BindsNET (more control)
python baselines/diehl_cook/train.py \
    --train --extract \
    --n_train 1000 \
    --n_epochs 1 \
    --device cuda
```

**Note**: Diehl & Cook training can be slow. For quick testing:
- Use `--n_train 1000` to train on subset
- Use `--n_epochs 1` for single epoch
- Consider using CPU if no GPU available (will be slower)

### 4. Quick Commands

```bash
# List available baselines
python run.py --list

# Run quick test
python run.py --test

# Get help
python run.py --help
python run.py --help-test      # Testing help
python run.py --help-config    # Config help
```

### 5. View Results

Results are saved in `outputs/results/`:
```bash
cat outputs/results/flyhash_mnist_seed0.json
```

## 📊 Baselines

### Implemented

| Baseline | Year | Paper | Learning Rule | Status |
|----------|------|-------|---------------|--------|
| **FlyHash** | 2017 | Dasgupta et al., Science | Random projection + WTA | ✅ Complete |
| **Diehl & Cook** | 2015 | Front. Comput. Neurosci. | STDP + lateral inhibition | 🟡 Interface ready (full BindsNET training in progress) |

### To Implement

| Baseline | Year | Paper | Priority |
|----------|------|-------|----------|
| **SoftHebb** | 2022 | Moraitis et al., NCE | HIGH |
| **Deep STDP** | 2024 | Lu & Sengupta, NCE | HIGH |
| **BioHash** | 2020 | (if exists) | MEDIUM |

## 📈 Evaluation Metrics

### Clustering (MNIST, Fashion-MNIST)
- **NMI** (Normalized Mutual Information): Measures cluster-label agreement
- **ARI** (Adjusted Rand Index): Measures clustering similarity
- **ACC** (Accuracy): With Hungarian matching for optimal alignment
- **Silhouette Score**: Internal clustering quality (no labels needed)

### Retrieval (SIFT1M)
- **mAP** (Mean Average Precision): Ranking quality
- **Recall@K**: Fraction of true neighbors in top-K (K=10, 50, 100)
- **Precision@K**: Precision of top-K retrieved items

### SNN-Specific
- **Spike Sparsity**: 1 - (firing_rate), measures energy efficiency
- **Hamming Distance**: For binary codes
- **Temporal Dynamics**: (optional) Spike timing analysis

## 🔧 Adding a New Baseline

1. **Create encoder class** in `baselines/your_method/encoder.py`:

```python
from baselines.base_encoder import BaseEncoder

class YourEncoder(BaseEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your method
    
    def fit(self, train_data, train_labels=None):
        # Train your encoder
        self.is_trained = True
    
    def encode(self, data):
        # Encode data
        return {
            'pre_code': pre_code,  # Continuous features
            'code': code           # Binary/sparse code
        }
```

2. **Create config file** in `configs/your_method.yaml`

3. **Update** `scripts/run_baseline.py` to import your encoder

4. **Run**: `python scripts/run_baseline.py --config configs/your_method.yaml`

## 📝 Configuration

Example config (`configs/flyhash.yaml`):

```yaml
experiment_name: "flyhash_mnist"
seed: 0
dataset: "mnist"

encoder: "flyhash"
encoder_config:
  input_dim: 784
  projection_dim: 2000
  hash_length: 100

eval_clustering: true
n_clusters: 10
clustering_methods:
  - kmeans
  - kmedoids
```

## 🧪 Testing

### Quick Test (Recommended)

```bash
# One-click test - runs all checks
python scripts/quick_test.py

# Or use shell script
bash scripts/run_tests.sh

# Or use Makefile
make test           # Full test suite
make quick-test     # Quick tests only
```

### Individual Tests

```bash
# Test individual modules
python pipeline/datasets.py
python pipeline/metrics.py
python pipeline/binarization.py

# Test a baseline
python baselines/flyhash/encoder.py

# Test with pytest (if installed)
pytest tests/
```

## 📖 Documentation

All documentation is in the `docs/` directory:

### Getting Started
- **README.md** (this file) - Project overview and quick start
- **docs/INSTALL.md** - Detailed installation guide
- **docs/TROUBLESHOOTING.md** - Common issues and solutions

### Implementation Guides
- **docs/clustering_hashing_baseline_guide.md** - Complete implementation guide
- **docs/baseline_code_availability_report.md** - Baseline survey and availability
- **docs/BINDSNET_INTEGRATION.md** - BindsNET integration details
- **docs/bindsnet_status.md** - Current BindsNET status

### Testing
- **docs/TESTING_SUMMARY.md** - Quick testing reference (中文)
- **docs/BASELINE_TESTING.md** - Complete baseline testing guide
- **docs/TEST_GUIDE.md** - General testing documentation

### Project Documentation
- **Original project doc**: `../LLM for SNN architecture-2025Dec12-V1.docx`

## 🎯 Roadmap

### Phase 1: Setup & Quick Win (Week 1-2) ✅
- [x] Repository structure
- [x] Core pipeline modules
- [x] FlyHash baseline
- [x] MNIST evaluation

### Phase 2: Core Baselines (Week 3-5) 🔨
- [x] ✅ Complete Diehl & Cook (BindsNET integrated)
- [ ] Implement SoftHebb
- [ ] Investigate Lu & Sengupta 2024
- [ ] Run all on MNIST + Fashion-MNIST

### Phase 3: Hashing Evaluation (Week 6-7)
- [ ] SIFT1M retrieval evaluation
- [ ] Performance optimization (FAISS integration)

### Phase 4: Analysis & Documentation (Week 8)
- [ ] Statistical analysis
- [ ] Visualization notebooks
- [ ] Final baseline report

## 🤝 Contributing

This is a research project. For questions or contributions, contact Jingze Gai.

## 📄 License

(To be determined based on project requirements)

## 🙏 Acknowledgments

- **BindsNET**: PyTorch-based SNN framework
- **FlyHash**: Fruit fly-inspired hashing
- **Project Team**: LLM for SNN Architecture

---

**Status**: 🚧 Work in Progress (Phase 1 Complete)  
**Last Updated**: 2026-01-09
