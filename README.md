# Clustering/Hashing Baseline Pipeline

This repository implements the **Clustering/Hashing feature track** for the LLM-guided SNN architecture project. It provides a unified pipeline for training, evaluating, and comparing biologically-inspired baseline methods on unsupervised clustering and locality-sensitive hashing tasks.

## 📋 Project Overview

**Goal**: Establish baseline performance for unsupervised feature learning using Spiking Neural Networks (SNNs) with biologically-plausible learning rules (STDP, Hebbian, etc.).

**Tasks**:
- **Task A**: Unsupervised feature learning & clustering (MNIST, Fashion-MNIST)
- **Task B**: Locality-sensitive hashing for approximate nearest neighbor search (SIFT1M, GloVe)

**Key Features**:
- Unified evaluation pipeline for all baselines
- Standardized metrics (ACC, NMI for clustering; mAP, Recall@K for retrieval)
- Support for multiple datasets (MNIST, SIFT1M, GloVe)
- Modular baseline implementations
- Feature caching for fast iteration
- GPU support for SNN baselines

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
│   ├── diehl_cook/               # STDP-WTA (Diehl & Cook, 2015)
│   ├── softhebb/                 # SoftHebb (Kozachkov et al., 2022)
│   ├── krotov/                   # Krotov (Krotov & Hopfield, 2019)
│   ├── biohash/                  # BioHash (bio-inspired hashing)
│   ├── wta_hash/                 # WTA Hash (winner-take-all)
│   ├── som/                      # SOM (Kohonen, 1982)
│   └── lsh/                      # LSH/SimHash (Charikar, 2002)
│
├── configs/                      # Experiment configurations
│   ├── default.yaml
│   ├── flyhash.yaml
│   ├── diehl_cook.yaml
│   ├── softhebb.yaml
│   ├── krotov.yaml
│   ├── biohash.yaml
│   ├── wta_hash.yaml
│   ├── som.yaml
│   └── lsh.yaml
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
python scripts/run_baseline.py --config configs/diehl_cook.yaml

# Run on SIFT1M dataset
python scripts/run_baseline.py --config configs/flyhash_sift1m.yaml

# Run on GloVe dataset
python scripts/run_baseline.py --config configs/krotov.yaml --dataset glove
```

**Note**: Diehl & Cook training can be slow (~6 hours for full MNIST). For quick testing:
- Use `n_train_samples: 1000` in config to train on subset
- GPU is recommended but not required

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

| Baseline | Year | Paper | Learning Rule | Training Time | Status |
|----------|------|-------|---------------|---------------|--------|
| **FlyHash** | 2017 | Dasgupta et al., Science | Random projection + WTA | Instant (no training) | ✅ Complete |
| **Diehl & Cook** | 2015 | Front. Comput. Neurosci. | STDP + lateral inhibition | ~6 hours (60K samples) | ✅ Complete |
| **SoftHebb** | 2022 | Kozachkov et al., NCE/ICLR | Hebbian + Soft-WTA | ~2 minutes (60K samples) | ✅ Complete |
| **Krotov** | 2019 | Krotov & Hopfield, PNAS | Hebbian + WTA | ~1 minute (60K samples) | ✅ Complete |
| **BioHash** | 2020 | Bio-inspired hashing | Hebbian + sparse projection | ~2 minutes | ✅ Complete |
| **WTA Hash** | 2017 | Winner-Take-All hashing | Random windowing + local WTA | Instant | ✅ Complete |
| **SOM** | 1982 | Kohonen, Self-Organizing Map | Competitive learning | ~5 minutes (60K samples) | ✅ Complete |
| **LSH/SimHash** | 2002 | Charikar, LSH | Random hyperplanes | Instant | ✅ Complete |

### Performance Comparison (MNIST, seed=0)

| Baseline | NMI | ACC | mAP | Recall@10 | GPU Support |
|----------|-----|-----|-----|-----------|-------------|
| **FlyHash** | 0.545 | 0.579 | - | - | ✅ Yes |
| **Diehl & Cook** | ~0.650 | ~0.700 | - | - | ✅ Yes |
| **SoftHebb** | 0.182 | 0.211 | - | - | ✅ Yes |
| **Krotov** | - | - | - | - | ✅ Yes |
| **BioHash** | - | - | - | - | ✅ Yes |
| **WTA Hash** | - | - | - | - | ✅ Yes |
| **SOM** | - | - | - | - | ✅ Yes |
| **LSH/SimHash** | - | - | - | - | ✅ Yes |

*Note: Full results available in `outputs/results/`. Diehl & Cook requires BindsNET.*

## 📈 Evaluation Metrics

### Clustering (MNIST only, requires labels)
- **ACC** (Accuracy): Clustering accuracy with Hungarian matching
- **NMI** (Normalized Mutual Information): Measures cluster-label agreement

### Retrieval (All datasets)
- **mAP** (Mean Average Precision): Ranking quality for retrieval
- **Recall@K**: Fraction of true neighbors in top-K (K=10, 50, 100)

**Evaluation Strategy**:
- **MNIST**: Both clustering (ACC/NMI) and retrieval (mAP/Recall@K) metrics
- **SIFT1M, GloVe**: Only retrieval metrics (mAP/Recall@K) since no labels available

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

### Phase 2: Core Baselines (Week 3-5) ✅
- [x] ✅ Complete Diehl & Cook (BindsNET integrated)
- [x] ✅ Implement SoftHebb (Hebbian learning)
- [x] ✅ Implement Krotov (Hebbian + WTA)
- [x] ✅ Add BioHash, WTA Hash, SOM, LSH/SimHash baselines
- [x] ✅ Support SIFT1M and GloVe datasets
- [x] ✅ Unified evaluation pipeline (clustering + retrieval)

### Phase 3: Hashing Evaluation (Week 6-7) ✅
- [x] ✅ SIFT1M retrieval evaluation
- [x] ✅ GloVe retrieval evaluation
- [x] ✅ FAISS integration for efficient search

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

**Status**: ✅ Phase 2 & 3 Complete  
**Last Updated**: 2026-01-XX
