# Clustering/Hashing Baseline Implementation Guide

## 📋 Document Information
- **Project**: LLM for SNN Architecture - Clustering/Hashing Feature Track
- **Responsible**: Jingze Gai
- **Last Updated**: 2026-01-09
- **Status**: Implementation Ready

---

## 1. Task Definition & Objectives

### 1.1 Primary Tasks

This track focuses on **two complementary tasks**:

#### Task A: **Unsupervised Feature Learning & Clustering**
- **Goal**: Learn meaningful representations from unlabeled data using biologically-inspired SNN learning rules (STDP, Hebbian, etc.)
- **Evaluation**: Clustering quality on labeled datasets (for validation)
- **Metrics**: NMI, ARI, ACC (with Hungarian matching)

#### Task B: **Locality-Sensitive Hashing for ANN**
- **Goal**: Learn compact binary/sparse codes for efficient similarity search
- **Evaluation**: Retrieval performance on high-dimensional vector datasets
- **Metrics**: mAP, Recall@K (K=10, 50, 100), Precision@K

### 1.2 Why Both Tasks?

- **Clustering** validates that SNN learns semantically meaningful structures
- **Hashing** tests efficiency and scalability (critical for neuromorphic hardware)
- Both demonstrate **unsupervised learning** capability of biologically-inspired SNNs

### 1.3 SNN-Specific Focus

Unlike traditional deep learning baselines, we emphasize:
- **Biological plausibility**: STDP, Hebbian learning, lateral inhibition
- **Spike sparsity**: Energy efficiency through sparse spike patterns
- **Temporal dynamics**: Leveraging spike timing information
- **Zero/few-shot generalization**: Minimal supervision requirements

---

## 2. Datasets

### 2.1 Confirmed Datasets (Tier 1 - Must Have)

| Dataset | Task | Size | Dimension | Purpose |
|---------|------|------|-----------|---------|
| **MNIST** | Clustering | 70K images (60K train + 10K test) | 28×28=784 | Sanity check, unsupervised learning validation |
| **Fashion-MNIST** | Clustering | 70K images (60K train + 10K test) | 28×28=784 | More challenging clustering, visual features |
| **SIFT1M** | ANN/Hashing | 1M base + 10K queries | 128-dim | Primary hashing benchmark, retrieval evaluation |

### 2.2 Optional Datasets (Tier 2 - If Time Permits)

| Dataset | Task | Size | Dimension | Purpose |
|---------|------|------|-----------|---------|
| **CIFAR-10** (grayscale) | Clustering | 60K images | 32×32×3 → 1024 | More complex visual features |
| **GloVe-100d** | Hashing | 400K words | 100-dim | High-dim real-valued vectors |
| **SIFT10K** | ANN/Hashing | 10K base + 1K queries | 128-dim | Quick prototyping subset |

### 2.3 Data Download Scripts

#### MNIST & Fashion-MNIST
```python
# Automatic download via torchvision
from torchvision import datasets
datasets.MNIST(root='./data', train=True, download=True)
datasets.FashionMNIST(root='./data', train=True, download=True)
```

#### SIFT1M
```bash
mkdir -p data/sift1m && cd data/sift1m
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -zxvf sift.tar.gz
# Files: sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs
```

#### GloVe (Optional)
```bash
mkdir -p data/glove && cd data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# Use glove.6B.100d.txt or glove.6B.300d.txt
```

### 2.4 Dataset Justification

- **MNIST**: Referenced in project doc (Diehl & Cook 2015), standard SNN benchmark
- **Fashion-MNIST**: Similar structure but more challenging, tests generalization
- **SIFT1M**: Standard ANN benchmark, aligns with "hashing feature" requirement
- **Project doc explicitly lists**: SIFT, GloVe, MNIST as target datasets

---

## 3. Baseline Methods

### 3.1 Overview Table

| # | Method | Year | Venue | Learning Rule | Open Source | Priority |
|---|--------|------|-------|--------------|-------------|----------|
| 1 | **STDP-WTA (Diehl & Cook)** | 2015 | Front. Comput. Neurosci. | STDP + Winner-Take-All | ✅ (BindsNET) | **HIGH** |
| 2 | **SoftHebb** | 2022 | NCE | Soft Hebbian WTA | ✅ (Official) | **HIGH** |
| 3 | **Deep STDP (Lu & Sengupta)** | 2024 | NCE | Multi-layer STDP | ⚠️ (Need to verify) | **HIGH** |
| 4 | **FlyHash** | 2017 | Science (concept) | Random projection + WTA | ✅ (PyPI package) | **MEDIUM** |
| 5 | **BioHash** | 2020 | ICML (if exists) | Bio-inspired hashing | ❌ (Need to find/reimplement) | **MEDIUM** |
| 6 | **Lagani et al. Survey** | 2023 | arXiv | Survey (reference methods) | N/A | **LOW** |

### 3.2 Detailed Baseline Descriptions

---

#### **Baseline #1: STDP-WTA (Diehl & Cook 2015)**

**Paper**: 
- Title: "Unsupervised learning of digit recognition using spike-timing-dependent plasticity"
- DOI: 10.3389/fncom.2015.00099
- Link: https://www.frontiersin.org/articles/10.3389/fncom.2015.00099

**Key Idea**:
- Single-layer SNN (784 input → 400 excitatory neurons)
- STDP learning rule with adaptive thresholds
- Winner-Take-All lateral inhibition
- Spike-count-based feature extraction

**Open Source Status**: ✅ **Available**
- **Framework**: BindsNET (PyTorch-based SNN framework)
- **Repo**: https://github.com/BindsNET/bindsnet
- **Installation**:
```bash
pip install bindsnet
```

**Implementation Steps**:

1. **Install BindsNET**:
```bash
conda create -n bindsnet python=3.9
conda activate bindsnet
pip install bindsnet torch torchvision
```

2. **Adapt Diehl & Cook Architecture**:
```python
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre  # STDP rule

# Create network
network = Network()
inpt = Input(n=784, shape=(1, 28, 28))
exc = LIFNodes(n=400, thresh=-52.0, refrac=5)  # Excitatory layer

# STDP connection
conn = Connection(
    source=inpt,
    target=exc,
    wmin=0.0, wmax=1.0,
    update_rule=PostPre,  # STDP
    nu=(1e-4, 1e-2)  # Learning rates
)

network.add_layer(inpt, name="Input")
network.add_layer(exc, name="Excitatory")
network.add_connection(conn, source="Input", target="Excitatory")

# Add lateral inhibition (WTA)
# ... (see BindsNET examples)
```

3. **Training**:
```python
# Run MNIST samples through network
# Collect spike counts per neuron per sample
# spike_counts shape: (n_samples, 400)
```

4. **Feature Extraction**:
```python
# Use spike counts as feature vectors
pre_code = spike_counts  # (n_samples, 400)

# Apply WTA or top-k binarization
def top_k_binarization(x, k):
    code = np.zeros_like(x)
    top_k_idx = np.argsort(x, axis=1)[:, -k:]
    code[np.arange(len(x))[:, None], top_k_idx] = 1
    return code

code = top_k_binarization(pre_code, k=20)  # k=5% of 400
```

5. **Save Features**:
```python
np.save('outputs/codes/diehl_cook/mnist/pre_code_seed0.npy', pre_code)
np.save('outputs/codes/diehl_cook/mnist/code_m400_k20_seed0.npy', code)
```

**Expected Results** (from original paper):
- MNIST classification with linear SVM: ~95% accuracy
- Our task: Use features for clustering/hashing

**Difficulty**: ⭐⭐⭐ (Medium - need to understand BindsNET API)

---

#### **Baseline #2: SoftHebb**

**Paper**:
- Title: "SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks"
- Year: 2022 (NCE), 2023 (ICLR Oral extension)
- DOI: 10.1088/2634-4386/ac98a9
- Link: https://iopscience.iop.org/article/10.1088/2634-4386/ac98a9

**Key Idea**:
- Probabilistic Hebbian learning (Bayesian framework)
- Soft Winner-Take-All (differentiable)
- Multi-layer architecture possible
- Learns sparse distributed representations

**Open Source Status**: ✅ **Official Repository Available**
- **Repo**: https://github.com/NeuromorphicComputing/SoftHebb
- **Note**: Uses older PyTorch version (1.7.1), recommend isolated environment

**Implementation Steps**:

1. **Clone & Setup Isolated Environment**:
```bash
git clone https://github.com/NeuromorphicComputing/SoftHebb baselines/softhebb/SoftHebb
cd baselines/softhebb/SoftHebb

# Create isolated conda environment
conda create -n softhebb python=3.8
conda activate softhebb
pip install torch==1.7.1 torchvision==0.8.2 numpy matplotlib
```

2. **Run Original Demo**:
```bash
python demo.py  # Verify installation works
```

3. **Adapt for Feature Extraction**:
```python
# Create wrapper script: baselines/softhebb/extract_features.py

import torch
import numpy as np
from SoftHebb.model import SoftHebbNet  # Adjust import based on repo structure

# Load pretrained model or train new
model = SoftHebbNet(input_dim=784, hidden_dims=[500, 200])
model.train_on_dataset(mnist_loader)

# Extract features
def extract_features(model, dataloader):
    pre_codes = []
    for images, _ in dataloader:
        with torch.no_grad():
            z = model.encode(images)  # Get latent representation
            pre_codes.append(z.cpu().numpy())
    return np.vstack(pre_codes)

pre_code = extract_features(model, test_loader)
np.save('../../outputs/codes/softhebb/mnist/pre_code_seed0.npy', pre_code)

# Binarize
code = top_k_binarization(pre_code, k=int(0.05 * pre_code.shape[1]))
np.save('../../outputs/codes/softhebb/mnist/code_m200_k10_seed0.npy', code)
```

4. **Integration Strategy**:
- Train in isolated `softhebb` environment
- Export features as `.npy` files
- Load into main pipeline for evaluation (no dependency conflicts)

**Expected Results**: 
- Original paper reports strong performance on unsupervised learning tasks
- Check paper for specific metrics

**Difficulty**: ⭐⭐⭐⭐ (Medium-High - environment isolation, need to understand codebase)

---

#### **Baseline #3: Deep STDP (Lu & Sengupta 2024)**

**Paper**:
- Title: "Deep unsupervised learning using spike-timing-dependent plasticity"
- Year: 2024
- Venue: Neuromorphic Computing and Engineering
- DOI: 10.1088/2634-4386/ad5e6d
- Link: https://iopscience.iop.org/article/10.1088/2634-4386/ad5e6d

**Key Idea**:
- Multi-layer STDP learning (extends Diehl & Cook to deep networks)
- Layer-wise unsupervised learning
- Suitable for hierarchical feature learning

**Open Source Status**: ⚠️ **NEED TO VERIFY**
- **Action Required**: Check paper supplementary materials for code
- **Likely Options**:
  1. Authors provide GitHub link in paper
  2. Code available upon request
  3. Need to reimplement based on paper description

**Implementation Steps**:

**Step 1: Search for Code**
```bash
# Search strategies:
# 1. Check paper PDF for "Code availability" section
# 2. Search GitHub: "Lu Sengupta deep STDP 2024"
# 3. Check author webpages
# 4. Email authors if necessary
```

**Step 2a: If Code Available**
```bash
git clone [REPO_URL] baselines/lu_sengupta_2024/
cd baselines/lu_sengupta_2024/
# Follow their README
```

**Step 2b: If Code NOT Available - Reimplement**
```python
# Reimplement using BindsNET or SpikingJelly
# Key components based on paper:
# 1. Multi-layer SNN architecture
# 2. Layer-wise STDP training
# 3. Lateral inhibition per layer

# Pseudo-structure:
class DeepSTDP:
    def __init__(self, layer_sizes=[784, 500, 300, 100]):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            layer = STDPLayer(
                n_input=layer_sizes[i],
                n_output=layer_sizes[i+1],
                learning_rule='STDP'
            )
            self.layers.append(layer)
    
    def train_layer(self, layer_idx, data):
        # Train one layer at a time
        pass
    
    def extract_features(self, data, layer_idx=-1):
        # Forward pass to specified layer
        pass
```

**Step 3: Feature Extraction** (similar to other baselines)
```python
model = DeepSTDP()
model.train_layerwise(mnist_loader)
pre_code = model.extract_features(test_loader, layer_idx=-1)
# Save and binarize as before
```

**Priority**: **HIGH** - Explicitly mentioned in project document

**Difficulty**: ⭐⭐⭐⭐⭐ (High - may need reimplementation)

**Timeline Estimate**:
- If code available: 1-2 weeks integration
- If reimplementation needed: 3-4 weeks

---

#### **Baseline #4: FlyHash**

**Paper (Concept)**:
- Title: "A neural algorithm for a fundamental computing problem"
- Year: 2017, Venue: Science
- DOI: 10.1126/science.aam9868
- Link: https://science.sciencemag.org/content/358/6364/793

**Key Idea**:
- Fruit fly olfactory circuit-inspired hashing
- Random projection + Winner-Take-All
- Expansion → Sparsification
- Very simple but effective for high-dim data

**Open Source Status**: ✅ **Third-party Python Package Available**
- **Package**: FlyHash (PyPI)
- **Link**: https://pypi.org/project/FlyHash/
- **Note**: This is a third-party implementation, not official from the paper authors

**Implementation Steps**:

1. **Install**:
```bash
pip install FlyHash
```

2. **Basic Usage**:
```python
from flyhash import FlyHash

# Initialize
fly = FlyHash(
    input_dim=784,      # MNIST flattened
    projection_dim=2000, # Expansion factor ~2.5x
    hash_length=20,     # Number of "winner" neurons
    sampling_ratio=0.1  # 10% of projection neurons have connections
)

# Encode data
mnist_flat = mnist_images.reshape(-1, 784)  # (n_samples, 784)
codes = fly.hash(mnist_flat)  # (n_samples, projection_dim) sparse binary
```

3. **Extract Both Formats**:
```python
# Pre-WTA (for analysis)
pre_code = fly.project(mnist_flat)  # Before WTA, (n_samples, 2000)

# Post-WTA (binary code)
code = fly.hash(mnist_flat)  # After WTA, (n_samples, 2000) sparse

# Save
np.save('outputs/codes/flyhash/mnist/pre_code_seed0.npy', pre_code)
np.save('outputs/codes/flyhash/mnist/code_m2000_k20_seed0.npy', code)
```

4. **Apply to All Datasets**:
```python
# MNIST
fly_mnist = FlyHash(input_dim=784, projection_dim=2000, hash_length=20)

# SIFT1M
fly_sift = FlyHash(input_dim=128, projection_dim=320, hash_length=16)

# GloVe
fly_glove = FlyHash(input_dim=100, projection_dim=250, hash_length=12)
```

**Expected Results**:
- Original paper shows superior performance on odor similarity
- For our tasks: Good baseline for "random + sparse" approach

**Difficulty**: ⭐ (Very Easy - just a pip install)

**Priority**: MEDIUM (good for quick sanity check, but less biologically detailed than STDP)

---

#### **Baseline #5: BioHash / Bio-inspired Hashing Methods**

**Status**: 🔍 **NEEDS INVESTIGATION**

**Action Items**:
1. Search for ICML 2020 papers with keywords: "bio-inspired", "hashing", "spiking"
2. Check if there's a method called "BioHash" or similar
3. Possible candidates:
   - "Learning to Hash with Binary Deep Neural Networks" (if has bio-inspired variant)
   - Any neuromorphic hashing papers from ICML/NeurIPS 2019-2021

**If Found**:
- Follow same pattern: check for code → integrate or reimplement

**If Not Found**:
- Skip or replace with another bio-inspired baseline
- Alternatives:
  - "Deep Hashing with SNNs" (if such paper exists)
  - Extend FlyHash with more biological details

**Priority**: MEDIUM (nice-to-have, not critical for Phase 1)

---

#### **Baseline #6: Reference from Lagani et al. Survey (2023)**

**Paper**:
- Title: "Spiking neural networks and bio-inspired supervised deep learning: a survey"
- Link: https://arxiv.org/abs/2307.16235

**Purpose**: 
- This is a **survey paper**, not a method itself
- Use it to find additional baseline methods
- Check Section 4-5 for unsupervised learning methods

**Action**:
- Read relevant sections
- Extract 1-2 additional methods if they have code available

**Priority**: LOW (reference material)

---

## 4. Unified Pipeline Architecture

### 4.1 Directory Structure

```
clustering_hashing_pipeline/
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── mnist/                    # Auto-downloaded
│   ├── fashion_mnist/            # Auto-downloaded
│   ├── sift1m/                   # Manual download
│   └── glove/                    # Optional
│
├── baselines/
│   ├── diehl_cook/               # BindsNET-based STDP
│   │   ├── train.py
│   │   ├── extract_features.py
│   │   └── config.yaml
│   │
│   ├── softhebb/                 # Isolated environment
│   │   ├── SoftHebb/             # Git submodule
│   │   ├── extract_features.py
│   │   └── environment.yml
│   │
│   ├── lu_sengupta_2024/         # Deep STDP
│   │   ├── (code from authors or reimplementation)
│   │   └── ...
│   │
│   ├── flyhash/                  # FlyHash wrapper
│   │   ├── train.py
│   │   └── config.yaml
│   │
│   └── base_encoder.py           # Abstract interface
│
├── pipeline/
│   ├── __init__.py
│   ├── datasets.py               # Unified data loading
│   ├── encoders.py               # Wrapper for all baselines
│   ├── binarization.py           # WTA, top-k, threshold methods
│   ├── clustering.py             # K-means, k-medoids, spectral
│   ├── retrieval.py              # mAP, Recall@K, Precision@K
│   ├── metrics.py                # NMI, ARI, ACC, Silhouette, DB
│   └── utils.py                  # Helpers, seed setting, etc.
│
├── scripts/
│   ├── download_sift1m.sh
│   ├── download_glove.sh
│   ├── run_baseline.py           # Single baseline runner
│   └── run_all_baselines.sh      # Batch runner
│
├── configs/
│   ├── default.yaml              # Global settings
│   ├── diehl_cook.yaml
│   ├── softhebb.yaml
│   ├── lu_sengupta.yaml
│   └── flyhash.yaml
│
├── outputs/
│   ├── codes/                    # Cached features
│   │   ├── diehl_cook/
│   │   │   ├── mnist/
│   │   │   │   ├── pre_code_seed0.npy
│   │   │   │   └── code_m400_k20_seed0.npy
│   │   │   └── sift1m/
│   │   ├── softhebb/
│   │   └── ...
│   │
│   ├── results/                  # Evaluation results
│   │   ├── clustering_mnist.csv
│   │   ├── retrieval_sift1m.csv
│   │   └── summary_table.csv
│   │
│   └── logs/                     # Training logs
│
├── notebooks/                    # Analysis & visualization
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_comparison.ipynb
│   └── 03_ablation_studies.ipynb
│
└── docs/
    ├── clustering_hashing_baseline_guide.md  # This document
    └── evaluation_protocol.md
```

### 4.2 Unified Encoder Interface

```python
# baselines/base_encoder.py

from abc import ABC, abstractmethod
import numpy as np

class BaseEncoder(ABC):
    """
    Abstract base class for all baseline encoders.
    Ensures consistent API across different methods.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.is_trained = False
    
    @abstractmethod
    def fit(self, train_data: np.ndarray, train_labels: np.ndarray = None):
        """
        Train the encoder (unsupervised, labels only for analysis).
        
        Args:
            train_data: (n_samples, input_dim) numpy array
            train_labels: (n_samples,) optional, for analysis only
        """
        pass
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> dict:
        """
        Encode data into representations.
        
        Args:
            data: (n_samples, input_dim) numpy array
        
        Returns:
            dict with keys:
                - 'pre_code': (n_samples, code_dim) continuous representation
                - 'code': (n_samples, code_dim) sparse/binary code
        """
        pass
    
    def save(self, path: str):
        """Save trained model."""
        pass
    
    def load(self, path: str):
        """Load trained model."""
        pass
```

**Example Implementation**:

```python
# baselines/flyhash/encoder.py

from baselines.base_encoder import BaseEncoder
from flyhash import FlyHash
import numpy as np

class FlyHashEncoder(BaseEncoder):
    def __init__(self, config: dict):
        super().__init__(config)
        self.fly = FlyHash(
            input_dim=config['input_dim'],
            projection_dim=config['projection_dim'],
            hash_length=config['hash_length'],
            sampling_ratio=config.get('sampling_ratio', 0.1)
        )
        self.is_trained = True  # FlyHash doesn't need training
    
    def fit(self, train_data: np.ndarray, train_labels: np.ndarray = None):
        # FlyHash is non-parametric, no training needed
        pass
    
    def encode(self, data: np.ndarray) -> dict:
        pre_code = self.fly.project(data)
        code = self.fly.hash(data)
        return {
            'pre_code': pre_code,
            'code': code
        }
```

### 4.3 Unified Evaluation Pipeline

```python
# scripts/run_baseline.py

import argparse
import yaml
import numpy as np
from pathlib import Path

from pipeline.datasets import load_dataset
from pipeline.encoders import get_encoder
from pipeline.clustering import evaluate_clustering
from pipeline.retrieval import evaluate_retrieval
from pipeline.utils import set_seed, save_results

def main(args):
    # Load config
    config = yaml.safe_load(open(args.config))
    set_seed(config['seed'])
    
    # Load dataset
    dataset = load_dataset(
        name=config['dataset'],
        root=config['data_root']
    )
    
    # Initialize encoder
    encoder = get_encoder(
        name=config['encoder'],
        config=config['encoder_config']
    )
    
    # Train (if needed)
    if not encoder.is_trained:
        print(f"Training {config['encoder']}...")
        encoder.fit(dataset['train_data'], dataset['train_labels'])
    
    # Encode test data
    print(f"Encoding test data...")
    output_dir = Path(config['output_dir']) / 'codes' / config['encoder'] / config['dataset']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not (output_dir / f"code_seed{config['seed']}.npy").exists():
        encoded = encoder.encode(dataset['test_data'])
        np.save(output_dir / f"pre_code_seed{config['seed']}.npy", encoded['pre_code'])
        np.save(output_dir / f"code_seed{config['seed']}.npy", encoded['code'])
    else:
        print("Loading cached codes...")
        encoded = {
            'pre_code': np.load(output_dir / f"pre_code_seed{config['seed']}.npy"),
            'code': np.load(output_dir / f"code_seed{config['seed']}.npy")
        }
    
    # Evaluate
    results = {}
    
    if config['eval_clustering']:
        print("Evaluating clustering...")
        clustering_results = evaluate_clustering(
            codes=encoded['code'],
            labels=dataset['test_labels'],
            n_clusters=config['n_clusters'],
            methods=config.get('clustering_methods', ['kmeans'])
        )
        results['clustering'] = clustering_results
    
    if config['eval_retrieval']:
        print("Evaluating retrieval...")
        retrieval_results = evaluate_retrieval(
            codes=encoded['code'],
            queries=dataset.get('query_data'),
            groundtruth=dataset.get('groundtruth'),
            k_values=config.get('k_values', [10, 50, 100])
        )
        results['retrieval'] = retrieval_results
    
    # Save results
    save_results(results, output_dir / f"results_seed{config['seed']}.json")
    print(f"Results saved to {output_dir}")
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)
```

---

## 5. Evaluation Protocol

### 5.1 Clustering Evaluation (MNIST, Fashion-MNIST)

**Setup**:
- Input: Binary/sparse codes from encoder
- Ground truth: Dataset labels (used only for evaluation, not training)
- Number of clusters: Equal to number of classes (10 for MNIST/Fashion-MNIST)

**Algorithms**:
1. **K-Medoids** (recommended for binary codes with Hamming distance)
2. **Spectral Clustering** (works with arbitrary distance metrics)
3. **Hierarchical Clustering** (for comparison)
4. **K-Means** (only for continuous pre_code)

**Metrics**:
```python
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# NMI (Normalized Mutual Information)
nmi = normalized_mutual_info_score(labels_true, labels_pred)

# ARI (Adjusted Rand Index)
ari = adjusted_rand_score(labels_true, labels_pred)

# ACC (Accuracy with Hungarian matching)
def clustering_accuracy(y_true, y_pred):
    contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(y_true)
    return accuracy

acc = clustering_accuracy(labels_true, labels_pred)
```

**Reporting**:
- Run with 5 different random seeds
- Report: mean ± std for each metric
- Example: NMI = 0.756 ± 0.012

### 5.2 Retrieval Evaluation (SIFT1M)

**Setup**:
- Database: SIFT1M base vectors (1M)
- Queries: SIFT1M query vectors (10K)
- Ground truth: Provided in `sift_groundtruth.ivecs`

**Protocol**:
1. Encode both database and queries using same encoder
2. For each query, find K nearest neighbors in database (using Hamming distance for binary codes)
3. Compare with ground truth neighbors

**Metrics**:
```python
def mean_average_precision(retrieved, groundtruth, k=100):
    """
    Args:
        retrieved: (n_queries, k) indices of retrieved neighbors
        groundtruth: (n_queries, n_gt) indices of true neighbors
    """
    aps = []
    for i in range(len(retrieved)):
        gt_set = set(groundtruth[i])
        hits = 0
        precisions = []
        for j, idx in enumerate(retrieved[i][:k]):
            if idx in gt_set:
                hits += 1
                precisions.append(hits / (j + 1))
        ap = np.mean(precisions) if precisions else 0.0
        aps.append(ap)
    return np.mean(aps)

def recall_at_k(retrieved, groundtruth, k):
    """
    Fraction of true neighbors found in top-k retrieved.
    """
    recalls = []
    for i in range(len(retrieved)):
        gt_set = set(groundtruth[i])
        retrieved_set = set(retrieved[i][:k])
        recall = len(gt_set & retrieved_set) / len(gt_set)
        recalls.append(recall)
    return np.mean(recalls)

# Compute
mAP = mean_average_precision(retrieved, groundtruth, k=100)
R10 = recall_at_k(retrieved, groundtruth, k=10)
R50 = recall_at_k(retrieved, groundtruth, k=50)
R100 = recall_at_k(retrieved, groundtruth, k=100)
```

**Reporting**:
- Report: mAP, Recall@10, Recall@50, Recall@100
- Also report precision@K for comparison

### 5.3 SNN-Specific Metrics (Important!)

**Spike Sparsity**:
```python
def compute_spike_sparsity(spike_counts, n_neurons, n_timesteps):
    """
    Args:
        spike_counts: Total spikes across all neurons and timesteps
        n_neurons: Number of neurons in the network
        n_timesteps: Simulation time steps
    """
    max_possible_spikes = n_neurons * n_timesteps
    sparsity = 1.0 - (spike_counts / max_possible_spikes)
    return sparsity

# Report average sparsity across test set
```

**Energy Efficiency (if applicable)**:
```python
# Compare number of operations vs. equivalent ANN
def compute_ops(snn_spikes, ann_params):
    """
    SNN: Each spike = 1 MAC operation
    ANN: Each weight = 1 MAC operation per forward pass
    """
    snn_ops = np.sum(snn_spikes)  # Total spikes
    ann_ops = ann_params  # Total weights × activations
    efficiency_ratio = ann_ops / snn_ops
    return efficiency_ratio
```

**Temporal Dynamics** (if using temporal coding):
```python
# Analyze spike timing patterns
def analyze_spike_timing(spike_trains):
    """
    - First-spike latency distribution
    - Inter-spike intervals
    - Temporal precision
    """
    pass
```

### 5.4 Fairness & Reproducibility

**Fixed Hyperparameters** (align across all baselines):
- Code length (m): 100, 200, 400 (report all)
- Sparsity level (k or top-k%): 5%, 10%, 20%
- Distance metric: Hamming for binary, Cosine for real-valued
- Number of clusters: 10 (MNIST/Fashion-MNIST)

**Random Seeds**:
- Use seeds: 0, 1, 2, 3, 4
- Report mean ± std

**Timing**:
- Report training time
- Report encoding time per sample
- Use same hardware for all baselines (note GPU model)

---

## 6. Implementation Timeline & Milestones

### Phase 1: Setup & Quick Win (Week 1-2)

**Goals**:
- ✅ Setup repository structure
- ✅ Download datasets (MNIST, Fashion-MNIST, SIFT1M)
- ✅ Implement unified pipeline (datasets.py, metrics.py)
- ✅ Integrate FlyHash (easiest baseline)
- ✅ Run end-to-end test on MNIST

**Deliverables**:
- Working code repository
- FlyHash results on MNIST clustering
- Evaluation metrics verified

### Phase 2: Core Baselines (Week 3-5)

**Goals**:
- ✅ Integrate Diehl & Cook (BindsNET)
- ✅ Integrate SoftHebb (isolated environment)
- ✅ Investigate Lu & Sengupta 2024 (find code or plan reimplementation)
- ✅ Run all baselines on MNIST + Fashion-MNIST

**Deliverables**:
- 3-4 baselines integrated
- Clustering results table (NMI/ARI/ACC)
- Feature codes cached for all baselines

### Phase 3: Hashing Evaluation (Week 6-7)

**Goals**:
- ✅ Implement retrieval evaluation (mAP, Recall@K)
- ✅ Run all baselines on SIFT1M (or SIFT10K subset first)
- ✅ Optimize for performance if needed

**Deliverables**:
- Retrieval results table
- Performance comparison plots

### Phase 4: Analysis & Documentation (Week 8)

**Goals**:
- ✅ Statistical analysis (significance tests)
- ✅ Ablation studies (code length, sparsity, distance metrics)
- ✅ Visualizations (t-SNE, confusion matrices, retrieval curves)
- ✅ Write report summarizing baseline performance

**Deliverables**:
- **Final baseline performance report** (PDF/Markdown)
- Plots and tables for paper/presentation
- Clean, documented codebase

### Phase 5: Integration with LLM-Generated Architectures (Week 9+)

**Goals**:
- ✅ Adapt pipeline to evaluate LLM-generated SNN architectures
- ✅ Run comparative evaluation
- ✅ Provide feedback for Step 7 (LLM reflection)

**Deliverables**:
- Comparative results: Baselines vs. LLM-generated models
- Analysis of which architectural features improve clustering/hashing

---

## 7. GPU Resource Requirements

### 7.1 Realistic Assessment

| Phase | Task | Min GPU | Recommended GPU | Duration |
|-------|------|---------|-----------------|----------|
| Phase 1 | FlyHash (CPU only) | - | - | 2 days |
| Phase 2 | BindsNET training | 8GB | 16GB | 1-2 days per seed |
| Phase 2 | SoftHebb training | 16GB | 24GB | 2-3 days per config |
| Phase 2 | Lu & Sengupta | 16GB | 24GB | 3-5 days (if reimpl.) |
| Phase 3 | SIFT1M encoding | 8GB | 16GB | 1 day |
| Phase 4 | Analysis (CPU) | - | - | 3-5 days |

**Overall Recommendation**:
- **Minimum**: 1×16GB GPU (tight, sequential training)
- **Comfortable**: 1×24GB GPU (allows larger batches, faster iteration)
- **Ideal**: 2×24GB or 1×40GB+ (parallel experiments, Phase 5 integration)

**Cost-Saving Tips**:
- Use CPU for FlyHash
- Cache features aggressively (encode once, evaluate many times)
- Run hyperparameter search on downscaled data first
- Use cloud GPU only when training, do analysis on CPU

### 7.2 Compute Budget Estimate

Assuming 1×24GB GPU on cloud platform (e.g., AWS p3.2xlarge = $3/hour):

| Task | Hours | Cost |
|------|-------|------|
| Phase 1: Setup & FlyHash | 10 | $30 |
| Phase 2: Train 3 baselines × 5 seeds | 80 | $240 |
| Phase 3: SIFT1M experiments | 20 | $60 |
| Phase 5: LLM models (estimate) | 40 | $120 |
| **Total** | **150** | **$450** |

**Note**: This is conservative estimate; actual may be lower with caching and optimization.

---

## 8. Success Criteria

### 8.1 Minimum Viable Product (MVP)

By end of Phase 3, you must have:

✅ **Code**:
- 3+ baselines integrated (FlyHash, Diehl & Cook, SoftHebb)
- Unified evaluation pipeline
- Cached features for all baselines

✅ **Results**:
- Clustering results on MNIST (NMI/ARI/ACC for all baselines)
- Retrieval results on SIFT1M or SIFT10K subset

✅ **Documentation**:
- README with setup instructions
- Config files for reproducibility
- Results table (even if preliminary)

### 8.2 Full Delivery

By end of Phase 4:

✅ **Code**:
- 4-5 baselines (including Lu & Sengupta if possible)
- Complete evaluation on all datasets
- Statistical analysis & visualization scripts

✅ **Report** (5-10 pages):
1. **Introduction**: Task definition, biological motivation
2. **Methods**: Baseline descriptions, datasets, evaluation protocol
3. **Results**: Tables + plots for clustering and retrieval
4. **Analysis**: 
   - Which methods work best for clustering vs. hashing?
   - Role of sparsity and code length
   - SNN-specific advantages (sparsity, energy)
5. **Discussion**: Insights for LLM architecture generation

✅ **Deliverables for Project**:
- Performance baseline numbers for comparison
- Evaluation codebase for Phase 5 (LLM models)
- Recommendations: Which SNN features to prioritize?

---

## 9. Risk Mitigation

### 9.1 Identified Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Lu & Sengupta code unavailable | High | Medium | Plan reimplementation OR find alternative baseline |
| SoftHebb environment conflicts | Medium | High | Use isolated conda env + feature export strategy |
| SIFT1M too large for initial tests | Low | Medium | Start with SIFT10K subset for prototyping |
| Baseline performance too low | Medium | Low | Focus on relative comparison, not absolute numbers |
| Timeline delay | High | Medium | Prioritize MVP (3 baselines), defer nice-to-haves |

### 9.2 Contingency Plans

**If Lu & Sengupta 2024 is too hard to implement**:
- **Option A**: Email authors for code
- **Option B**: Simplify to single-layer deep STDP variant
- **Option C**: Replace with another STDP paper (e.g., Tavanaei et al. 2019)

**If GPU resources run out**:
- **Option A**: Reduce number of random seeds (3 instead of 5)
- **Option B**: Train on MNIST only, use SIFT10K instead of SIFT1M
- **Option C**: Request additional allocation or use personal GPU

**If timeline too tight**:
- **Cut**: GloVe dataset, BioHash baseline, extensive ablations
- **Keep**: MNIST + SIFT1M, 3 core baselines, basic evaluation

---

## 10. References & Resources

### 10.1 Key Papers

1. **Diehl & Cook (2015)**: "Unsupervised learning of digit recognition using spike-timing-dependent plasticity"
   - https://www.frontiersin.org/articles/10.3389/fncom.2015.00099

2. **SoftHebb (2022)**: "SoftHebb: Bayesian inference in unsupervised hebbian soft winner-take-all networks"
   - https://iopscience.iop.org/article/10.1088/2634-4386/ac98a9

3. **Lu & Sengupta (2024)**: "Deep unsupervised learning using spike-timing-dependent plasticity"
   - https://iopscience.iop.org/article/10.1088/2634-4386/ad5e6d

4. **FlyHash (2017)**: "A neural algorithm for a fundamental computing problem"
   - https://science.sciencemag.org/content/358/6364/793

5. **Lagani et al. (2023)**: "Spiking neural networks and bio-inspired supervised deep learning: a survey"
   - https://arxiv.org/abs/2307.16235

### 10.2 Code Repositories

- **BindsNET**: https://github.com/BindsNET/bindsnet
- **SoftHebb**: https://github.com/NeuromorphicComputing/SoftHebb (needs verification)
- **FlyHash (PyPI)**: https://pypi.org/project/FlyHash/
- **SpikingJelly**: https://github.com/fangwei123456/spikingjelly (alternative SNN framework)

### 10.3 Datasets

- **MNIST**: http://yann.lecun.com/exdb/mnist/
- **Fashion-MNIST**: https://github.com/zalandoresearch/fashion-mnist
- **SIFT1M**: ftp://ftp.irisa.fr/local/texmex/corpus/
- **GloVe**: https://nlp.stanford.edu/projects/glove/

### 10.4 Useful Tools

- **FAISS**: https://github.com/facebookresearch/faiss (fast similarity search)
- **Scikit-learn**: Clustering algorithms and metrics
- **Matplotlib/Seaborn**: Visualization
- **Weights & Biases**: Experiment tracking (optional but recommended)

---

## 11. Next Steps (Action Items)

### Immediate (This Week):
1. [ ] **Verify document alignment**: Share this document with project supervisor, confirm task scope
2. [ ] **Setup repository**: Initialize Git repo with directory structure (Section 4.1)
3. [ ] **Download datasets**: MNIST, Fashion-MNIST, SIFT1M (Section 2.3)
4. [ ] **Environment setup**: Create conda environments for main pipeline and SoftHebb

### Week 1-2 (Phase 1):
5. [ ] **Implement `pipeline/datasets.py`**: Unified data loaders
6. [ ] **Implement `pipeline/metrics.py`**: NMI, ARI, ACC, mAP, Recall@K
7. [ ] **Integrate FlyHash**: Baseline #4, easiest starting point
8. [ ] **Run sanity check**: FlyHash on MNIST, verify metrics are correct

### Week 3-5 (Phase 2):
9. [ ] **Integrate Diehl & Cook**: BindsNET-based implementation
10. [ ] **Integrate SoftHebb**: Isolated environment + feature export
11. [ ] **Investigate Lu & Sengupta**: Find code or start reimplementation
12. [ ] **Run clustering evaluation**: All baselines on MNIST + Fashion-MNIST

### Week 6-8 (Phase 3-4):
13. [ ] **Implement retrieval evaluation**: mAP, Recall@K on SIFT1M
14. [ ] **Run full evaluation**: All baselines, all datasets, 5 seeds
15. [ ] **Statistical analysis**: Significance tests, ablation studies
16. [ ] **Write baseline report**: Document results for project

### Week 9+ (Phase 5):
17. [ ] **Adapt pipeline for LLM models**: Evaluate generated architectures
18. [ ] **Comparative analysis**: Baselines vs. LLM-generated models
19. [ ] **Provide feedback for Step 7**: Which features work best?

---

## 12. Contact & Collaboration

### Questions to Clarify with Project Team:

1. **Task Priority**: Is clustering or hashing more important? Equal weight?
2. **Dataset Confirmation**: Are MNIST/Fashion-MNIST/SIFT1M the final datasets?
3. **Timeline**: What is the hard deadline for Phase 4 (baseline results)?
4. **GPU Access**: What GPU resources are currently available?
5. **Code Integration**: Should this pipeline integrate with other team members' code?
6. **Publication Plan**: Are we targeting a specific venue? What's the timeline?

### Collaboration Points:

- **Step 2-3 (NAS team)**: They need your evaluation code to test generated architectures
- **Step 6 (Training team)**: Share cached features and evaluation protocol
- **Step 7 (LLM reflection)**: Provide insights on which architectural features matter

---

## Appendix A: Quick Start Commands

```bash
# 1. Clone repository (replace with actual URL)
git clone [YOUR_REPO_URL] clustering_hashing_pipeline
cd clustering_hashing_pipeline

# 2. Setup main environment
conda env create -f environment.yml
conda activate clustering_pipeline

# 3. Download datasets
bash scripts/download_sift1m.sh
python -c "from torchvision import datasets; datasets.MNIST('./data', download=True); datasets.FashionMNIST('./data', download=True)"

# 4. Install FlyHash
pip install FlyHash

# 5. Install BindsNET
pip install bindsnet

# 6. Setup SoftHebb (isolated environment)
cd baselines/softhebb
conda env create -n softhebb -f environment.yml
cd ../..

# 7. Run FlyHash baseline on MNIST
python scripts/run_baseline.py --config configs/flyhash.yaml

# 8. Run all baselines
bash scripts/run_all_baselines.sh

# 9. Generate results table
python scripts/summarize_results.py --output outputs/results/summary_table.csv
```

---

## Appendix B: Example Config File

```yaml
# configs/diehl_cook.yaml

# Experiment settings
experiment_name: "diehl_cook_mnist"
seed: 0

# Dataset
dataset: "mnist"
data_root: "./data"

# Encoder settings
encoder: "diehl_cook"
encoder_config:
  n_input: 784
  n_excitatory: 400
  n_inhibitory: 400
  learning_rate: [1e-4, 1e-2]
  simulation_time: 350  # ms
  dt: 1.0  # ms
  thresh: -52.0
  refrac: 5  # ms

# Feature extraction
code_length: 400
sparsity_level: 0.05  # Top 5% neurons

# Evaluation
eval_clustering: true
eval_retrieval: false
n_clusters: 10
clustering_methods: ["kmedoids", "spectral"]

# Output
output_dir: "./outputs"
save_model: true
save_codes: true
```

---

**Document Version**: 1.0  
**Author**: Generated for Jingze Gai (Clustering/Hashing Track)  
**Status**: Ready for Review and Implementation

**Next Step**: Share with supervisor, get approval, then start Phase 1 implementation! 🚀
