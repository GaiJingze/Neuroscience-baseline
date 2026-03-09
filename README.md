# Neuroscience-baseline

Unified evaluation pipeline for biologically-inspired unsupervised feature learning baselines (SNN / Hebbian / hashing).

## Baselines

| Baseline | Method | Training |
|---|---|---|
| **FlyHash** | Random projection + WTA | Instant |
| **Diehl & Cook** | STDP + lateral inhibition (BindsNET) | ~6 h (60K, GPU) |
| **Deep STDP** | Multi-layer STDP + K-means bootstrapping | ~8 h (60K, GPU) |
| **LC-SNN** | Locally-competitive SNN (patch-based STDP) | ~4 h (60K, GPU) |
| **LM-SNN** | Laterally-modulated SNN (topological inhibition) | ~6 h (60K, GPU) |
| **CSDP** | Contrastive signal-dependent plasticity (Hebbian SNN) | ~3 min (5K, CPU) |
| **SoftHebb** | Hebbian + Soft-WTA | ~2 min |
| **Krotov** | Hebbian + WTA | ~1 min |
| **BioHash** | Hebbian + sparse projection | ~2 min |
| **WTA Hash** | Random windowing + local WTA | Instant |
| **SOM** | Competitive learning | ~5 min |
| **LSH / SimHash** | Random hyperplanes | Instant |

## Datasets

| Dataset | Dim | Clustering | Retrieval |
|---|---|---|---|
| MNIST | 784 | ACC, NMI, ARI | mAP, Recall@K |
| Fashion-MNIST | 784 | ACC, NMI, ARI | mAP, Recall@K |
| SIFT1M | 128 | — | mAP, Recall@K |

## Quick Start

### 1. Environment setup

```bash
# Create and activate conda environment (Python 3.9 – 3.11)
conda create -n neuro-baseline python=3.10 -y
conda activate neuro-baseline

# Install PyTorch (pick the command matching your CUDA version)
# See https://pytorch.org/get-started/locally/ for other options
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (includes BindsNET from GitHub)
pip install -r requirements.txt
```

> **Tip**: If `pip install -r requirements.txt` fails on the BindsNET line
> and you don't need the Diehl & Cook baseline, comment out the
> `git+https://...bindsnet.git` line and install the rest.

### 2. Verify installation

```bash
python -c "
import torch, torchvision, numpy, sklearn, bindsnet
print(f'PyTorch {torch.__version__}  CUDA {torch.cuda.is_available()}')
print(f'BindsNET {bindsnet.__version__}')
print('All OK')
"
```

### 3. Download datasets

```bash
# MNIST & Fashion-MNIST (auto-downloaded on first use, or manually):
python -c "
from torchvision import datasets
datasets.MNIST('./data', download=True)
datasets.FashionMNIST('./data', download=True)
"

# SIFT1M (~400 MB, needed only for retrieval evaluation):
bash scripts/download_sift1m.sh
```

### 4. Run a single baseline

```bash
# Using run.py entry point
python run.py --baseline flyhash --dataset mnist --seed 0

# Or with a config file
python scripts/run_baseline.py --config configs/diehl_cook.yaml

# Diehl & Cook sanity check (fast, trains on 500 samples)
python scripts/test_diehl_cook_sanity.py
```

### 5. Run ALL baselines on ALL datasets (one seed)

```bash
python scripts/run_benchmark.py --seeds 0
```

This runs every available baseline on `mnist`, `fashion_mnist`, and `sift1m`
with seed 0 and prints a summary table at the end.  Results are saved to
`outputs/benchmark/`.

To exclude the slow Diehl & Cook baseline, use `--quick` (MNIST only,
skips `diehl_cook`).  To pick specific combinations:

```bash
python scripts/run_benchmark.py \
    --baselines flyhash krotov softhebb \
    --datasets mnist fashion_mnist \
    --seeds 0 1 2
```

### 6. Run the full benchmark script

A convenience shell script wraps the above with error handling:

```bash
bash scripts/run_all_baselines.sh
```

## Encoder Interface Specification

All baseline encoders inherit from `BaseEncoder` (`baselines/base_encoder.py`)
and follow a unified input/output contract. This makes it easy to swap
encoders in the pipeline and to add new tasks (clustering, retrieval,
visualisation, etc.) without per-encoder adaptation.

### Config (`__init__`)

```python
config = {
    'input_dim':  784,   # (required) dimensionality of one input sample
    'output_dim': 256,   # (required) dimensionality of the output code
    # ... encoder-specific parameters (learning rate, neuron count, etc.)
}
encoder = SomeEncoder(config)
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `input_dim` | int | Yes | Flattened input dimension (e.g. 784 for MNIST) |
| `output_dim` | int | Yes | Output code dimension (pre_code & code width) |
| *others* | — | No | Encoder-specific, with sensible defaults |

### Training (`fit`)

```python
encoder.fit(train_data, train_labels=None)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `train_data` | `np.ndarray` float32 | `(n_samples, input_dim)` | Training samples, values in [0, 1] |
| `train_labels` | `np.ndarray` int, optional | `(n_samples,)` | Ground-truth labels — **not used for training** (unsupervised). For logging / analysis only. |

Sets `self.is_trained = True` on completion.

### Encoding (`encode`)

```python
result = encoder.encode(data)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `data` (input) | `np.ndarray` float32 | `(n_samples, input_dim)` | Samples to encode |
| `result['pre_code']` | `np.ndarray` float32 | `(n_samples, output_dim)` | Continuous representation before binarisation |
| `result['code']` | `np.ndarray` float32 | `(n_samples, output_dim)` | Binary code after binarisation, values in {0, 1} |

> **Key constraint**: `pre_code` and `code` must have the **same shape**.
> Downstream tasks (clustering, hashing retrieval) consume `code`;
> `pre_code` is kept for analysis and alternative metrics.

### Persistence (`save` / `load`)

```python
encoder.save('outputs/model.pkl')
encoder.load('outputs/model.pkl')
```

Encoders with framework-specific state (e.g. PyTorch weights) override
these methods and store additional files alongside the base pickle.

### Minimal Example

```python
import numpy as np
from baselines.flyhash.encoder import FlyHashEncoder

config = {'input_dim': 784, 'output_dim': 256}
encoder = FlyHashEncoder(config)

train_data = np.random.rand(1000, 784).astype(np.float32)
encoder.fit(train_data)

test_data = np.random.rand(200, 784).astype(np.float32)
result = encoder.encode(test_data)

assert result['pre_code'].shape == (200, 256)
assert result['code'].shape     == (200, 256)
assert set(np.unique(result['code'])).issubset({0.0, 1.0})
```

### Data Flow Diagram

```
                        ┌──────────────┐
  np.ndarray            │              │
  (n, input_dim) ──────►│  encoder.fit │  (unsupervised training)
  float32, [0,1]        │              │
                        └──────────────┘
                               │
                        is_trained = True
                               │
                        ┌──────────────┐      ┌──────────────────────────────┐
  np.ndarray            │              │      │ result['pre_code']           │
  (n, input_dim) ──────►│encoder.encode├─────►│   shape: (n, output_dim)     │
  float32, [0,1]        │              │      │   dtype: float32 (continuous)│
                        └──────────────┘      │ result['code']               │
                                              │   shape: (n, output_dim)     │
                                              │   dtype: float32, {0, 1}     │
                                              └──────────────────────────────┘
```

## Project Structure

```
.
├── baselines/          # Encoder implementations (all follow BaseEncoder API)
│   ├── base_encoder.py #   Interface specification & BaseEncoder ABC
│   ├── flyhash/
│   ├── diehl_cook/
│   ├── deep_stdp/
│   ├── lc_snn/
│   ├── lm_snn/
│   ├── csdp/
│   ├── softhebb/
│   ├── krotov/
│   ├── biohash/
│   ├── wta_hash/
│   ├── som/
│   └── lsh/
├── configs/            # YAML configs per baseline
├── pipeline/           # Evaluation pipeline (datasets, metrics, clustering, retrieval)
├── scripts/            # Entry points & utilities
│   ├── run_baseline.py
│   ├── run_benchmark.py
│   └── run_all_baselines.sh
├── outputs/            # Results, codes, logs (git-ignored)
├── data/               # Datasets (git-ignored)
├── run.py              # Main CLI entry point
└── requirements.txt
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: bindsnet` | `pip install git+https://github.com/BindsNET/bindsnet.git` |
| `torch._six` import error | You installed bindsnet from PyPI (0.2.7). Reinstall from GitHub (see above). |
| `numpy >= 2.0` incompatibility | Pin: `pip install "numpy>=1.21,<2"` |
| CUDA out of memory (Diehl & Cook) | Set `device: cpu` in `configs/diehl_cook.yaml`, or reduce `n_neurons` |
| SIFT1M not found | Run `bash scripts/download_sift1m.sh` |
