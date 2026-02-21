# Neuroscience-baseline

Unified evaluation pipeline for biologically-inspired unsupervised feature learning baselines (SNN / Hebbian / hashing).

## Baselines

| Baseline | Method | Training |
|---|---|---|
| **FlyHash** | Random projection + WTA | Instant |
| **Diehl & Cook** | STDP + lateral inhibition (BindsNET) | ~6 h (60K, GPU) |
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

## Project Structure

```
.
├── baselines/          # Encoder implementations
│   ├── base_encoder.py
│   ├── flyhash/
│   ├── diehl_cook/
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
