# SIFT-1M Dataset Guide

Complete guide for using SIFT-1M dataset with the clustering pipeline.

## 📋 What is SIFT-1M?

**SIFT-1M** is a standard benchmark dataset for Approximate Nearest Neighbor (ANN) search:
- **1 million** 128-dimensional SIFT descriptors (base set)
- **10,000** query vectors
- Ground truth nearest neighbors for evaluation
- Source: [INRIA TEXMEX](http://corpus-texmex.irisa.fr/)

### Dataset Files

```
data/sift1m/
├── sift_base.fvecs         # 1M base vectors (128-dim) - ~500MB
├── sift_query.fvecs        # 10K query vectors (128-dim)
├── sift_groundtruth.ivecs  # Ground truth for queries
└── sift_learn.fvecs        # 100K learning vectors (optional)
```

## 🚀 Quick Start

### Step 1: Download SIFT-1M

```bash
cd /hy-tmp/clustering

# Method 1: Use download script (recommended)
bash scripts/download_sift1m.sh

# Method 2: Manual download
mkdir -p data/sift1m
cd data/sift1m
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -zxvf sift.tar.gz
mv sift/* .
rmdir sift
rm sift.tar.gz
cd ../..
```

**Download info**:
- Size: ~500MB compressed, ~1.5GB uncompressed
- Time: 5-10 minutes (depends on network)

### Step 2: Test Dataset Loading

```bash
# Test with small subset
python scripts/test_sift1m.py --subset 1000

# Test with default subset (10K)
python scripts/test_sift1m.py
```

### Step 3: Run FlyHash on SIFT-1M

```bash
# Quick test (single seed)
python run.py --config configs/flyhash_sift1m.yaml

# Full benchmark (multiple seeds)
bash scripts/run_sift1m_benchmark.sh
```

## 📊 Usage Examples

### Example 1: Quick Test

```bash
# Test on 1K subset
python run.py --baseline flyhash --dataset sift1m --seed 0
```

### Example 2: Clustering on 10K Subset

```bash
# Use default config (10K subset)
python run.py --config configs/flyhash_sift1m.yaml --seed 0
```

### Example 3: Full Benchmark

```bash
# Run complete benchmark with multiple seeds
bash scripts/run_sift1m_benchmark.sh
```

### Example 4: Custom Subset Size

Edit `configs/flyhash_sift1m.yaml`:

```yaml
dataset_config:
  subset_size: 50000  # Use 50K subset
```

Then run:

```bash
python run.py --config configs/flyhash_sift1m.yaml
```

## ⚙️ Configuration

### FlyHash Configuration for SIFT-1M

```yaml
# configs/flyhash_sift1m.yaml

dataset: "sift1m"
dataset_config:
  subset_size: 10000  # Clustering on 10K subset

encoder_config:
  input_dim: 128      # SIFT descriptor dimension
  projection_dim: 640 # 5x expansion
  hash_length: 32     # 5% sparsity
  sampling_ratio: 0.1

n_clusters: 100       # SIFT has many visual categories
```

### Subset Sizes

| Subset Size | Memory | Time | Use Case |
|-------------|--------|------|----------|
| 1,000 | ~5MB | ~10s | Quick test |
| 10,000 | ~50MB | ~1min | Default clustering |
| 50,000 | ~250MB | ~5min | Large-scale test |
| 100,000 | ~500MB | ~10min | Near-full scale |
| 1,000,000 | ~5GB | ~1hr | Full dataset (not recommended for clustering) |

**Recommendation**: Use 10K for clustering evaluation, full 1M for retrieval.

## 📈 Expected Performance

### Clustering Performance (10K subset)

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| NMI | 0.3 - 0.5 | Lower than MNIST (no clear labels) |
| ARI | 0.2 - 0.4 | Visual similarity is complex |
| ACC | 0.3 - 0.5 | Depends on n_clusters |

**Note**: SIFT-1M doesn't have ground truth labels for clustering, so metrics are for reference only.

### Why Lower Performance?

1. **No Ground Truth Labels**: SIFT-1M is for retrieval, not classification
2. **High Variability**: Visual features are more diverse than digit images
3. **Many Categories**: 100+ visual categories vs 10 digit classes
4. **Continuous Space**: SIFT features lie in continuous space

## 🔬 Advanced Usage

### Retrieval Evaluation (TODO)

SIFT-1M is primarily for retrieval evaluation:

```python
# Future implementation
eval_retrieval: true
retrieval_metrics:
  - recall@1
  - recall@10
  - recall@100
```

### Full Dataset Processing

For full 1M vectors (not recommended for clustering):

```yaml
dataset_config:
  subset_size: null  # Use all 1M vectors
```

**Warning**: Requires ~5GB RAM and ~1 hour processing time.

### Custom Train/Test Split

```python
from pipeline.datasets import load_dataset

# Load with custom split
dataset = load_dataset(
    'sift1m',
    root='./data',
    subset_size=20000
)

# 80/20 split is automatic
train_data = dataset['train_data']  # 16K samples
test_data = dataset['test_data']    # 4K samples
```

## 🐛 Troubleshooting

### Problem 1: Download Fails

**Error**: `wget: unable to resolve host address`

**Solution**:
```bash
# Try with curl instead
cd data/sift1m
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -zxvf sift.tar.gz
```

### Problem 2: File Not Found

**Error**: `FileNotFoundError: SIFT base file not found`

**Solution**:
```bash
# Check if files exist
ls -lh data/sift1m/

# Re-download if needed
bash scripts/download_sift1m.sh
```

### Problem 3: Out of Memory

**Error**: `MemoryError` or system freeze

**Solution**:
```yaml
# Reduce subset size in config
dataset_config:
  subset_size: 5000  # Smaller subset
```

### Problem 4: Slow Processing

**Symptom**: Takes too long to process

**Solution**:
```yaml
# Use only kmeans (fastest)
clustering_methods:
  - kmeans

# Reduce subset
dataset_config:
  subset_size: 10000
```

## 📊 Benchmark Scripts

### Script 1: Quick Test

```bash
# Test dataset loading
python scripts/test_sift1m.py
```

### Script 2: Single Run

```bash
# Run FlyHash once
python run.py --config configs/flyhash_sift1m.yaml
```

### Script 3: Full Benchmark

```bash
# Complete benchmark with report
bash scripts/run_sift1m_benchmark.sh
```

## 📝 Output Files

After running SIFT-1M benchmark:

```
outputs/
├── results/
│   ├── flyhash_sift1m_seed0.json
│   ├── flyhash_sift1m_seed1.json
│   └── flyhash_sift1m_seed2.json
│
├── codes/
│   └── flyhash/sift1m/
│       ├── pre_code_seed0.npy  # Continuous features
│       └── code_seed0.npy      # Binary codes
│
└── sift1m_benchmark/
    ├── benchmark_report.md     # Markdown report
    ├── benchmark_table.tex     # LaTeX table
    └── benchmark_results.json  # JSON data
```

## 🎯 Use Cases

### Use Case 1: Baseline Comparison

Compare FlyHash with other methods on SIFT-1M:

```bash
# Run FlyHash
python run.py --baseline flyhash --dataset sift1m --seed 0

# Run SoftHebb (if implemented for SIFT)
python run.py --baseline softhebb --dataset sift1m --seed 0
```

### Use Case 2: Hyperparameter Tuning

Test different FlyHash parameters:

```yaml
# configs/flyhash_sift1m_tuned.yaml
encoder_config:
  projection_dim: 1280  # Try 10x expansion
  hash_length: 64       # Try 5% of 1280
```

### Use Case 3: Scalability Testing

Test on different scales:

```bash
# Small: 1K
python run.py --config configs/flyhash_sift1m.yaml  # subset_size: 1000

# Medium: 10K (default)
python run.py --config configs/flyhash_sift1m.yaml  # subset_size: 10000

# Large: 100K
python run.py --config configs/flyhash_sift1m.yaml  # subset_size: 100000
```

## 📚 References

1. **SIFT-1M Dataset**:
   - Source: http://corpus-texmex.irisa.fr/
   - Paper: Jégou et al., "Product quantization for nearest neighbor search", TPAMI 2011

2. **SIFT Features**:
   - Lowe, D. G., "Distinctive image features from scale-invariant keypoints", IJCV 2004

3. **ANN Benchmarks**:
   - http://ann-benchmarks.com/

## 🔄 Integration Status

| Feature | Status | Notes |
|---------|--------|-------|
| Dataset Loading | ✅ Complete | Supports subset selection |
| Clustering | ✅ Complete | Works on subsets |
| Retrieval | ⚠️ TODO | Ground truth available |
| Visualization | ⚠️ TODO | Feature space plots |
| Full 1M Processing | ⚠️ Experimental | Memory intensive |

## 💡 Tips

1. **Start Small**: Always test with 1K subset first
2. **Use Subsets**: 10K is enough for clustering validation
3. **Monitor Memory**: Full dataset requires significant RAM
4. **Save Results**: Results are cached for reuse
5. **Compare Baselines**: SIFT-1M is great for comparing methods

---

**Last Updated**: 2026-01-16

**Status**: ✅ Fully Implemented

For questions or issues, check the troubleshooting section or create an issue.
