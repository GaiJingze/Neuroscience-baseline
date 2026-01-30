# Universal Clustering Methods Analysis

**Question**: What clustering methods work across MNIST, SIFT-1M, and GloVE?

**Date**: 2026-01-30

---

## 🎯 The Challenge

### Three Very Different Datasets

| Dataset | Type | Dimensionality | Labels? | Raw Input? |
|---------|------|----------------|---------|------------|
| **MNIST** | Pixels | 784-dim | ✅ Yes (10) | ✅ Yes |
| **SIFT-1M** | Visual features | 128-dim | ❌ No | ❌ No (pre-extracted) |
| **GloVE** | Text embeddings | 50-300-dim | ❌ No | ❌ No (pre-trained) |

### Key Requirements for Universal Method

```
Must be:
✅ Feature-agnostic (works on any vector representation)
✅ Dimension-flexible (handles different input sizes)
✅ Unsupervised (doesn't need labels for training)
✅ Scalable (handles thousands to millions of samples)

Should NOT require:
❌ Raw sensory input
❌ Domain-specific architecture
❌ Learning from scratch on each dataset
```

---

## ✅ Methods That Work Universally

### 1. **FlyHash** ⭐⭐⭐ (Best Universal Method)

#### Why It Works

```python
# FlyHash is completely feature-agnostic
# Only needs: input_dim, projection_dim, hash_length

# MNIST (784-dim)
flyhash_mnist = FlyHash(input_dim=784, projection_dim=2000, hash_length=100)
codes_mnist = flyhash_mnist.encode(mnist_pixels)

# SIFT-1M (128-dim)
flyhash_sift = FlyHash(input_dim=128, projection_dim=512, hash_length=64)
codes_sift = flyhash_sift.encode(sift_features)

# GloVE (300-dim)
flyhash_glove = FlyHash(input_dim=300, projection_dim=1000, hash_length=80)
codes_glove = flyhash_glove.encode(glove_embeddings)
```

#### Advantages

✅ **No training required** - instant encoding  
✅ **Pure random projection** - works on any features  
✅ **Biologically inspired** - fruit fly olfactory system  
✅ **Proven on SIFT** - original paper used SIFT descriptors  
✅ **Dimensionality reduction** - maps to fixed hash length  
✅ **Extremely fast** - just matrix multiplication + WTA  

#### Performance Expectations

| Dataset | Expected NMI | Expected Recall@100 | Notes |
|---------|--------------|---------------------|-------|
| MNIST | ~0.55 | ~0.90 | ✅ Already tested |
| SIFT-1M | N/A | ~0.85-0.95 | Good for retrieval |
| GloVE | ~0.30-0.40 | ~0.80-0.90 | Semantic clusters |

#### Configuration Tips

```yaml
# MNIST (high-dim pixels → clustering)
projection_dim: 2000  # 2-3x input_dim
hash_length: 100      # ~1/8 of input_dim
top_k_percent: 0.05   # 5% sparsity

# SIFT-1M (low-dim features → retrieval)
projection_dim: 512   # 4-5x input_dim
hash_length: 64       # ~1/2 of input_dim
top_k_percent: 0.05   # 5% sparsity

# GloVE (medium-dim embeddings → clustering)
projection_dim: 1000  # 3-4x input_dim
hash_length: 80       # ~1/4 of input_dim
top_k_percent: 0.05   # 5% sparsity
```

---

### 2. **Traditional Clustering Methods** ⭐⭐

#### K-means

```python
from sklearn.cluster import KMeans

# Works on ANY feature vectors
kmeans = KMeans(n_clusters=10)

# MNIST
kmeans.fit(mnist_features)  # (70000, 784)

# SIFT-1M
kmeans.fit(sift_features)   # (1000000, 128)

# GloVE
kmeans.fit(glove_embeddings) # (400000, 300)
```

**Advantages**:
- ✅ Simple and fast
- ✅ Well-understood
- ✅ Works on any dimensional data
- ✅ Available in sklearn

**Disadvantages**:
- ❌ Needs to specify K in advance
- ❌ Sensitive to initialization
- ❌ Assumes spherical clusters
- ❌ Slow on large datasets (SIFT-1M)

#### Spectral Clustering

```python
from sklearn.cluster import SpectralClustering

# Good for non-convex clusters
spectral = SpectralClustering(n_clusters=10, affinity='nearest_neighbors')
```

**Advantages**:
- ✅ Handles non-convex clusters
- ✅ Works on any features

**Disadvantages**:
- ❌ Very slow on large data (O(n³))
- ❌ Memory intensive
- ❌ Not practical for SIFT-1M or GloVE

#### DBSCAN

```python
from sklearn.cluster import DBSCAN

# Density-based, doesn't need K
dbscan = DBSCAN(eps=0.5, min_samples=5)
```

**Advantages**:
- ✅ No need to specify K
- ✅ Finds arbitrary-shaped clusters
- ✅ Robust to outliers

**Disadvantages**:
- ❌ Hard to tune eps and min_samples
- ❌ Different datasets need different parameters
- ❌ Slow on high-dimensional data

---

### 3. **Locality-Sensitive Hashing (LSH)** ⭐⭐

#### Random Projection LSH

```python
# Similar concept to FlyHash
# Random hyperplanes divide space

class SimpleLSH:
    def __init__(self, input_dim, n_hash_bits):
        self.random_planes = np.random.randn(input_dim, n_hash_bits)
    
    def hash(self, X):
        # X @ random_planes > 0
        return (X @ self.random_planes > 0).astype(int)
```

**Advantages**:
- ✅ Feature-agnostic
- ✅ Fast hashing
- ✅ Good for retrieval

**Disadvantages**:
- ❌ Not optimized for clustering quality
- ❌ FlyHash is better (adds WTA)

---

### 4. **Autoencoder-Based Methods** ⭐

#### Deep Clustering (DEC)

```python
# Train an autoencoder, then cluster in latent space
# Can work on any data after training

encoder = AutoEncoder(input_dim=784, latent_dim=10)
encoder.fit(mnist_data)
latent_codes = encoder.encode(mnist_data)
kmeans.fit(latent_codes)
```

**Advantages**:
- ✅ Learns good representations
- ✅ Can handle any input dimension
- ✅ State-of-the-art clustering performance

**Disadvantages**:
- ❌ Requires training on each dataset
- ❌ Computationally expensive
- ❌ Needs GPU for large datasets
- ❌ Not truly "universal" (dataset-specific training)

---

## ❌ Methods That DON'T Work Universally

### Your Current SNN Methods

#### ❌ Diehl & Cook (STDP-SNN)

```
Problem: Architecture is vision-specific
- Poisson encoding expects pixel intensities [0, 255]
- LIF neuron parameters tuned for vision
- STDP learning designed for spatial patterns

MNIST:     ✅ Works (pixels)
SIFT-1M:   ⚠️ Could work but loses SNN advantage
GloVE:     ❌ Wrong domain (not designed for text embeddings)
```

#### ❌ SoftHebb (Hebbian Learning)

```
Problem: Designed to learn from raw inputs
- Hebbian plasticity learns input correlations
- Multi-layer architecture for hierarchical features
- Optimized for vision tasks

MNIST:     ✅ Works (learns from pixels)
SIFT-1M:   ⚠️ Could work but defeats the purpose
GloVE:     ❌ Domain mismatch
```

#### ❌ Krotov-Hopfield

```
Problem: Learns input-specific weight patterns
- Competitive learning assumes raw sensory input
- Anti-Hebbian learning for selectivity
- Optimized for visual features

MNIST:     ✅ Works well (learns from pixels)
SIFT-1M:   ⚠️ Could work but loses learning value
GloVE:     ❌ Not optimized for text embeddings
```

---

## 📊 Comprehensive Comparison

### Universality Score

| Method | MNIST | SIFT-1M | GloVE | Training? | Speed | Universality |
|--------|-------|---------|-------|-----------|-------|--------------|
| **FlyHash** | ✅ 0.55 | ✅ Good | ✅ OK | ❌ No | ⚡ Instant | ⭐⭐⭐ Perfect |
| **K-means** | ✅ 0.50 | ✅ Good | ✅ OK | ❌ No | ⚡ Fast | ⭐⭐⭐ Perfect |
| **Random LSH** | ✅ OK | ✅ Good | ✅ OK | ❌ No | ⚡ Instant | ⭐⭐⭐ Perfect |
| **DBSCAN** | ✅ Varies | ✅ Varies | ✅ Varies | ❌ No | 🐌 Slow | ⭐⭐ Tuning needed |
| **Spectral** | ✅ 0.55 | ❌ Too slow | ❌ Too slow | ❌ No | 🐌 Very slow | ⭐ Limited scale |
| **Autoencoder** | ✅ 0.70+ | ✅ 0.80+ | ✅ 0.60+ | ✅ Yes | 🐌 Slow train | ⭐ Per-dataset |
| **Diehl & Cook** | ✅ 0.00* | ⚠️ Maybe | ❌ No | ✅ Yes | 🐌 Very slow | ❌ Vision only |
| **SoftHebb** | ✅ 0.18 | ⚠️ Maybe | ❌ No | ✅ Yes | ⚡ Fast | ❌ Vision only |
| **Krotov** | ✅ 0.59 | ⚠️ Maybe | ❌ No | ✅ Yes | 🐌 Slow | ❌ Vision only |

*Current implementation is broken, theoretically should be ~0.65

---

## 🎯 Recommendations

### For Your Research Project

#### If Goal is "Show Biological Learning Works"

```
❌ Don't use SIFT-1M or GloVE

Reason: 
- These defeat the purpose of your SNN methods
- SNNs are about learning from raw input
- Using pre-extracted features hides their value

Recommendation:
- Stick with MNIST + Fashion-MNIST
- Add CIFAR-10 if you want more challenge
- Compare: Krotov vs FlyHash vs traditional methods
```

#### If Goal is "Compare Universal Clustering"

```
✅ Use all three datasets

Test these universal methods:
1. FlyHash (your main contribution)
2. K-means (classical baseline)
3. Random LSH (hashing baseline)
4. Spectral (if computationally feasible)

Don't test:
- SNN methods on SIFT-1M/GloVE (unfair)
- Methods requiring per-dataset training (not universal)
```

### Specific Recommendations

#### **Recommendation 1: FlyHash as Universal Method** ⭐

```python
# Create a unified benchmark
datasets = ['mnist', 'fashion_mnist', 'sift1m', 'glove']

for dataset in datasets:
    # Load data
    X, y = load_data(dataset)
    
    # FlyHash (same algorithm, different params)
    flyhash = FlyHash(
        input_dim=X.shape[1],
        projection_dim=X.shape[1] * 3,
        hash_length=max(64, X.shape[1] // 8)
    )
    
    codes = flyhash.encode(X)
    
    # Evaluate
    if y is not None:  # MNIST, Fashion-MNIST
        nmi, ari, acc = evaluate_clustering(codes, y)
    else:  # SIFT-1M, GloVE
        recall_at_k = evaluate_retrieval(codes, ground_truth)
```

**This shows**: FlyHash's versatility across domains

#### **Recommendation 2: Two-Track Evaluation**

```
Track 1: Feature Learning (MNIST + Fashion-MNIST)
├─ Methods: Krotov, SoftHebb, Diehl & Cook, FlyHash
├─ Purpose: Show biological learning works
└─ Metrics: NMI, ARI, ACC

Track 2: Universal Hashing (MNIST + SIFT-1M + GloVE)
├─ Methods: FlyHash, LSH, K-means
├─ Purpose: Show FlyHash generalizes
└─ Metrics: Mixed (NMI for MNIST, Recall@K for others)

Keep these tracks SEPARATE in your paper!
```

#### **Recommendation 3: Position FlyHash Correctly**

```
FlyHash's Value:

Primary: 
  "Brain-inspired universal hashing"
  → Works across domains with zero training
  → Biologically plausible
  → Efficient

Secondary:
  "Competitive with learned methods on specific tasks"
  → Krotov is better on MNIST (but needs training)
  → FlyHash is more general-purpose
```

---

## 🔬 Experimental Design

### Option A: Focus on Biological Learning (Recommended)

```yaml
Datasets: 
  - MNIST
  - Fashion-MNIST
  - (Optional) CIFAR-10

Methods:
  - Krotov (best learned features)
  - FlyHash (best no-training baseline)
  - SoftHebb (alternative Hebbian)
  - Diehl & Cook (classical STDP)

Metrics:
  - NMI, ARI, ACC (clustering quality)
  - Training time
  - Spike sparsity

Story:
  "Biological learning rules can discover meaningful features
   from raw sensory input, competitive with random projection."
```

### Option B: Universal Hashing Benchmark

```yaml
Datasets:
  - MNIST (vision + labels)
  - SIFT-1M (vision + no labels)
  - GloVE (text + no labels)

Methods:
  - FlyHash (bio-inspired)
  - Random LSH (classical)
  - K-means (clustering)
  
Metrics:
  - NMI (where labels exist)
  - Recall@K (all datasets)
  - Hash computation time

Story:
  "FlyHash is a universal hashing method inspired by fruit fly
   olfaction, working across vision and text domains."
```

### Option C: Both Tracks (Most Complete)

```yaml
Part 1: Feature Learning from Raw Input
  → Use MNIST + Fashion-MNIST
  → Compare all SNN methods
  → Show Krotov > FlyHash on these

Part 2: Universal Hashing Capability
  → Add SIFT-1M (+ optionally GloVE)
  → Test only FlyHash (+ classical baselines)
  → Show generalization

Key Message:
  "Biological methods excel at learning from raw input,
   but random projection (FlyHash) is surprisingly general."
```

---

## 💡 Key Insights

### Why FlyHash is the Only Truly Universal Method

```
FlyHash = Random Projection + WTA

Properties:
1. Feature-agnostic: Doesn't care about input type
2. No training: Works immediately
3. Biologically plausible: Inspired by real neural circuits
4. Scalable: O(n*d*m) complexity
5. Proven: Original paper tested on SIFT

This is why fruit flies can generalize across odors!
```

### Why SNNs are NOT Universal (and that's OK!)

```
SNNs (Diehl & Cook, Krotov, SoftHebb):
- Designed to LEARN features from raw sensory input
- Architecture tuned for specific modality (vision)
- Value is in learning, not just clustering

Using them on SIFT/GloVE:
- Skips the learning (features already extracted)
- Like using a race car to push a shopping cart
- Technically works, but misses the point
```

---

## 📝 Conclusion

### Direct Answer

**Yes, there are universal methods that work across all three datasets:**

1. **FlyHash** ⭐⭐⭐ - Best universal method
   - Random projection + WTA
   - Zero training
   - Works on any features
   - You already have this implemented!

2. **K-means** ⭐⭐⭐ - Classical baseline
   - Simple and effective
   - Feature-agnostic
   - Good for comparison

3. **Random LSH** ⭐⭐ - Hashing baseline
   - Similar to FlyHash but simpler
   - Good for retrieval

### But Should You Use All Three Datasets?

**Depends on your research story:**

#### Story 1: "Biological Learning Works" → Use MNIST + Fashion-MNIST
- Focus on learning from raw input
- Compare SNN methods
- Show Krotov's success

#### Story 2: "Universal Bio-Inspired Hashing" → Use all datasets
- Focus on FlyHash's generalization
- Compare with classical methods
- Show cross-domain capability

#### Story 3: "Both!" → Two-track evaluation
- Track 1: Feature learning (MNIST/Fashion-MNIST, all SNNs)
- Track 2: Universal hashing (all datasets, FlyHash only)
- Show different strengths

### My Recommendation

Given your current results:
```
✅ Keep focusing on MNIST + Fashion-MNIST
✅ Your SNN methods (especially Krotov) show real value here
✅ This demonstrates biological learning principles

🔧 Optionally add SIFT-1M for FlyHash only
✅ Shows FlyHash's versatility
✅ Connects to original FlyHash paper
✅ Different from SNN track

❌ Skip GloVE
❌ Wrong domain for vision SNNs
❌ Adds complexity without clear value
```

---

**Summary**: **FlyHash** is your universal method. It works across all datasets because it doesn't learn—it just projects randomly (like a fruit fly). Your other SNN methods are specialists that excel at learning from raw sensory input. Present them accordingly!
