# Dataset Suitability Analysis for Clustering Pipeline

**Date**: 2026-01-30  
**Question**: Are SIFT-1M and GloVE suitable for our clustering tests?

---

## 📊 Current Status

### ✅ Currently Used Datasets

| Dataset | Type | Size | Labels | Purpose | Status |
|---------|------|------|--------|---------|--------|
| **MNIST** | Images (28×28) | 60K train, 10K test | 10 digits | Clustering | ✅ Working well |
| **Fashion-MNIST** | Images (28×28) | 60K train, 10K test | 10 classes | Clustering | ✅ Working well |
Alternative: CIFAR-10


### 📋 Proposed Datasets

| Dataset | Type | Dimensionality | Size | Labels | Original Purpose |
|---------|------|----------------|------|--------|------------------|
| **SIFT-1M** | Feature vectors | 128-dim | 1M base, 10K query | No semantic labels | ANN search / Retrieval |
| **GloVE** | Word embeddings | 50/100/200/300-dim | 400K-2.2M words | No semantic labels | NLP similarity |

---

## 🎯 Analysis: SIFT-1M

### ✅ Advantages

1. **Standard Benchmark**
   - Widely used in hashing/retrieval research
   - Easy to compare with other methods
   - Well-documented performance baselines

2. **Large Scale**
   - 1 million samples (good for scalability testing)
   - Tests performance on realistic data volumes

3. **Ground Truth Neighbors**
   - Has pre-computed nearest neighbors
   - Perfect for retrieval evaluation (Recall@K, mAP)

4. **FlyHash Originally Designed for This**
   - FlyHash paper (Dasgupta et al., 2017) used SIFT for hashing tasks
   - Natural fit for our FlyHash baseline

### ❌ Disadvantages

1. **Not Raw Sensory Input** ⚠️
   - SIFT features are **already extracted** from images
   - Our SNN methods (Diehl & Cook, SoftHebb, Krotov) are designed to learn from raw inputs
   - **Missing the "learning" step** that makes SNNs interesting

2. **No Semantic Labels**
   - SIFT-1M has no class labels (digit/object categories)
   - **Cannot compute NMI, ARI, or ACC** (require ground truth labels)
   - Only suitable for **retrieval metrics** (Recall@K, mAP)

3. **Mismatch with SNN Philosophy**
   - SNNs are inspired by biology → process raw sensory data
   - SIFT is hand-crafted feature engineering
   - Using pre-extracted features defeats the purpose of learning features

4. **Limited Interpretability**
   - Hard to visualize what clusters mean
   - No clear "correct" clustering (no semantic categories)

### 🎯 Verdict: SIFT-1M

**Suitable for**: ✅ **Task B - Locality-Sensitive Hashing / Retrieval**  
**NOT suitable for**: ❌ **Task A - Unsupervised Feature Learning / Clustering**

#### Recommended Use

```
✅ Use SIFT-1M for:
  - Testing FlyHash hashing performance
  - Approximate Nearest Neighbor (ANN) evaluation
  - Retrieval metrics: Recall@K, mAP, Precision@K

❌ Do NOT use SIFT-1M for:
  - Clustering evaluation (NMI, ARI, ACC)
  - Training SNNs (Diehl & Cook, SoftHebb, Krotov)
  - Comparing feature learning quality
```

---

## 🎯 Analysis: GloVE

### ✅ Advantages

1. **Large Vocabulary**
   - 400K-2.2M word embeddings
   - Good for scalability testing

2. **Semantic Structure**
   - Words have semantic relationships
   - Could form natural clusters (synonyms, topics)

3. **Standard NLP Benchmark**
   - Well-known dataset
   - Easy to compare with other methods

### ❌ Disadvantages

1. **Even Further from Raw Input** ⚠️⚠️
   - GloVE embeddings are learned by another model
   - **Two levels removed** from raw data (text → tokens → embeddings)
   - Completely defeats the purpose of SNN feature learning

2. **No Ground Truth Labels**
   - No semantic class labels for words
   - **Cannot compute NMI, ARI, ACC**
   - Hard to define "correct" clustering

3. **Domain Mismatch**
   - Our SNNs are designed for **vision tasks** (images)
   - GloVE is from **NLP domain** (text)
   - Architecture/hyperparameters not optimized for text

4. **Unclear Clustering Goal**
   - What is the "correct" clustering of words?
   - By topic? By part-of-speech? By semantic field?
   - No clear ground truth

5. **Not Biologically Plausible**
   - SNNs inspired by vision/audition
   - Processing word embeddings ≠ biological neural processing

### 🎯 Verdict: GloVE

**Suitable for**: ❌ **Neither Task A nor Task B**  
**Reason**: Too far from raw sensory input, wrong domain for vision SNNs

#### Recommendation

```
❌ Do NOT use GloVE for:
  - Clustering evaluation
  - Training vision-based SNNs
  - Comparison with image-based methods
  
🤔 Could consider IF:
  - You want to test method generalization
  - You have text-specific SNN baselines
  - You want to show limitations of vision SNNs on NLP
```

---

## 💡 Recommendations

### ✅ Keep Using for Task A (Clustering)

1. **MNIST** ⭐
   - Perfect for clustering evaluation
   - Clear semantic labels (10 digits)
   - Raw pixel input (good for SNNs)
   - Reasonably challenging

2. **Fashion-MNIST** ⭐
   - More challenging than MNIST
   - Clear semantic labels (10 clothing types)
   - Same format as MNIST (easy to compare)
   - Tests generalization

3. **CIFAR-10** (Recommended to add) 🆕
   - Color images (32×32×3)
   - 10 clear semantic classes
   - More challenging than Fashion-MNIST
   - Standard benchmark
   - Still has labels for NMI/ARI/ACC

### ✅ Use for Task B (Retrieval/Hashing)

1. **SIFT-1M** ⭐
   - Standard hashing benchmark
   - Good for FlyHash evaluation
   - Has ground truth neighbors
   - Use Recall@K, mAP metrics

### ❌ Do NOT Use

1. **GloVE**
   - Wrong domain (text vs. vision)
   - No ground truth labels
   - Too far from raw input

---

## 📋 Comparison Table

| Dataset | Raw Input? | Labels? | Size | SNN Suitable? | Clustering Metrics? | Retrieval Metrics? | Recommendation |
|---------|-----------|---------|------|---------------|--------------------|--------------------|----------------|
| **MNIST** | ✅ Yes (pixels) | ✅ Yes (10) | 70K | ✅ Excellent | ✅ NMI, ARI, ACC | ✅ Yes | ⭐ Keep |
| **Fashion-MNIST** | ✅ Yes (pixels) | ✅ Yes (10) | 70K | ✅ Excellent | ✅ NMI, ARI, ACC | ✅ Yes | ⭐ Keep |
| **CIFAR-10** | ✅ Yes (pixels) | ✅ Yes (10) | 60K | ✅ Excellent | ✅ NMI, ARI, ACC | ✅ Yes | 🆕 Add |
| **SIFT-1M** | ❌ No (features) | ❌ No | 1M | ⚠️ Limited | ❌ No | ✅ Recall@K, mAP | ⚡ Hashing only |
| **GloVE** | ❌ No (embeddings) | ❌ No | 400K-2.2M | ❌ Poor | ❌ No | ⚠️ Unclear | ❌ Skip |

---

## 🎯 Final Recommendations

### For Our Current Project

#### Task A: Unsupervised Feature Learning & Clustering

```
Current: MNIST ✅ + Fashion-MNIST ✅
Recommendation: Keep both, consider adding CIFAR-10

Reason:
- These have semantic labels → can compute NMI, ARI, ACC
- Raw pixel input → SNNs can learn features
- Clear clustering goal → meaningful evaluation
```

#### Task B: Locality-Sensitive Hashing / Retrieval

```
Current: None fully tested
Recommendation: Use SIFT-1M for FlyHash only

Reason:
- Standard hashing benchmark
- But ONLY test FlyHash (no training required)
- Don't train SNNs on pre-extracted features
```

### Why SIFT-1M and GloVE Are Problematic

#### Philosophical Mismatch

```
SNNs (Diehl & Cook, SoftHebb, Krotov):
  Raw Input → Learn Features → Clustering
  [This is what makes them interesting]

SIFT-1M / GloVE:
  Images → SIFT Extraction → [Our SNN here?]
           ^^^^^^^^^^^^^^^^
           This step is ALREADY feature learning!
  
  → Our SNNs become just clustering algorithms
  → Not testing their feature learning ability
  → Missing the point of biologically-inspired learning
```

#### Practical Issues

```
Problem 1: No Labels
- SIFT-1M and GloVE have no semantic labels
- Cannot compute NMI, ARI, ACC
- Cannot evaluate clustering quality objectively

Problem 2: Pre-extracted Features
- SNNs designed to process raw sensory input
- Using pre-extracted features is like:
  "Testing a chef's cooking skills by giving them frozen dinners"
  
Problem 3: Unfair Comparison
- FlyHash: Works on any features (including SIFT)
- SNNs: Designed for raw input
- Comparing them on SIFT is unfair to SNNs
```

---

## 📊 Suggested Dataset Strategy

### Short Term (Current Phase)

**Focus on Task A (Clustering):**
```bash
1. MNIST               ✅ Already working
2. Fashion-MNIST       ✅ Already working
3. CIFAR-10            🆕 Consider adding
```

**Metrics**: NMI, ARI, ACC, Silhouette  
**Methods**: All 4 baselines (FlyHash, Krotov, SoftHebb, Diehl & Cook)

### Medium Term (If extending)

**Add Task B (Retrieval) - FlyHash Only:**
```bash
1. SIFT-1M            Use for FlyHash hashing evaluation
```

**Metrics**: Recall@K, mAP, Precision@K  
**Methods**: FlyHash only (others not suitable)

### Long Term (Future work)

**If you want more challenging clustering:**
```bash
1. CIFAR-100          100 classes, more complex
2. ImageNet subset    Real-world complexity
3. STL-10             96×96 images, semi-supervised
```

All have:
- ✅ Raw pixel input
- ✅ Semantic labels
- ✅ Suitable for SNNs

---

## 🔬 Special Case: SIFT-1M for FlyHash

### When SIFT-1M Makes Sense

If you want to show **FlyHash's versatility**:

```python
# FlyHash doesn't need training
# Can work on any feature representation

# Test 1: Learn from raw images (MNIST)
flyhash_mnist = FlyHash(input_dim=784)
codes_mnist = flyhash_mnist.encode(mnist_images)  # Learn features

# Test 2: Hash pre-extracted features (SIFT-1M)
flyhash_sift = FlyHash(input_dim=128)
codes_sift = flyhash_sift.encode(sift_features)  # Just hash

# This demonstrates FlyHash's flexibility
# But DON'T test SNNs this way (unfair comparison)
```

### Separate Evaluation Tracks

```
Track 1: Feature Learning + Clustering (Main focus)
├─ Datasets: MNIST, Fashion-MNIST, CIFAR-10
├─ Metrics: NMI, ARI, ACC
└─ Methods: All 4 baselines

Track 2: Hashing / Retrieval (Secondary, FlyHash only)
├─ Datasets: SIFT-1M
├─ Metrics: Recall@K, mAP
└─ Methods: FlyHash only

Don't mix these two tracks!
```

---

## 📝 Conclusion

### Direct Answer

**SIFT-1M**: ⚠️ **Partially suitable**
- ✅ Good for FlyHash hashing/retrieval evaluation
- ❌ Not suitable for SNN feature learning
- ❌ Cannot evaluate clustering (no labels)

**GloVE**: ❌ **Not suitable**
- ❌ Wrong domain (text vs. vision)
- ❌ Too far from raw sensory input
- ❌ No clear clustering ground truth

### Recommendation

```
🎯 Current Strategy (Task A): Keep MNIST + Fashion-MNIST
   → These are perfect for our SNN clustering evaluation
   
🎯 Optional Addition: CIFAR-10
   → More challenging, still has labels, still raw pixels
   
⚡ Optional Track B: SIFT-1M for FlyHash only
   → Shows FlyHash's hashing capability
   → Don't test other SNNs on this
   
❌ Skip: GloVE
   → Not suitable for vision-based SNNs
```

---

**Summary**: Stick with MNIST and Fashion-MNIST for clustering. They're perfect for our biologically-inspired SNNs. SIFT-1M and GloVE would distract from demonstrating what makes our methods special: learning features from raw sensory input.
