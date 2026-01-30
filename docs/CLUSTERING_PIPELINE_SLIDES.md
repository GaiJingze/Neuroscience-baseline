---
marp: true
theme: default
paginate: true
---

# Biological Learning for Unsupervised Clustering

**A Comparison of Bio-Inspired Hashing Methods**

---

## Pipeline Overview

**Unified evaluation framework for bio-inspired clustering algorithms**

### Key Features
- 🧬 Multiple biologically-inspired baselines
- 📊 Standardized evaluation on vision datasets
- ⚡ Efficient unsupervised feature learning
- 📈 Comprehensive clustering metrics (NMI, ARI, ACC)

### Workflow
```
Dataset → Encoder (Training) → Feature Extraction → Clustering → Evaluation
```

---

## Datasets

### MNIST
- **Size**: 60K training + 10K test samples
- **Description**: Handwritten digits (0-9), 28×28 grayscale images
- **Challenge**: Classic baseline for unsupervised learning

### Fashion-MNIST
- **Size**: 60K training + 10K test samples  
- **Description**: Fashion items (10 categories), 28×28 grayscale images
- **Challenge**: More complex patterns than MNIST

Both normalized to [0,1] range, flattened to 784-dimensional vectors.

---

## Baselines (I)

### FlyHash
- **Inspiration**: Drosophila olfactory system with random projection
- **Training**: None (parameter-free random hashing)
- **Performance**: NMI=0.55 (MNIST), fast and stable baseline

### Krotov-Hopfield ⭐
- **Inspiration**: Competing hidden units with anti-Hebbian learning
- **Training**: 200 epochs with k-WTA competition mechanism
- **Performance**: NMI=0.58 (MNIST), **best clustering results**

---

## Baselines (II)

### SoftHebb
- **Inspiration**: Soft winner-take-all with Hebbian plasticity
- **Training**: 10 epochs with probabilistic competition
- **Performance**: NMI=0.18-0.41, faster but less effective

### Diehl & Cook (STDP-SNN)
- **Inspiration**: Spiking neural networks with STDP learning
- **Training**: ~6 hours with Poisson spike encoding
- **Status**: Under debugging (current implementation has encoding issues)

---

## Results & Performance

### Clustering Performance (MNIST, K-means)

| Method | NMI | ARI | ACC | Training Time |
|--------|-----|-----|-----|---------------|
| **Krotov** | **0.5837** | **0.4683** | **0.6313** | ~15 min |
| FlyHash | 0.5494 | 0.4089 | 0.5748 | - |
| SoftHebb | 0.1806 | 0.0878 | 0.2094 | ~10 min |

### Key Findings
✅ Learned features (Krotov) outperform random projection (+6.2%)  
✅ Biologically-inspired learning achieves competitive results  
✅ Efficient training (<20 min) enables practical applications

---
