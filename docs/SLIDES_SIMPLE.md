# Biological Learning for Clustering - Slides

---

## Slide 1: Title

```
═══════════════════════════════════════════════════════════════

        Biological Learning for Unsupervised Clustering
        
        A Comparison of Bio-Inspired Hashing Methods
        
═══════════════════════════════════════════════════════════════
```

---

## Slide 2: Pipeline Overview

```
Pipeline Overview
─────────────────────────────────────────────────────────────

• Unified framework for bio-inspired clustering algorithms
• Standardized evaluation on vision datasets
• Efficient unsupervised feature learning

Workflow:
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌────────────┐
│ Dataset │ ─→ │ Encoder │ ─→ │ Features │ ─→ │ Clustering │
└─────────┘    └─────────┘    └──────────┘    └────────────┘
                  (Train)        (Extract)        (Evaluate)

Metrics: NMI, ARI, Accuracy
```

---

## Slide 3: Datasets

```
Datasets
─────────────────────────────────────────────────────────────

MNIST
• 60K training + 10K test samples
• Handwritten digits (0-9), 28×28 grayscale
• Classic benchmark for unsupervised learning

Fashion-MNIST
• 60K training + 10K test samples
• Fashion items (10 categories), 28×28 grayscale  
• More complex visual patterns

Both: Normalized [0,1], flattened to 784D vectors
```

---

## Slide 4: Baselines

```
Biologically-Inspired Baselines
─────────────────────────────────────────────────────────────

1. FlyHash
   ○ Drosophila olfactory system (random projection)
   ○ No training required | NMI: 0.55 (MNIST)

2. Krotov-Hopfield ⭐
   ○ Competing units with anti-Hebbian learning
   ○ 200 epochs, ~15 min | NMI: 0.58 (MNIST) - BEST

3. SoftHebb
   ○ Soft WTA with Hebbian plasticity
   ○ 10 epochs, ~10 min | NMI: 0.18-0.41

4. Diehl & Cook (STDP-SNN)
   ○ Spiking neurons with STDP
   ○ Under debugging
```

---

## Slide 5: Results

```
Performance Comparison (MNIST)
─────────────────────────────────────────────────────────────

Method          NMI     ARI     ACC     Time
──────────────────────────────────────────────────────────
Krotov ⭐       0.5837  0.4683  0.6313  ~15 min
FlyHash         0.5494  0.4089  0.5748  instant
SoftHebb        0.1806  0.0878  0.2094  ~10 min


Key Findings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Learned features outperform random projection (+6.2%)
✓ Krotov achieves best performance with efficient training
✓ Bio-inspired learning is both effective and practical
```

---
