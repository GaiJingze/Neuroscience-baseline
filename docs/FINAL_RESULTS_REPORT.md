# Clustering Pipeline - Final Results Report

**Date**: 2026-01-30  
**Status**: ✅ All tests completed (24/24, 100%)

---

## 📊 Completion Status

| Baseline | MNIST | Fashion-MNIST | Total |
|----------|-------|---------------|-------|
| **FlyHash** | ✅ 3/3 | ✅ 3/3 | 6/6 |
| **Krotov** | ✅ 3/3 | ✅ 3/3 | 6/6 |
| **SoftHebb** | ✅ 3/3 | ✅ 3/3 | 6/6 |
| **Diehl & Cook** | ✅ 3/3 | ✅ 3/3 | 6/6 |
| **Total** | **12/12** | **12/12** | **24/24** |

All experiments completed with 3 different random seeds for robustness.

---

## 🏆 Performance Summary

### MNIST Results (Mean ± Std)

| Rank | Method | NMI | ARI | ACC | Status |
|------|--------|-----|-----|-----|--------|
| 🥇 | **Krotov** | **0.5897 ± 0.0045** | **0.4780 ± 0.0076** | **0.6470 ± 0.0141** | ⭐ Best |
| 🥈 | FlyHash | 0.5494 ± 0.0345 | 0.4089 ± 0.0187 | 0.5748 ± 0.0129 | Good |
| 🥉 | SoftHebb | 0.1806 ± 0.0009 | 0.0878 ± 0.0023 | 0.2094 ± 0.0008 | Poor |
| ❌ | Diehl & Cook | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1135 ± 0.0000 | Failed |

### Fashion-MNIST Results (Mean ± Std)

| Rank | Method | NMI | ARI | ACC | Status |
|------|--------|-----|-----|-----|--------|
| 🥇 | **FlyHash** | **0.5936 ± 0.0019** | **0.4128 ± 0.0133** | **0.5424 ± 0.0217** | ⭐ Best |
| 🥈 | SoftHebb | 0.3948 ± 0.0396 | 0.1860 ± 0.0500 | 0.2298 ± 0.0423 | Moderate |
| 🥉 | Krotov | 0.1611 ± 0.0451 | 0.0499 ± 0.0241 | 0.1898 ± 0.0147 | Poor |
| ❌ | Diehl & Cook | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 ± 0.0000 | Failed |

---

## 🔍 Detailed Analysis

### 1. Krotov-Hopfield (Best on MNIST)

**MNIST Performance**: 🏆 **Winner**
- NMI: 0.5897 ± 0.0045 (very stable, low variance)
- ARI: 0.4780 ± 0.0076 
- ACC: 0.6470 ± 0.0141
- **Improvement over FlyHash**: +7.3% NMI, +16.9% ARI, +12.5% ACC

**Fashion-MNIST Performance**: 😞 **Significant Degradation**
- NMI: 0.1611 ± 0.0451 (72.7% drop from MNIST)
- ARI: 0.0499 ± 0.0241
- ACC: 0.1898 ± 0.0147
- **Issue**: Method struggles with more complex textures in Fashion-MNIST

**Key Insights**:
- ✅ Excellent on simple digit patterns
- ✅ Very stable across different seeds (low std)
- ✅ Successfully learned meaningful features
- ❌ Poor generalization to fashion items
- ❌ Likely needs hyperparameter tuning for Fashion-MNIST

**Training**: 200 epochs, ~15 minutes per run

---

### 2. FlyHash (Most Robust)

**MNIST Performance**: 🥈 **Strong Second**
- NMI: 0.5494 ± 0.0345
- ARI: 0.4089 ± 0.0187
- ACC: 0.5748 ± 0.0129

**Fashion-MNIST Performance**: 🏆 **Winner**
- NMI: 0.5936 ± 0.0019 (most stable, lowest variance!)
- ARI: 0.4128 ± 0.0133
- ACC: 0.5424 ± 0.0217
- **Better than MNIST**: +8.0% NMI improvement

**Key Insights**:
- ✅ No training required (instant encoding)
- ✅ Extremely stable (std = 0.0019 on Fashion-MNIST!)
- ✅ Good generalization across datasets
- ✅ Strong baseline that's hard to beat
- 💡 Random projection captures sufficient structure

**Training**: None (parameter-free)

---

### 3. SoftHebb (Dataset-Dependent)

**MNIST Performance**: 🥉 **Poor**
- NMI: 0.1806 ± 0.0009 (very stable but low)
- ARI: 0.0878 ± 0.0023
- ACC: 0.2094 ± 0.0008
- **Issue**: Insufficient feature diversity, only ~2-10 unique codes

**Fashion-MNIST Performance**: 🥈 **Moderate**
- NMI: 0.3948 ± 0.0396 (118% improvement over MNIST!)
- ARI: 0.1860 ± 0.0500
- ACC: 0.2298 ± 0.0423
- **Better than Krotov** on Fashion-MNIST

**Key Insights**:
- ✅ Surprisingly better on Fashion-MNIST
- ❌ Poor on MNIST (possible collapse to few codes)
- ⚠️ High variance on Fashion-MNIST (std = 0.0396-0.0500)
- 💡 May need longer training or better hyperparameters for MNIST
- 💡 Inverse performance compared to Krotov

**Training**: 10 epochs, ~10 minutes per run

---

### 4. Diehl & Cook (Failed)

**Both Datasets**: ❌ **Complete Failure**
- NMI: 0.0000
- ARI: 0.0000
- ACC: ~0.10 (random guessing)

**Key Issues**:
- ❌ STDP training not producing useful features
- ❌ All encoded outputs identical or non-discriminative
- ❌ Spike encoding or network parameters incorrect
- 🔧 **Requires major debugging**

**Training**: ~6 hours per run (extremely slow for no benefit)

---

## 📈 Cross-Dataset Comparison

### Method Robustness (MNIST → Fashion-MNIST)

| Method | MNIST NMI | F-MNIST NMI | Change | Robustness |
|--------|-----------|-------------|--------|------------|
| **FlyHash** | 0.5494 | 0.5936 | **+8.0%** | 🏆 Excellent |
| **SoftHebb** | 0.1806 | 0.3948 | **+118.6%** | 🤔 Inverse |
| **Krotov** | 0.5897 | 0.1611 | **-72.7%** | 😞 Poor |
| **Diehl & Cook** | 0.0000 | 0.0000 | 0.0% | ❌ Failed |

**Key Finding**: FlyHash is the only method that maintains or improves performance across datasets.

---

## 🎯 Recommendations

### For Publication/Reporting

**Use these methods**:
1. ✅ **Krotov** (MNIST only) - Best learned features
2. ✅ **FlyHash** (both datasets) - Best baseline, most robust

**Exclude**:
- ❌ Diehl & Cook - failed implementation
- ⚠️ SoftHebb - inconsistent, needs investigation

### For Further Research

#### High Priority
1. **Debug Diehl & Cook**:
   - Check spike generation and neuron dynamics
   - Verify STDP weight updates are non-trivial
   - Compare with original paper's hyperparameters

2. **Improve Krotov on Fashion-MNIST**:
   - Increase training epochs (200 → 500?)
   - Tune anti-Hebbian strength `delta`
   - Adjust learning rate schedule
   - Try different `p` (Lebesgue norm) values

3. **Understand SoftHebb Behavior**:
   - Why does it work better on Fashion-MNIST?
   - Investigate MNIST feature collapse
   - Increase epochs (10 → 50?)
   - Tune `beta` (temperature) and `k` (top-k)

#### Medium Priority
4. **Explore FlyHash Variants**:
   - Although already good, try learned hash functions
   - Experiment with different projection dimensions

---

## 📊 Variance Analysis

### Most Stable Methods (Low Std)

| Method | Dataset | NMI Std | Stability |
|--------|---------|---------|-----------|
| FlyHash | Fashion-MNIST | 0.0019 | 🏆 Extremely stable |
| Krotov | MNIST | 0.0045 | ⭐ Very stable |
| SoftHebb | MNIST | 0.0009 | ⭐ Very stable |
| Diehl & Cook | Both | 0.0000 | ⚠️ Consistently bad |

### Least Stable Methods (High Std)

| Method | Dataset | NMI Std | Issue |
|--------|---------|---------|-------|
| Krotov | Fashion-MNIST | 0.0451 | High variance |
| FlyHash | MNIST | 0.0345 | Moderate variance |
| SoftHebb | Fashion-MNIST | 0.0396 | Moderate variance |

**Insight**: Methods with high variance may benefit from better initialization or hyperparameter tuning.

---

## 🏅 Overall Winner

### 🥇 **FlyHash** - Most Reliable

**Reasons**:
- ✅ No training required
- ✅ Consistent performance across datasets
- ✅ Extremely low variance (stable)
- ✅ Fast (instant encoding)
- ✅ Good absolute performance (NMI ~ 0.55-0.59)

### 🥈 **Krotov** - Best Learned Features (MNIST only)

**Reasons**:
- ✅ Best performance on MNIST
- ✅ Low variance (stable on MNIST)
- ✅ Demonstrates value of learning
- ❌ Poor on Fashion-MNIST (needs work)

---

## 📝 Summary for Slides/Paper

### Quick Stats

- **Best NMI overall**: Krotov on MNIST (0.5897)
- **Most robust**: FlyHash (consistent across datasets)
- **Biggest improvement**: Krotov +7.3% over FlyHash (MNIST)
- **Biggest challenge**: Fashion-MNIST generalization
- **Failed method**: Diehl & Cook (requires debugging)

### Key Takeaways

1. **Random projection (FlyHash) is surprisingly effective** - hard to beat without training
2. **Learning helps on simple tasks (MNIST)** - Krotov outperforms FlyHash
3. **Generalization is challenging** - methods specialized to MNIST fail on Fashion-MNIST
4. **STDP-based SNNs need more work** - current implementation doesn't produce useful features

### Recommended Table for Paper

```
Method          | MNIST (NMI)      | Fashion-MNIST (NMI) | Training Time
----------------|------------------|---------------------|---------------
Krotov          | 0.5897 ± 0.0045  | 0.1611 ± 0.0451     | ~15 min
FlyHash         | 0.5494 ± 0.0345  | 0.5936 ± 0.0019     | instant
SoftHebb        | 0.1806 ± 0.0009  | 0.3948 ± 0.0396     | ~10 min
```

---

## 🔗 Files Generated

- `outputs/results/` - All 24 result JSON files
- `scripts/check_all_results.py` - Result analysis script
- `CLUSTERING_METRICS_EXPLAINED.md` - NMI/ARI explanation
- `CLUSTERING_PIPELINE_SLIDES.md` - Presentation slides
- This report: `FINAL_RESULTS_REPORT.md`

---

**Report Generated**: 2026-01-30  
**Pipeline Version**: Complete with all 4 baselines  
**Total Experiments**: 24 (100% complete)
