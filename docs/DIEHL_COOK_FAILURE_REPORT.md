# Diehl & Cook Encoder Failure Analysis

**Date**: 2026-01-30  
**Status**: ❌ Complete Failure (NMI=0, ARI=0)

---

## 🔴 Problem Summary

The Diehl & Cook STDP-based encoder is producing **identical codes for all input samples**, resulting in:
- **NMI = 0.0** (No mutual information)
- **ARI = 0.0** (No agreement with true labels)
- **ACC = 0.1135** (Random guessing ≈ 1/10)

---

## 🔬 Diagnostic Findings

### Critical Issue: All Codes Are Identical

```
Unique codes: 1 / 10,000 samples
```

**Every single sample** produces the exact same binary code:
- 20 active neurons (5% sparsity) at positions: [0, 36-39, 385-399]
- Remaining 380 neurons are never selected

### Root Cause: Uniform Neuron Responses

#### Spike Count Analysis

| Metric | Value | Problem |
|--------|-------|---------|
| Neurons with spikes | 400 / 400 | ✅ All neurons fire |
| Spikes per neuron (std) | **0.00** | ❌ **All neurons fire IDENTICALLY** |
| Spikes per neuron (mean) | 559,354 | All neurons have same total spike count |
| Variance across samples per neuron | 6.94 | ✅ Some variance |
| Unique spike patterns | **20 / 10,000** | ❌ **Extremely low diversity** |

#### Per-Sample Analysis

```
Sample 0: All 400 neurons fire 53 spikes each → Total: 21,200 spikes
Sample 1: All 400 neurons fire 55 spikes each → Total: 22,000 spikes
Sample 2: All 400 neurons fire 47 spikes each → Total: 18,800 spikes
Sample 3: All 400 neurons fire 58 spikes each → Total: 23,200 spikes
Sample 4: All 400 neurons fire 54 spikes each → Total: 21,600 spikes
```

**Key Observation**: Within each sample, **all 400 neurons produce the exact same spike count**.

---

## 🧠 What This Means

### The Network Is Input-Agnostic

1. **All neurons respond identically** regardless of which neuron they are
2. **Each neuron's weight vector is essentially the same**
3. **STDP training did NOT differentiate neurons** to respond to different input patterns
4. **The network acts like a single uniform layer** rather than selective feature detectors

### Why Top-K Binarization Fails

When all neurons have the same spike count:
- Top-K selection becomes arbitrary (based on numerical ordering)
- Always selects the same neurons (e.g., neurons with smallest indices: 0, 36-39, 385-399)
- No input-dependent selectivity

---

## 🐛 Root Causes

### 1. ⚠️ Weight Normalization Too Strong

```python
# From encoder.py
norm=78.4,  # Weight normalization
```

**Problem**: Constant weight normalization may force all weight vectors to be too similar.

**Effect**: After normalization, all neurons have similar weight magnitudes, leading to similar responses.

### 2. ⚠️ Uniform Weight Initialization

**Problem**: Weights are initialized uniformly:
```python
wmin=-0.3, wmax=1.0  # Uniform initialization
```

**Effect**: If STDP doesn't cause sufficient differentiation, neurons remain similar.

### 3. ⚠️ Lateral Inhibition May Be Broken

```python
# Inhibitory -> Excitatory (all-to-all except diagonal)
w = 10.4 * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons)))
```

**Problem**: All neurons inhibit each other equally.

**Effect**: No winner-take-all competition; all neurons fire equally.

### 4. ⚠️ STDP Learning May Not Be Effective

```python
nu=(1e-4, 1e-2)  # Learning rates (pre=0.0001, post=0.01)
```

**Problem**: Learning rate might be too small or STDP updates might be negligible.

**Effect**: Weights don't differentiate during training.

### 5. ⚠️ Insufficient Training

```python
n_train_samples: null  # Uses all 60,000 samples
simulation_time: 350 ms per sample
```

**Problem**: While 60K samples seems sufficient, if weight changes are tiny, it's still not enough.

---

## 🔍 Comparison with Working Methods

### Krotov (MNIST NMI=0.59)
```
Unique codes: ~10,000 / 10,000 (high diversity)
Neurons with variance: High variance across samples
Learning mechanism: Works effectively
```

### FlyHash (MNIST NMI=0.55)
```
Unique codes: ~10,000 / 10,000 (high diversity)
No training needed: Random projection is sufficient
```

### Diehl & Cook (MNIST NMI=0.00)
```
Unique codes: 1 / 10,000 (no diversity)
All neurons respond identically
Training did not create selectivity
```

---

## 🛠️ Proposed Fixes

### Fix 1: Disable or Reduce Weight Normalization ⭐ (Most Likely)

```python
# Current
norm=78.4

# Proposed: Remove normalization entirely
norm=None  # or significantly reduce it

# Or use adaptive normalization only during training
```

**Rationale**: Allow weights to develop diverse magnitudes.

### Fix 2: Increase STDP Learning Rate

```python
# Current
nu=(1e-4, 1e-2)

# Proposed
nu=(1e-3, 1e-1)  # 10x increase
# or
nu=(1e-2, 1.0)   # 100x increase
```

**Rationale**: Make weight updates more significant.

### Fix 3: Add Random Initial Weight Diversity

```python
# Current
wmin=-0.3, wmax=1.0  # Too narrow range

# Proposed: Add Gaussian noise or wider range
wmin=-1.0, wmax=2.0
# or initialize with random patterns
```

**Rationale**: Start with more diverse weights.

### Fix 4: Fix Lateral Inhibition (Winner-Take-All)

The current implementation has:
```python
# Excitatory → Inhibitory: one-to-one (correct)
# Inhibitory → Excitatory: all-to-all with same weight (PROBLEM)
```

**Proposed**: Implement proper WTA with stronger/adaptive inhibition:
```python
# Use stronger inhibition weights
# Or implement k-winners-take-all
# Or use adaptive inhibition based on activity
```

### Fix 5: Verify STDP Is Actually Running

Add diagnostic output during training:
```python
# After each sample or every 100 samples, check:
# - Weight statistics (mean, std, min, max)
# - Has weight distribution changed?
# - Are weights becoming more diverse?
```

---

## 🎯 Recommended Action Plan

### Phase 1: Quick Validation (30 min)

1. **Test without weight normalization**
   ```python
   # In configs/diehl_cook.yaml, add:
   encoder_config:
     norm: null  # Disable normalization
   ```

2. **Run on small subset** (100 samples)
   ```bash
   python scripts/diagnose_diehl_cook.py  # Already created
   ```

3. **Check if codes become diverse**

### Phase 2: Parameter Tuning (2-3 hours)

If Fix 1 doesn't work:

1. **Increase learning rate**:
   ```yaml
   nu: [0.001, 0.1]  # 10x increase
   ```

2. **Adjust weight initialization**:
   ```yaml
   wmin: -1.0
   wmax: 2.0
   ```

3. **Test with 1,000 samples, verify diversity**

### Phase 3: Architecture Fix (if needed)

1. **Implement proper WTA** with adaptive inhibition
2. **Add weight monitoring** during training
3. **Verify STDP updates** are non-trivial

---

## 📚 Reference: Original Paper Hyperparameters

**Diehl & Cook (2015) used**:
- Input: 784 neurons (Poisson encoding, rates 0-63.75 Hz)
- Excitatory: 400 LIF neurons
- Threshold: -52 mV (with adaptive threshold)
- Weight normalization: **78.4** (same as ours)
- Learning rates: **nu_pre = 0.0001, nu_post = 0.01** (same as ours)
- Lateral inhibition: All-to-all except diagonal

**But they also had**:
- **Homeostatic normalization** (redistribute weights to maintain sum)
- **Training order randomization** (our implementation does this)
- **Per-neuron weight normalization** (sum to constant)

### Key Difference ⚠️

The paper's weight normalization is **per-neuron** (each neuron's total input weight sums to 78.4), not global. This maintains diversity between neurons while normalizing each neuron's scale.

**Our implementation might be normalizing globally**, making all neurons identical!

---

## 🔥 Most Likely Fix

### Implement Per-Neuron Weight Normalization

**Current**: Global normalization (might make all neurons similar)

**Should be**: Per-neuron normalization
```python
# After each weight update:
for neuron_idx in range(n_neurons):
    neuron_weights = weights[neuron_idx, :]
    neuron_weights *= 78.4 / neuron_weights.sum()
```

This is almost certainly the issue! BindsNET's `norm` parameter might not be doing per-neuron normalization as expected.

---

## 📊 Success Criteria

After fixes, we should see:
- ✅ **Unique codes > 5,000** (at least 50% diversity)
- ✅ **NMI > 0.3** (some clustering structure)
- ✅ **Neurons have different spike statistics** (std > 0)
- ✅ **Weight diversity increases** during training

---

## 🎓 Learning Points

1. **Weight normalization is critical but subtle** in SNNs
2. **Visual inspection of codes is essential** for debugging
3. **Aggregate statistics can hide problems** (e.g., "all neurons fire" sounds good, but they all fire identically!)
4. **STDP requires proper WTA competition** to develop selectivity

---

## 📎 Files for Debugging

- `scripts/quick_diagnose.py` - Check code diversity
- `scripts/deep_diagnose.py` - Analyze spike counts
- `scripts/diagnose_diehl_cook.py` - Full training + diagnosis
- `baselines/diehl_cook/encoder.py` - Implementation to fix

---

**Next Steps**: Implement per-neuron weight normalization and retest.
