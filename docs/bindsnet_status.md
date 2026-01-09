# BindsNET Integration Status

## Current Status: ⚠️ Version Compatibility Issue

**Last Checked**: 2026-01-09

### Issue

BindsNET has compatibility issues with:
- PyTorch >= 2.0 (missing `torch._six` module)
- NumPy >= 2.0

### Attempted Solutions

1. ✅ NumPy downgrade to 1.26.4 - **Success**
2. ⚠️ BindsNET import still fails due to PyTorch version

### Working Configuration (Recommended)

Based on BindsNET requirements, the working combination is:

```
torch==1.13.1  # Or any 1.x version
torchvision==0.14.1
numpy==1.24.3
bindsnet==0.3.1
```

### Options Moving Forward

#### Option 1: Use Compatible Versions (Recommended)

Create a separate environment for BindsNET:

```bash
# Create isolated environment
conda create -n bindsnet_env python=3.9
conda activate bindsnet_env

# Install compatible versions
pip install torch==1.13.1 torchvision==0.14.1
pip install numpy==1.24.3
pip install bindsnet

# Test
python -c "import bindsnet; print('OK')"
```

Then:
- Train in `bindsnet_env`
- Export features as `.npy` files
- Load features in main environment for evaluation

#### Option 2: Use Alternative SNN Framework

Replace BindsNET with SpikingJelly or Norse:

```bash
pip install spikingjelly
# Or
pip install norse-torch
```

#### Option 3: Simplified Implementation

Use our encoder skeleton without full BindsNET training:
- Keep the interface
- Use simplified STDP approximation
- Focus on getting baseline numbers

### Recommendation

**For immediate progress**: Use **Option 3** (simplified implementation)
- `baselines/diehl_cook/encoder.py` already has a working skeleton
- Can generate features for evaluation
- Iterate on improving the implementation later

**For full STDP**: Use **Option 1** (isolated environment)
- More setup work but authentic implementation
- Better for publication-quality results

### Next Steps

1. **Short-term** (this week):
   - Use encoder skeleton with placeholder training
   - Get pipeline working end-to-end
   - Generate initial baseline results

2. **Medium-term** (next 2 weeks):
   - Set up isolated BindsNET environment
   - Implement full STDP training
   - Compare with simplified version

3. **Documentation**:
   - Update requirements.txt with version notes
   - Create setup guide for BindsNET environment
   - Document the two-environment workflow

### Files Modified

- ✅ `requirements.txt` - Added NumPy < 2.0 constraint
- ✅ `baselines/diehl_cook/encoder.py` - Skeleton ready
- ✅ `baselines/diehl_cook/train.py` - Full training script (needs compatible env)
- ✅ `baselines/diehl_cook/README.md` - Documentation
- ✅ `BINDSNET_INTEGRATION.md` - Integration guide
- ✅ `TROUBLESHOOTING.md` - Common issues
- ✅ `INSTALL.md` - Installation guide

### Impact on Project Timeline

- **No delay**: Encoder interface works
- **Pipeline ready**: Can run with simplified STDP
- **Full implementation**: Can be done in parallel

### Contact

For questions about BindsNET integration:
- Check `BINDSNET_INTEGRATION.md`
- See `TROUBLESHOOTING.md`
- Contact: Jingze Gai

---

**Status**: 🟡 Partial (interface ready, full training needs isolated env)  
**Priority**: Medium (can proceed with simplified version)  
**Blocker**: No (workarounds available)
