# Version Compatibility Status

## Current Environment

**Python**: 3.11  
**Date**: 2026-01-09

### Installed Versions

| Package | Version | Status |
|---------|---------|--------|
| numpy | 1.26.4 | ✅ Compatible |
| torch | 2.2.1+cu121 | ✅ Installed |
| torchvision | 0.17.1+cu121 | ✅ Installed |
| bindsnet | 0.2.7 | ⚠️ Has compatibility issues |
| scikit-learn | 1.8.0 | ✅ Compatible |

## Known Issues

### Issue 1: BindsNET 0.2.7 with PyTorch 2.2.1

**Error:**
```
ModuleNotFoundError: No module named 'torch._six'
```

**Root Cause:**
- BindsNET 0.2.7 (released ~2019-2020) uses `torch._six` module
- PyTorch 2.x removed `torch._six` (deprecated in PyTorch 1.8+)
- This is a known issue with older BindsNET versions

**Current Status:**
- ⚠️ BindsNET 0.2.7 is the latest version on PyPI (as of 2026-01)
- ⚠️ No version 0.3.x available on PyPI
- ✅ GitHub repository may have newer compatible code

## Solutions

### Solution 1: Install from GitHub (Recommended)

The GitHub version may have fixes for PyTorch 2.x compatibility:

```bash
pip uninstall bindsnet
pip install git+https://github.com/BindsNET/bindsnet.git
```

**Pros:**
- May have latest fixes
- Better PyTorch 2.x compatibility

**Cons:**
- Not a stable release
- May have breaking changes

### Solution 2: Use Compatible PyTorch Version

Downgrade PyTorch to 1.x series:

```bash
pip install torch==1.13.1 torchvision==0.14.1
pip install bindsnet==0.2.7
```

**Pros:**
- Stable versions
- Guaranteed compatibility

**Cons:**
- Older PyTorch version
- May not have latest features

### Solution 3: Patch BindsNET Locally

Create a compatibility patch for `torch._six`:

```bash
# Run the setup script with patching enabled
bash setup_bindsnet_env.sh --patch
```

**Pros:**
- Keep current versions
- Minimal changes

**Cons:**
- Requires manual patching
- May break with updates

### Solution 4: Use Alternative SNN Framework

For Diehl & Cook baseline, consider alternatives:

```bash
# SpikingJelly (actively maintained)
pip install spikingjelly

# Norse (PyTorch-based)
pip install norse
```

**Pros:**
- Better maintained
- PyTorch 2.x compatible

**Cons:**
- Different API
- Requires code adaptation

## Recommendations

### For Testing (Quick Start)

**Use FlyHash baseline first** - it has no dependencies issues:

```bash
python run.py --baseline flyhash
```

### For Production (Diehl & Cook)

**Option A: Use GitHub version**
```bash
pip uninstall bindsnet
pip install git+https://github.com/BindsNET/bindsnet.git
```

**Option B: Use compatible PyTorch**
```bash
pip install torch==1.13.1 torchvision==0.14.1
pip install bindsnet==0.2.7
```

### For Development

**Consider migrating to SpikingJelly or Norse:**
- Better maintained
- Modern PyTorch support
- Active community

## Version Matrix

| Configuration | NumPy | PyTorch | BindsNET | Status |
|---------------|-------|---------|----------|--------|
| **Current** | 1.26.4 | 2.2.1 | 0.2.7 (PyPI) | ❌ torch._six error |
| **GitHub** | 1.26.4 | 2.2.1 | git main | ❓ Unknown |
| **Legacy** | 1.21.x | 1.13.1 | 0.2.7 | ✅ Should work |
| **Alternative** | 1.26.4 | 2.2.1 | SpikingJelly | ✅ Compatible |

## Testing Results

### FlyHash Baseline
```bash
python run.py --baseline flyhash
```
**Status**: ✅ **WORKS** - No dependencies on BindsNET

### Diehl & Cook Baseline
```bash
python run.py --baseline diehl_cook
```
**Status**: ❌ **BLOCKED** - Requires BindsNET fix

## Next Steps

1. **Short term**: Use FlyHash baseline for testing
2. **Medium term**: Try GitHub BindsNET or downgrade PyTorch
3. **Long term**: Consider migrating to SpikingJelly/Norse

## References

- BindsNET GitHub: https://github.com/BindsNET/bindsnet
- PyPI Package: https://pypi.org/project/bindsnet/
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
- Norse: https://github.com/norse/norse

## Update History

- **2026-01-09**: Initial version status documentation
  - Identified PyPI BindsNET version is 0.2.7 (not 0.3.1)
  - Confirmed torch._six incompatibility with PyTorch 2.2.1
  - NumPy successfully downgraded to 1.26.4
  - FlyHash baseline confirmed working

---

**Last Updated**: 2026-01-09  
**Maintainer**: Jingze Gai  
**Priority**: High (blocks Diehl & Cook baseline)
