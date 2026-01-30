#!/usr/bin/env python3
"""
Diagnose SoftHebb encoding quality
"""
import numpy as np
from pathlib import Path

def diagnose_codes(codes_dir: str = "./outputs/codes/softhebb/mnist"):
    """Check if codes are diverse or degenerate."""
    
    codes_path = Path(codes_dir)
    
    print(f"\n{'='*80}")
    print(f"🔍 Diagnosing codes in: {codes_path}")
    print(f"{'='*80}\n")
    
    for seed in [0, 1, 2]:
        code_file = codes_path / f"code_seed{seed}.npy"
        pre_code_file = codes_path / f"pre_code_seed{seed}.npy"
        
        if not code_file.exists():
            print(f"⏭️  Skipping seed {seed} (file not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 Seed {seed}")
        print(f"{'='*60}")
        
        # Load codes
        codes = np.load(code_file)
        pre_codes = np.load(pre_code_file)
        
        print(f"\n📐 Shape:")
        print(f"  Binary codes: {codes.shape}")
        print(f"  Pre-codes: {pre_codes.shape}")
        
        # Check sparsity
        sparsity = 1.0 - np.mean(codes)
        print(f"\n🎯 Sparsity: {sparsity:.4f} ({np.mean(codes)*100:.2f}% active)")
        
        # Check if all samples have identical codes
        unique_codes = np.unique(codes, axis=0)
        print(f"\n🔢 Unique codes: {len(unique_codes)} / {len(codes)}")
        print(f"   Diversity: {len(unique_codes)/len(codes)*100:.2f}%")
        
        # Check pre-code statistics
        print(f"\n📈 Pre-code statistics:")
        print(f"   Mean: {np.mean(pre_codes):.4f}")
        print(f"   Std: {np.std(pre_codes):.4f}")
        print(f"   Min: {np.min(pre_codes):.4f}")
        print(f"   Max: {np.max(pre_codes):.4f}")
        print(f"   Median: {np.median(pre_codes):.4f}")
        
        # Check if all pre-codes are the same
        if np.std(pre_codes) < 1e-6:
            print(f"   ⚠️  WARNING: All pre-codes are nearly identical!")
        
        # Check for zero values
        zero_samples = np.sum(np.sum(pre_codes, axis=1) == 0)
        print(f"\n🔇 Samples with zero activation: {zero_samples} / {len(pre_codes)} ({zero_samples/len(pre_codes)*100:.2f}%)")
        
        # Check per-neuron activity
        per_neuron_activity = np.sum(codes, axis=0)
        active_neurons = np.sum(per_neuron_activity > 0)
        print(f"\n🧠 Neuron activity:")
        print(f"   Active neurons: {active_neurons} / {codes.shape[1]}")
        print(f"   Mean activations per neuron: {np.mean(per_neuron_activity):.2f}")
        print(f"   Std activations per neuron: {np.std(per_neuron_activity):.2f}")
        
        # Overall assessment
        print(f"\n{'='*60}")
        if len(unique_codes) < len(codes) * 0.1:
            print(f"❌ PROBLEM: Very low code diversity ({len(unique_codes)/len(codes)*100:.2f}%)")
            print(f"   This explains why clustering fails!")
        elif np.std(pre_codes) < 0.1:
            print(f"❌ PROBLEM: Nearly constant pre-code values")
            print(f"   The network is not learning meaningful representations")
        elif active_neurons < codes.shape[1] * 0.5:
            print(f"⚠️  WARNING: Only {active_neurons/codes.shape[1]*100:.1f}% neurons are active")
        else:
            print(f"✅ Codes look reasonable")
            if len(unique_codes)/len(codes) < 0.5:
                print(f"   But diversity could be better ({len(unique_codes)/len(codes)*100:.1f}%)")

def main():
    # Check MNIST
    print("\n" + "="*80)
    print("🔍 DIAGNOSING SOFTHEBB ENCODINGS")
    print("="*80)
    
    for dataset in ['mnist', 'fashion_mnist']:
        codes_path = Path(f"./outputs/codes/softhebb/{dataset}")
        if codes_path.exists():
            diagnose_codes(str(codes_path))
    
    print("\n" + "="*80)
    print("✅ Diagnosis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
