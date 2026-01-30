#!/usr/bin/env python3
"""
Collect FlyHash clustering results for table reporting
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

def collect_results(output_dir: str = "./outputs") -> Dict:
    """Collect all clustering metrics from FlyHash experiments."""
    
    results_path = Path(output_dir) / "results"
    
    if not results_path.exists():
        print(f"❌ No results found at {results_path}")
        return {}
    
    # Collect results by dataset
    results_by_dataset = {}
    
    # Find all flyhash result files
    result_files = list(results_path.glob("flyhash_*_seed*.json"))
    
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract dataset name and seed from filename
        # Format: flyhash_{dataset}_seed{seed}.json
        filename = result_file.stem  # e.g., "flyhash_mnist_seed0"
        parts = filename.replace('flyhash_', '').split('_seed')
        if len(parts) == 2:
            dataset = parts[0]
            seed = int(parts[1])
        else:
            continue
        
        # Get clustering metrics (use kmeans)
        if 'clustering' in data and 'kmeans' in data['clustering']:
            metrics = data['clustering']['kmeans']
            
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = []
            
            results_by_dataset[dataset].append({
                'seed': seed,
                'nmi': metrics.get('nmi', 0),
                'ari': metrics.get('ari', 0),
                'acc': metrics.get('acc', 0)
            })
    
    return results_by_dataset

def print_table(results: Dict):
    """Print results in table format."""
    
    print("\n" + "="*80)
    print("📊 FlyHash Clustering Results")
    print("="*80)
    
    for dataset in sorted(results.keys()):
        dataset_results = results[dataset]
        
        # Sort by seed
        dataset_results.sort(key=lambda x: x['seed'])
        
        # Calculate mean and std
        nmis = [r['nmi'] for r in dataset_results]
        aris = [r['ari'] for r in dataset_results]
        accs = [r['acc'] for r in dataset_results]
        
        nmi_mean = np.mean(nmis)
        nmi_std = np.std(nmis)
        ari_mean = np.mean(aris)
        ari_std = np.std(aris)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        
        print(f"\n📁 {dataset.upper()}")
        print("-" * 80)
        print(f"{'Seed':<8} {'NMI':<12} {'ARI':<12} {'ACC':<12}")
        print("-" * 80)
        
        for r in dataset_results:
            print(f"{r['seed']:<8} {r['nmi']:<12.4f} {r['ari']:<12.4f} {r['acc']:<12.4f}")
        
        print("-" * 80)
        print(f"{'Mean':<8} {nmi_mean:<12.4f} {ari_mean:<12.4f} {acc_mean:<12.4f}")
        print(f"{'Std':<8} {nmi_std:<12.4f} {ari_std:<12.4f} {acc_std:<12.4f}")
    
    print("\n" + "="*80)
    print("📋 Summary for Table (Mean ± Std)")
    print("="*80)
    
    for dataset in sorted(results.keys()):
        dataset_results = results[dataset]
        
        nmis = [r['nmi'] for r in dataset_results]
        aris = [r['ari'] for r in dataset_results]
        accs = [r['acc'] for r in dataset_results]
        
        nmi_mean = np.mean(nmis)
        nmi_std = np.std(nmis)
        ari_mean = np.mean(aris)
        ari_std = np.std(aris)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        
        print(f"\n{dataset.upper()}:")
        print(f"  NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")
        print(f"  ARI: {ari_mean:.4f} ± {ari_std:.4f}")
        print(f"  ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    
    print("\n" + "="*80)
    print("📊 Copy-Paste Format (for Excel/Sheets)")
    print("="*80)
    print("\nBaseline\tDataset\tNMI\tARI\tACC")
    
    for dataset in sorted(results.keys()):
        dataset_results = results[dataset]
        
        nmis = [r['nmi'] for r in dataset_results]
        aris = [r['ari'] for r in dataset_results]
        accs = [r['acc'] for r in dataset_results]
        
        nmi_mean = np.mean(nmis)
        ari_mean = np.mean(aris)
        acc_mean = np.mean(accs)
        
        print(f"FlyHash\t{dataset}\t{nmi_mean:.4f}\t{ari_mean:.4f}\t{acc_mean:.4f}")
    
    print("\n")

def main():
    results = collect_results()
    
    if not results:
        print("\n❌ No results found!")
        print("Make sure you have run the experiments first:")
        print("  python scripts/run_baseline.py --config configs/flyhash.yaml --dataset mnist --seed 0")
        return
    
    print_table(results)

if __name__ == '__main__':
    main()
