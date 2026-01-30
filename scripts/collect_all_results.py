#!/usr/bin/env python3
"""
Collect all baseline clustering results for comprehensive reporting
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def collect_results(output_dir: str = "./outputs") -> Dict:
    """Collect all clustering metrics from all experiments."""
    
    results_path = Path(output_dir) / "results"
    
    if not results_path.exists():
        print(f"❌ No results found at {results_path}")
        return {}
    
    # Collect results by baseline and dataset
    results = defaultdict(lambda: defaultdict(list))
    
    # Find all result files
    result_files = list(results_path.glob("*_seed*.json"))
    
    print(f"Found {len(result_files)} result files\n")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
        except:
            print(f"⚠️  Failed to read {result_file}")
            continue
        
        # Extract baseline, dataset, and seed from filename
        # Format: {baseline}_{dataset}_seed{seed}.json
        filename = result_file.stem
        parts = filename.rsplit('_seed', 1)
        
        if len(parts) != 2:
            continue
            
        baseline_dataset = parts[0]
        seed = int(parts[1])
        
        # Split baseline and dataset
        for baseline in ['flyhash', 'diehl_cook', 'softhebb']:
            if baseline_dataset.startswith(baseline):
                dataset = baseline_dataset[len(baseline)+1:]  # +1 for underscore
                break
        else:
            continue
        
        # Get clustering metrics (prefer kmeans)
        if 'clustering' not in data:
            continue
            
        metrics = None
        for method in ['kmeans', 'kmedoids', 'spectral']:
            if method in data['clustering']:
                metrics = data['clustering'][method]
                break
        
        if metrics:
            results[baseline][dataset].append({
                'seed': seed,
                'nmi': metrics.get('nmi', 0),
                'ari': metrics.get('ari', 0),
                'acc': metrics.get('acc', 0)
            })
    
    return results

def print_detailed_table(results: Dict):
    """Print detailed results table."""
    
    print("\n" + "="*100)
    print("📊 DETAILED CLUSTERING RESULTS")
    print("="*100)
    
    for baseline in sorted(results.keys()):
        print(f"\n{'='*100}")
        print(f"🔹 {baseline.upper()}")
        print(f"{'='*100}")
        
        for dataset in sorted(results[baseline].keys()):
            dataset_results = results[baseline][dataset]
            
            # Sort by seed
            dataset_results.sort(key=lambda x: x['seed'])
            
            # Calculate statistics
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
            print("-" * 100)
            print(f"{'Seed':<8} {'NMI':<15} {'ARI':<15} {'ACC':<15}")
            print("-" * 100)
            
            for r in dataset_results:
                print(f"{r['seed']:<8} {r['nmi']:<15.4f} {r['ari']:<15.4f} {r['acc']:<15.4f}")
            
            print("-" * 100)
            print(f"{'Mean':<8} {nmi_mean:<15.4f} {ari_mean:<15.4f} {acc_mean:<15.4f}")
            print(f"{'Std':<8} {nmi_std:<15.4f} {ari_std:<15.4f} {acc_std:<15.4f}")

def print_summary_table(results: Dict):
    """Print summary table for paper/report."""
    
    print("\n" + "="*100)
    print("📋 SUMMARY TABLE (Mean ± Std)")
    print("="*100)
    print()
    
    # Organize by dataset
    datasets = set()
    for baseline in results:
        datasets.update(results[baseline].keys())
    
    for dataset in sorted(datasets):
        print(f"\n{'='*100}")
        print(f"📁 {dataset.upper()}")
        print(f"{'='*100}")
        print(f"{'Baseline':<20} {'NMI':<25} {'ARI':<25} {'ACC':<25}")
        print("-" * 100)
        
        for baseline in sorted(results.keys()):
            if dataset not in results[baseline]:
                continue
                
            dataset_results = results[baseline][dataset]
            
            nmis = [r['nmi'] for r in dataset_results]
            aris = [r['ari'] for r in dataset_results]
            accs = [r['acc'] for r in dataset_results]
            
            nmi_mean = np.mean(nmis)
            nmi_std = np.std(nmis)
            ari_mean = np.mean(aris)
            ari_std = np.std(aris)
            acc_mean = np.mean(accs)
            acc_std = np.std(accs)
            
            nmi_str = f"{nmi_mean:.4f} ± {nmi_std:.4f}"
            ari_str = f"{ari_mean:.4f} ± {ari_std:.4f}"
            acc_str = f"{acc_mean:.4f} ± {acc_std:.4f}"
            
            print(f"{baseline:<20} {nmi_str:<25} {ari_str:<25} {acc_str:<25}")

def print_excel_format(results: Dict):
    """Print results in Excel/Google Sheets format."""
    
    print("\n" + "="*100)
    print("📊 COPY-PASTE FORMAT (for Excel/Google Sheets)")
    print("="*100)
    print()
    print("Baseline\tDataset\tNMI\tARI\tACC\tNMI_Std\tARI_Std\tACC_Std\tN_Seeds")
    print("-" * 100)
    
    for baseline in sorted(results.keys()):
        for dataset in sorted(results[baseline].keys()):
            dataset_results = results[baseline][dataset]
            
            nmis = [r['nmi'] for r in dataset_results]
            aris = [r['ari'] for r in dataset_results]
            accs = [r['acc'] for r in dataset_results]
            
            nmi_mean = np.mean(nmis)
            nmi_std = np.std(nmis)
            ari_mean = np.mean(aris)
            ari_std = np.std(aris)
            acc_mean = np.mean(accs)
            acc_std = np.std(accs)
            
            print(f"{baseline}\t{dataset}\t{nmi_mean:.4f}\t{ari_mean:.4f}\t{acc_mean:.4f}\t"
                  f"{nmi_std:.4f}\t{ari_std:.4f}\t{acc_std:.4f}\t{len(dataset_results)}")

def print_latex_table(results: Dict):
    """Print results in LaTeX table format."""
    
    print("\n" + "="*100)
    print("📄 LATEX TABLE FORMAT")
    print("="*100)
    print()
    
    # Organize by dataset
    datasets = sorted(set(dataset for baseline in results for dataset in results[baseline]))
    baselines = sorted(results.keys())
    
    print("% LaTeX table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Clustering Performance on MNIST and Fashion-MNIST}")
    print("\\begin{tabular}{l|ccc|ccc}")
    print("\\hline")
    print("\\multirow{2}{*}{Method} & \\multicolumn{3}{c|}{MNIST} & \\multicolumn{3}{c}{Fashion-MNIST} \\\\")
    print(" & NMI & ARI & ACC & NMI & ARI & ACC \\\\")
    print("\\hline")
    
    for baseline in baselines:
        row = [baseline.replace('_', '\\_')]
        
        for dataset in datasets:
            if dataset in results[baseline]:
                dataset_results = results[baseline][dataset]
                nmis = [r['nmi'] for r in dataset_results]
                aris = [r['ari'] for r in dataset_results]
                accs = [r['acc'] for r in dataset_results]
                
                nmi_mean = np.mean(nmis)
                ari_mean = np.mean(aris)
                acc_mean = np.mean(accs)
                
                row.append(f"{nmi_mean:.4f}")
                row.append(f"{ari_mean:.4f}")
                row.append(f"{acc_mean:.4f}")
            else:
                row.extend(["-", "-", "-"])
        
        print(" & ".join(row) + " \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

def print_statistics(results: Dict):
    """Print overall statistics."""
    
    print("\n" + "="*100)
    print("📈 OVERALL STATISTICS")
    print("="*100)
    
    total_experiments = 0
    by_baseline = defaultdict(int)
    by_dataset = defaultdict(int)
    
    for baseline in results:
        for dataset in results[baseline]:
            n = len(results[baseline][dataset])
            total_experiments += n
            by_baseline[baseline] += n
            by_dataset[dataset] += n
    
    print(f"\nTotal experiments: {total_experiments}")
    print(f"\nBy baseline:")
    for baseline in sorted(by_baseline.keys()):
        print(f"  {baseline}: {by_baseline[baseline]} experiments")
    
    print(f"\nBy dataset:")
    for dataset in sorted(by_dataset.keys()):
        print(f"  {dataset}: {by_dataset[dataset]} experiments")

def main():
    print("\n" + "="*100)
    print("🔍 COLLECTING ALL RESULTS")
    print("="*100)
    
    results = collect_results()
    
    if not results:
        print("\n❌ No results found!")
        print("Make sure you have run the experiments first.")
        return
    
    # Print all formats
    print_detailed_table(results)
    print_summary_table(results)
    print_excel_format(results)
    print_latex_table(results)
    print_statistics(results)
    
    print("\n" + "="*100)
    print("✅ Results collection complete!")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
