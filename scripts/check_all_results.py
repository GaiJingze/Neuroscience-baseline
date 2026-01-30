#!/usr/bin/env python3
"""Check completion status and collect all clustering results."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def check_all_results():
    results_dir = Path('./outputs/results')
    
    # Define expected tests
    baselines = ['flyhash', 'krotov', 'softhebb', 'diehl_cook']
    datasets = ['mnist', 'fashion_mnist']
    seeds = [0, 1, 2]
    
    print("=" * 80)
    print("CLUSTERING PIPELINE - TEST COMPLETION STATUS")
    print("=" * 80)
    print()
    
    # Track completion
    completion_status = defaultdict(lambda: defaultdict(lambda: {'completed': [], 'missing': []}))
    all_results = defaultdict(lambda: defaultdict(list))
    
    # Check each combination
    for baseline in baselines:
        for dataset in datasets:
            for seed in seeds:
                result_file = results_dir / f"{baseline}_{dataset}_seed{seed}.json"
                
                if result_file.exists():
                    completion_status[baseline][dataset]['completed'].append(seed)
                    # Load results
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            all_results[baseline][dataset].append({
                                'seed': seed,
                                'data': data
                            })
                    except Exception as e:
                        print(f"⚠️  Error loading {result_file}: {e}")
                else:
                    completion_status[baseline][dataset]['missing'].append(seed)
    
    # Print completion matrix
    print("📊 Completion Status Matrix:")
    print()
    print(f"{'Baseline':<15} {'MNIST':<20} {'Fashion-MNIST':<20}")
    print("-" * 60)
    
    for baseline in baselines:
        mnist_status = f"{len(completion_status[baseline]['mnist']['completed'])}/3"
        fashion_status = f"{len(completion_status[baseline]['fashion_mnist']['completed'])}/3"
        
        mnist_icon = "✅" if len(completion_status[baseline]['mnist']['completed']) == 3 else "⚠️"
        fashion_icon = "✅" if len(completion_status[baseline]['fashion_mnist']['completed']) == 3 else "⚠️"
        
        print(f"{baseline:<15} {mnist_icon} {mnist_status:<18} {fashion_icon} {fashion_status:<18}")
    
    print()
    
    # Total completion
    total_tests = len(baselines) * len(datasets) * len(seeds)
    completed_tests = sum(
        len(completion_status[b][d]['completed'])
        for b in baselines
        for d in datasets
    )
    
    print(f"Total: {completed_tests}/{total_tests} tests completed ({completed_tests/total_tests*100:.1f}%)")
    print()
    
    # Show missing tests
    missing_tests = []
    for baseline in baselines:
        for dataset in datasets:
            missing = completion_status[baseline][dataset]['missing']
            if missing:
                for seed in missing:
                    missing_tests.append(f"{baseline}_{dataset}_seed{seed}")
    
    if missing_tests:
        print("⚠️  Missing tests:")
        for test in missing_tests:
            print(f"   - {test}")
        print()
    else:
        print("✅ All tests completed!")
        print()
    
    # Performance summary
    print("=" * 80)
    print("PERFORMANCE SUMMARY (K-means Clustering)")
    print("=" * 80)
    print()
    
    for baseline in baselines:
        print(f"\n{'─' * 80}")
        print(f"📌 {baseline.upper()}")
        print(f"{'─' * 80}")
        
        for dataset in datasets:
            results_list = all_results[baseline][dataset]
            
            if not results_list:
                print(f"\n{dataset}: No results")
                continue
            
            print(f"\n{dataset}:")
            
            # Show individual seed results
            nmi_values = []
            ari_values = []
            acc_values = []
            
            for result_dict in sorted(results_list, key=lambda x: x['seed']):
                seed = result_dict['seed']
                data = result_dict['data']
                
                # Try both possible keys for clustering results
                clustering_data = data.get('clustering') or data.get('clustering_results')
                
                if clustering_data and 'kmeans' in clustering_data:
                    kmeans_result = clustering_data['kmeans']
                    nmi = kmeans_result.get('nmi', 'N/A')
                    ari = kmeans_result.get('ari', 'N/A')
                    acc = kmeans_result.get('acc', 'N/A')
                    
                    if isinstance(nmi, (int, float)):
                        print(f"  Seed {seed}: NMI={nmi:.4f}, ARI={ari:.4f}, ACC={acc:.4f}")
                        nmi_values.append(nmi)
                        ari_values.append(ari)
                        acc_values.append(acc)
                    else:
                        print(f"  Seed {seed}: NMI={nmi}, ARI={ari}, ACC={acc}")
                else:
                    print(f"  Seed {seed}: No kmeans results")
            
            # Calculate mean and std if we have multiple seeds
            if len(nmi_values) >= 2:
                nmi_mean, nmi_std = np.mean(nmi_values), np.std(nmi_values)
                ari_mean, ari_std = np.mean(ari_values), np.std(ari_values)
                acc_mean, acc_std = np.mean(acc_values), np.std(acc_values)
                
                print(f"\n  📊 Mean ± Std (n={len(nmi_values)}):")
                print(f"     NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")
                print(f"     ARI: {ari_mean:.4f} ± {ari_std:.4f}")
                print(f"     ACC: {acc_mean:.4f} ± {acc_std:.4f}")
            elif len(nmi_values) == 1:
                print(f"\n  📊 Single result (n=1)")
    
    print()
    print("=" * 80)
    
    # Generate comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE - MNIST")
    print("=" * 80)
    print()
    print(f"{'Method':<15} {'NMI':<20} {'ARI':<20} {'ACC':<20}")
    print("-" * 80)
    
    for baseline in baselines:
        results_list = all_results[baseline]['mnist']
        
        if results_list:
            nmi_values = []
            ari_values = []
            acc_values = []
            
            for result_dict in results_list:
                data = result_dict['data']
                clustering_data = data.get('clustering') or data.get('clustering_results')
                
                if clustering_data and 'kmeans' in clustering_data:
                    kmeans_result = clustering_data['kmeans']
                    nmi = kmeans_result.get('nmi')
                    ari = kmeans_result.get('ari')
                    acc = kmeans_result.get('acc')
                    
                    if isinstance(nmi, (int, float)):
                        nmi_values.append(nmi)
                        ari_values.append(ari)
                        acc_values.append(acc)
            
            if nmi_values:
                if len(nmi_values) >= 2:
                    nmi_str = f"{np.mean(nmi_values):.4f} ± {np.std(nmi_values):.4f}"
                    ari_str = f"{np.mean(ari_values):.4f} ± {np.std(ari_values):.4f}"
                    acc_str = f"{np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}"
                else:
                    nmi_str = f"{nmi_values[0]:.4f}"
                    ari_str = f"{ari_values[0]:.4f}"
                    acc_str = f"{acc_values[0]:.4f}"
                
                print(f"{baseline:<15} {nmi_str:<20} {ari_str:<20} {acc_str:<20}")
            else:
                print(f"{baseline:<15} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
        else:
            print(f"{baseline:<15} {'No results':<20} {'':<20} {'':<20}")
    
    print()
    print("=" * 80)
    print("COMPARISON TABLE - Fashion-MNIST")
    print("=" * 80)
    print()
    print(f"{'Method':<15} {'NMI':<20} {'ARI':<20} {'ACC':<20}")
    print("-" * 80)
    
    for baseline in baselines:
        results_list = all_results[baseline]['fashion_mnist']
        
        if results_list:
            nmi_values = []
            ari_values = []
            acc_values = []
            
            for result_dict in results_list:
                data = result_dict['data']
                clustering_data = data.get('clustering') or data.get('clustering_results')
                
                if clustering_data and 'kmeans' in clustering_data:
                    kmeans_result = clustering_data['kmeans']
                    nmi = kmeans_result.get('nmi')
                    ari = kmeans_result.get('ari')
                    acc = kmeans_result.get('acc')
                    
                    if isinstance(nmi, (int, float)):
                        nmi_values.append(nmi)
                        ari_values.append(ari)
                        acc_values.append(acc)
            
            if nmi_values:
                if len(nmi_values) >= 2:
                    nmi_str = f"{np.mean(nmi_values):.4f} ± {np.std(nmi_values):.4f}"
                    ari_str = f"{np.mean(ari_values):.4f} ± {np.std(ari_values):.4f}"
                    acc_str = f"{np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}"
                else:
                    nmi_str = f"{nmi_values[0]:.4f}"
                    ari_str = f"{ari_values[0]:.4f}"
                    acc_str = f"{acc_values[0]:.4f}"
                
                print(f"{baseline:<15} {nmi_str:<20} {ari_str:<20} {acc_str:<20}")
            else:
                print(f"{baseline:<15} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
        else:
            print(f"{baseline:<15} {'No results':<20} {'':<20} {'':<20}")
    
    print()
    print("=" * 80)
    print()

if __name__ == '__main__':
    check_all_results()
