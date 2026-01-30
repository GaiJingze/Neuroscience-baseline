#!/usr/bin/env python
"""
Check progress of Diehl & Cook batch training.

Usage:
    python scripts/check_diehl_cook_progress.py
"""

import json
from pathlib import Path
import numpy as np


def check_progress():
    """Check and display Diehl & Cook training progress."""
    
    output_dir = Path('./outputs')
    results_dir = output_dir / 'results'
    
    datasets = ['mnist', 'fashion_mnist', 'cifar10']
    seeds = [0, 1, 2]
    
    print("=" * 80)
    print("Diehl & Cook Training Progress")
    print("=" * 80)
    print()
    
    total_experiments = len(datasets) * len(seeds)
    completed_count = 0
    failed_count = 0
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}")
        print("-" * 80)
        print(f"{'Seed':<6} {'Status':<12} {'NMI':<10} {'ARI':<10} {'ACC':<10}")
        print("-" * 80)
        
        dataset_results = []
        
        for seed in seeds:
            result_file = results_dir / f'diehl_cook_{dataset}_seed{seed}.json'
            
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    
                    clustering = data.get('clustering', {}).get('kmeans', {})
                    nmi = clustering.get('nmi')
                    ari = clustering.get('ari')
                    acc = clustering.get('acc')
                    
                    if isinstance(nmi, (int, float)):
                        print(f"{seed:<6} {'✅ Complete':<12} {nmi:<10.4f} {ari:<10.4f} {acc:<10.4f}")
                        dataset_results.append({'nmi': nmi, 'ari': ari, 'acc': acc})
                        completed_count += 1
                    else:
                        print(f"{seed:<6} {'⚠️  Invalid':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
                        failed_count += 1
                
                except Exception as e:
                    print(f"{seed:<6} {'❌ Error':<12} {str(e)[:30]:<30}")
                    failed_count += 1
            else:
                print(f"{seed:<6} {'⏳ Pending':<12} {'-':<10} {'-':<10} {'-':<10}")
        
        # Show average for completed experiments
        if dataset_results:
            nmis = [r['nmi'] for r in dataset_results]
            aris = [r['ari'] for r in dataset_results]
            accs = [r['acc'] for r in dataset_results]
            
            print("-" * 80)
            print(f"{'Avg':<6} {'':<12} {np.mean(nmis):<10.4f} {np.mean(aris):<10.4f} {np.mean(accs):<10.4f}")
            if len(dataset_results) > 1:
                print(f"{'Std':<6} {'':<12} {np.std(nmis):<10.4f} {np.std(aris):<10.4f} {np.std(accs):<10.4f}")
    
    # Overall summary
    pending_count = total_experiments - completed_count - failed_count
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total experiments:  {total_experiments}")
    print(f"  ✅ Completed:     {completed_count} ({completed_count/total_experiments*100:.1f}%)")
    print(f"  ❌ Failed:        {failed_count}")
    print(f"  ⏳ Pending:       {pending_count}")
    print("=" * 80)
    
    # Check for checkpoint
    checkpoint_file = output_dir / 'diehl_cook_progress.json'
    if checkpoint_file.exists():
        print(f"\n📊 Checkpoint file: {checkpoint_file}")
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        print(f"   Last updated: {checkpoint.get('last_updated', 'N/A')}")
        
        if checkpoint.get('experiment_times'):
            times = checkpoint['experiment_times']
            avg_time = np.mean(list(times.values())) / 60
            print(f"   Average time: {avg_time:.1f} min per experiment")
            
            if pending_count > 0:
                est_remaining = avg_time * pending_count
                print(f"   Estimated remaining: {est_remaining:.1f} min ({est_remaining/60:.1f} hours)")
    
    # Show next steps
    if pending_count > 0:
        print("\n" + "=" * 80)
        print("Next Steps")
        print("=" * 80)
        print("\nTo continue training:")
        print("  python scripts/run_diehl_cook_batch.py --resume")
        print("\nOr run a specific experiment:")
        
        for dataset in datasets:
            for seed in seeds:
                result_file = results_dir / f'diehl_cook_{dataset}_seed{seed}.json'
                if not result_file.exists():
                    print(f"  python run.py --baseline diehl_cook --dataset {dataset} --seed {seed}")
                    break
            else:
                continue
            break
    else:
        print("\n🎉 All experiments completed!")
        print("\nView report:")
        print("  cat outputs/reports/diehl_cook_batch_report.txt")


if __name__ == '__main__':
    try:
        check_progress()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
