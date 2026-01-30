#!/usr/bin/env python3
"""
Complete Diehl & Cook Training Script with Real STDP
====================================================
This script:
1. Clears old caches and checkpoints
2. Runs full dataset training with real STDP
3. Tests on multiple datasets with multiple seeds
4. Generates comprehensive reports

Usage:
    python scripts/run_diehl_cook_full.py [options]

Options:
    --datasets: Comma-separated list of datasets (default: mnist,fashion_mnist)
    --seeds: Comma-separated list of seeds (default: 0,1,2)
    --output-dir: Output directory (default: ./outputs)
    --skip-clear: Skip clearing cache (default: False)
    --dry-run: Show what would be done without doing it
"""

import argparse
import subprocess
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time

class DiehlCookFullRunner:
    """Runner for complete Diehl & Cook experiments with real STDP."""
    
    def __init__(
        self,
        datasets: List[str],
        seeds: List[int],
        output_dir: str = "./outputs",
        skip_clear: bool = False,
        dry_run: bool = False
    ):
        self.datasets = datasets
        self.seeds = seeds
        self.output_dir = Path(output_dir)
        self.skip_clear = skip_clear
        self.dry_run = dry_run
        self.baseline = "diehl_cook"
        
        # Results tracking
        self.results = {
            'start_time': datetime.now().isoformat(),
            'experiments': [],
            'summary': {}
        }
        
    def clear_old_results(self):
        """Clear old caches and checkpoints."""
        print("\n" + "="*80)
        print("🗑️  STEP 1: Clearing old results")
        print("="*80)
        
        if self.skip_clear:
            print("⏭️  Skipping cache clearing (--skip-clear enabled)")
            return
        
        # Clear cache
        cache_dir = self.output_dir / self.baseline
        if cache_dir.exists():
            print(f"\n📂 Found cache directory: {cache_dir}")
            if self.dry_run:
                print(f"   [DRY RUN] Would delete: {cache_dir}")
            else:
                try:
                    shutil.rmtree(cache_dir)
                    print(f"   ✅ Deleted: {cache_dir}")
                except Exception as e:
                    print(f"   ⚠️  Error deleting cache: {e}")
        else:
            print(f"✓ No cache found at {cache_dir}")
        
        print("\n✅ Old results cleared!")
        
    def run_experiment(self, dataset: str, seed: int) -> Dict:
        """Run a single experiment."""
        print("\n" + "="*80)
        print(f"🔥 Running: {self.baseline} on {dataset} (seed={seed})")
        print("="*80)
        
        start_time = time.time()
        
        # Build config path
        config_path = f"configs/{self.baseline}.yaml"
        
        cmd = [
            "python", "scripts/run_baseline.py",
            "--config", config_path,
            "--dataset", dataset,
            "--seed", str(seed)
        ]
        
        print(f"\n📝 Command: {' '.join(cmd)}")
        
        if self.dry_run:
            print("[DRY RUN] Would execute above command")
            return {
                'dataset': dataset,
                'seed': seed,
                'status': 'dry_run',
                'duration': 0
            }
        
        try:
            result = subprocess.run(
                cmd,
                cwd="/hy-tmp/clustering",
                capture_output=False,
                text=True,
                check=True
            )
            
            duration = time.time() - start_time
            
            print(f"\n✅ Completed in {duration:.1f}s")
            
            return {
                'dataset': dataset,
                'seed': seed,
                'status': 'success',
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"\n❌ Failed after {duration:.1f}s")
            print(f"Error: {e}")
            
            return {
                'dataset': dataset,
                'seed': seed,
                'status': 'failed',
                'duration': duration,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user!")
            raise
        
    def run_all_experiments(self):
        """Run all experiments."""
        print("\n" + "="*80)
        print("🚀 STEP 2: Running all experiments")
        print("="*80)
        print(f"\nDatasets: {', '.join(self.datasets)}")
        print(f"Seeds: {', '.join(map(str, self.seeds))}")
        print(f"Total experiments: {len(self.datasets) * len(self.seeds)}")
        
        experiment_num = 0
        total_experiments = len(self.datasets) * len(self.seeds)
        
        for dataset in self.datasets:
            for seed in self.seeds:
                experiment_num += 1
                print(f"\n{'='*80}")
                print(f"📊 Experiment {experiment_num}/{total_experiments}")
                print(f"{'='*80}")
                
                result = self.run_experiment(dataset, seed)
                self.results['experiments'].append(result)
                
                # Show progress
                success_count = sum(1 for r in self.results['experiments'] if r['status'] == 'success')
                failed_count = sum(1 for r in self.results['experiments'] if r['status'] == 'failed')
                print(f"\n📈 Progress: {experiment_num}/{total_experiments} "
                      f"(✅ {success_count} success, ❌ {failed_count} failed)")
        
    def generate_report(self):
        """Generate final report."""
        print("\n" + "="*80)
        print("📊 STEP 3: Generating report")
        print("="*80)
        
        # Calculate summary statistics
        total = len(self.results['experiments'])
        success = sum(1 for r in self.results['experiments'] if r['status'] == 'success')
        failed = sum(1 for r in self.results['experiments'] if r['status'] == 'failed')
        
        # Calculate average duration (only for completed experiments)
        completed_durations = [r['duration'] for r in self.results['experiments'] 
                              if r['status'] in ['success', 'failed']]
        avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
        total_time = sum(completed_durations)
        
        self.results['end_time'] = datetime.now().isoformat()
        self.results['summary'] = {
            'total_experiments': total,
            'successful': success,
            'failed': failed,
            'success_rate': f"{success/total*100:.1f}%" if total > 0 else "0%",
            'avg_duration_seconds': round(avg_duration, 1),
            'total_duration_seconds': round(total_time, 1),
            'total_duration_hours': round(total_time / 3600, 2)
        }
        
        # Save results to JSON
        results_file = self.output_dir / f"diehl_cook_full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("📋 SUMMARY")
        print("="*80)
        print(f"\nTotal experiments: {total}")
        print(f"✅ Successful: {success}")
        print(f"❌ Failed: {failed}")
        print(f"Success rate: {self.results['summary']['success_rate']}")
        print(f"\nAverage duration: {avg_duration:.1f}s ({avg_duration/60:.1f}m)")
        print(f"Total duration: {total_time/60:.1f}m ({total_time/3600:.2f}h)")
        
        # Print per-dataset summary
        print("\n" + "-"*80)
        print("Per-dataset results:")
        print("-"*80)
        
        for dataset in self.datasets:
            dataset_results = [r for r in self.results['experiments'] if r['dataset'] == dataset]
            dataset_success = sum(1 for r in dataset_results if r['status'] == 'success')
            dataset_total = len(dataset_results)
            print(f"\n{dataset}:")
            print(f"  ✅ {dataset_success}/{dataset_total} successful")
            for seed in self.seeds:
                seed_result = next((r for r in dataset_results if r['seed'] == seed), None)
                if seed_result:
                    status_icon = "✅" if seed_result['status'] == 'success' else "❌"
                    duration_str = f"{seed_result['duration']:.1f}s" if seed_result['duration'] > 0 else "N/A"
                    print(f"    {status_icon} seed={seed}: {seed_result['status']} ({duration_str})")
        
        # Print failed experiments details
        failed_experiments = [r for r in self.results['experiments'] if r['status'] == 'failed']
        if failed_experiments:
            print("\n" + "-"*80)
            print("Failed experiments details:")
            print("-"*80)
            for i, exp in enumerate(failed_experiments, 1):
                print(f"\n{i}. {exp['dataset']} (seed={exp['seed']})")
                if 'error' in exp:
                    print(f"   Error: {exp['error']}")
        
        print("\n" + "="*80)
        print("✅ All done!")
        print("="*80)
        print(f"\nResults saved to: {results_file}")
        print("\nNext steps:")
        print("  1. Check the results JSON file for detailed metrics")
        print("  2. View output directories for saved models and features")
        print(f"  3. Results location: {self.output_dir / self.baseline}")
        
    def run(self):
        """Run the complete pipeline."""
        print("\n" + "="*80)
        print("🔥 DIEHL & COOK FULL TRAINING WITH REAL STDP")
        print("="*80)
        print(f"\nBaseline: {self.baseline}")
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"Seeds: {', '.join(map(str, self.seeds))}")
        print(f"Output dir: {self.output_dir}")
        print(f"Dry run: {self.dry_run}")
        print(f"Skip clear: {self.skip_clear}")
        
        if not self.dry_run:
            print("\n⚠️  This will run REAL STDP training (much slower than skeleton!)")
            print("Expected time per experiment:")
            print("  - MNIST: ~30-60 minutes")
            print("  - Fashion-MNIST: ~30-60 minutes")
            print("  - CIFAR-10: ~60-120 minutes (if working)")
            
            total_time_estimate = len(self.datasets) * len(self.seeds) * 45  # 45 min average
            print(f"\nEstimated total time: ~{total_time_estimate/60:.1f} hours")
            
            response = input("\nContinue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        
        try:
            # Step 1: Clear old results
            self.clear_old_results()
            
            # Step 2: Run all experiments
            self.run_all_experiments()
            
            # Step 3: Generate report
            self.generate_report()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user!")
            print("Generating partial report...")
            self.results['end_time'] = datetime.now().isoformat()
            self.results['summary']['status'] = 'interrupted'
            self.generate_report()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete Diehl & Cook training with real STDP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on MNIST and Fashion-MNIST with 3 seeds
  python scripts/run_diehl_cook_full.py

  # Run on MNIST only with 5 seeds
  python scripts/run_diehl_cook_full.py --datasets mnist --seeds 0,1,2,3,4

  # Dry run to see what would happen
  python scripts/run_diehl_cook_full.py --dry-run

  # Skip clearing cache (resume training)
  python scripts/run_diehl_cook_full.py --skip-clear
        """
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        default='mnist,fashion_mnist',
        help='Comma-separated list of datasets (default: mnist,fashion_mnist)'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default='0,1,2',
        help='Comma-separated list of seeds (default: 0,1,2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory (default: ./outputs)'
    )
    parser.add_argument(
        '--skip-clear',
        action='store_true',
        help='Skip clearing old cache (resume training)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without doing it'
    )
    
    args = parser.parse_args()
    
    # Parse datasets and seeds
    datasets = [d.strip() for d in args.datasets.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Create and run
    runner = DiehlCookFullRunner(
        datasets=datasets,
        seeds=seeds,
        output_dir=args.output_dir,
        skip_clear=args.skip_clear,
        dry_run=args.dry_run
    )
    
    runner.run()


if __name__ == '__main__':
    main()
