#!/usr/bin/env python
"""
Batch training script for Diehl & Cook baseline.

This script automatically runs all remaining Diehl & Cook experiments,
tracks progress, handles errors, and generates summary reports.

Usage:
    # Run all remaining experiments
    python scripts/run_diehl_cook_batch.py
    
    # Run only MNIST
    python scripts/run_diehl_cook_batch.py --datasets mnist
    
    # Run specific seeds
    python scripts/run_diehl_cook_batch.py --seeds 1 2
    
    # Skip CIFAR-10 (fast mode)
    python scripts/run_diehl_cook_batch.py --skip-cifar10
    
    # Resume from checkpoint
    python scripts/run_diehl_cook_batch.py --resume
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import sys
import numpy as np


class DiehlCookBatchRunner:
    """Batch runner for Diehl & Cook experiments."""
    
    def __init__(
        self,
        datasets: List[str] = None,
        seeds: List[int] = None,
        output_dir: str = './outputs',
        checkpoint_file: str = './outputs/diehl_cook_progress.json'
    ):
        self.datasets = datasets or ['mnist', 'fashion_mnist', 'cifar10']
        self.seeds = seeds or [0, 1, 2]
        self.output_dir = Path(output_dir)
        self.checkpoint_file = Path(checkpoint_file)
        
        # Track progress
        self.completed = set()
        self.failed = {}
        self.start_time = None
        self.experiment_times = {}
        
        # Load checkpoint if exists
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load progress from checkpoint file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.completed = set(data.get('completed', []))
                self.failed = data.get('failed', {})
                self.experiment_times = data.get('experiment_times', {})
            print(f"✅ Loaded checkpoint: {len(self.completed)} completed, {len(self.failed)} failed")
    
    def save_checkpoint(self):
        """Save progress to checkpoint file."""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'completed': list(self.completed),
                'failed': self.failed,
                'experiment_times': self.experiment_times,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def is_experiment_completed(self, dataset: str, seed: int) -> bool:
        """Check if experiment is already completed."""
        exp_id = f"{dataset}_seed{seed}"
        
        # Check in memory
        if exp_id in self.completed:
            return True
        
        # Check result file
        result_file = self.output_dir / 'results' / f'diehl_cook_{dataset}_seed{seed}.json'
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                # Verify result is valid
                if 'clustering' in data and 'kmeans' in data['clustering']:
                    nmi = data['clustering']['kmeans'].get('nmi')
                    if isinstance(nmi, (int, float)):
                        self.completed.add(exp_id)
                        return True
            except Exception as e:
                print(f"⚠️  Warning: Invalid result file {result_file}: {e}")
        
        return False
    
    def get_pending_experiments(self) -> List[Tuple[str, int]]:
        """Get list of pending experiments."""
        pending = []
        for dataset in self.datasets:
            for seed in self.seeds:
                exp_id = f"{dataset}_seed{seed}"
                if not self.is_experiment_completed(dataset, seed):
                    pending.append((dataset, seed))
        return pending
    
    def estimate_time(self, dataset: str) -> str:
        """Estimate training time for a dataset."""
        times = {
            'mnist': '30-60 min',
            'fashion_mnist': '30-60 min',
            'cifar10': '60-120 min'
        }
        return times.get(dataset, 'unknown')
    
    def run_experiment(self, dataset: str, seed: int) -> bool:
        """Run a single experiment."""
        exp_id = f"{dataset}_seed{seed}"
        
        print("\n" + "=" * 70)
        print(f"🚀 Running: {exp_id}")
        print(f"   Estimated time: {self.estimate_time(dataset)}")
        print("=" * 70)
        
        # Build command
        cmd = [
            'python', 'run.py',
            '--baseline', 'diehl_cook',
            '--dataset', dataset,
            '--seed', str(seed)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True
            )
            
            elapsed = time.time() - start_time
            self.experiment_times[exp_id] = elapsed
            
            # Verify result was created
            if self.is_experiment_completed(dataset, seed):
                self.completed.add(exp_id)
                self.save_checkpoint()
                
                print("\n" + "=" * 70)
                print(f"✅ Completed: {exp_id} ({elapsed/60:.1f} min)")
                print("=" * 70)
                return True
            else:
                raise Exception("Result file not created or invalid")
        
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            error_msg = f"Process exited with code {e.returncode}"
            self.failed[exp_id] = {
                'error': error_msg,
                'time': datetime.now().isoformat(),
                'elapsed': elapsed
            }
            self.save_checkpoint()
            
            print("\n" + "=" * 70)
            print(f"❌ Failed: {exp_id}")
            print(f"   Error: {error_msg}")
            print("=" * 70)
            return False
        
        except Exception as e:
            elapsed = time.time() - start_time
            self.failed[exp_id] = {
                'error': str(e),
                'time': datetime.now().isoformat(),
                'elapsed': elapsed
            }
            self.save_checkpoint()
            
            print("\n" + "=" * 70)
            print(f"❌ Failed: {exp_id}")
            print(f"   Error: {e}")
            print("=" * 70)
            return False
    
    def print_summary(self):
        """Print summary of all experiments."""
        print("\n" + "=" * 70)
        print("📊 Batch Training Summary")
        print("=" * 70)
        
        total = len(self.datasets) * len(self.seeds)
        completed = len(self.completed)
        failed = len(self.failed)
        pending = total - completed - failed
        
        print(f"\nTotal experiments: {total}")
        print(f"  ✅ Completed: {completed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⏳ Pending: {pending}")
        
        if self.completed:
            print(f"\n✅ Completed experiments:")
            for exp_id in sorted(self.completed):
                time_str = ""
                if exp_id in self.experiment_times:
                    mins = self.experiment_times[exp_id] / 60
                    time_str = f" ({mins:.1f} min)"
                print(f"   - {exp_id}{time_str}")
        
        if self.failed:
            print(f"\n❌ Failed experiments:")
            for exp_id, info in sorted(self.failed.items()):
                print(f"   - {exp_id}: {info['error']}")
        
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\nTotal batch time: {total_time/3600:.2f} hours")
        
        print("=" * 70)
    
    def generate_report(self):
        """Generate detailed report of all results."""
        report_file = self.output_dir / 'reports' / 'diehl_cook_batch_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Diehl & Cook Batch Training Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            total = len(self.datasets) * len(self.seeds)
            completed = len(self.completed)
            failed = len(self.failed)
            
            f.write("Summary\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total experiments: {total}\n")
            f.write(f"Completed: {completed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success rate: {completed/total*100:.1f}%\n\n")
            
            # Results by dataset
            f.write("Results by Dataset\n")
            f.write("-" * 80 + "\n\n")
            
            for dataset in self.datasets:
                f.write(f"{dataset.upper()}\n")
                f.write("-" * 40 + "\n")
                
                results = []
                for seed in self.seeds:
                    result_file = self.output_dir / 'results' / f'diehl_cook_{dataset}_seed{seed}.json'
                    if result_file.exists():
                        try:
                            with open(result_file) as rf:
                                data = json.load(rf)
                            clustering = data.get('clustering', {}).get('kmeans', {})
                            results.append({
                                'seed': seed,
                                'nmi': clustering.get('nmi', 'N/A'),
                                'ari': clustering.get('ari', 'N/A'),
                                'acc': clustering.get('acc', 'N/A')
                            })
                        except Exception as e:
                            results.append({
                                'seed': seed,
                                'error': str(e)
                            })
                
                if results:
                    # Print table
                    f.write(f"{'Seed':<6} {'NMI':<10} {'ARI':<10} {'ACC':<10}\n")
                    f.write("-" * 40 + "\n")
                    
                    nmis, aris, accs = [], [], []
                    for r in results:
                        if 'error' not in r:
                            seed = r['seed']
                            nmi = r['nmi']
                            ari = r['ari']
                            acc = r['acc']
                            
                            if isinstance(nmi, (int, float)):
                                f.write(f"{seed:<6} {nmi:<10.4f} {ari:<10.4f} {acc:<10.4f}\n")
                                nmis.append(nmi)
                                aris.append(ari)
                                accs.append(acc)
                            else:
                                f.write(f"{seed:<6} {str(nmi):<10} {str(ari):<10} {str(acc):<10}\n")
                    
                    # Average
                    if nmis:
                        f.write("-" * 40 + "\n")
                        f.write(f"{'Avg':<6} {sum(nmis)/len(nmis):<10.4f} "
                               f"{sum(aris)/len(aris):<10.4f} {sum(accs)/len(accs):<10.4f}\n")
                        f.write(f"{'Std':<6} {np.std(nmis):<10.4f} "
                               f"{np.std(aris):<10.4f} {np.std(accs):<10.4f}\n")
                
                f.write("\n")
            
            # Training times
            if self.experiment_times:
                f.write("Training Times\n")
                f.write("-" * 80 + "\n")
                for exp_id, elapsed in sorted(self.experiment_times.items()):
                    f.write(f"{exp_id:<30} {elapsed/60:>6.1f} min\n")
                f.write("\n")
            
            # Failed experiments
            if self.failed:
                f.write("Failed Experiments\n")
                f.write("-" * 80 + "\n")
                for exp_id, info in sorted(self.failed.items()):
                    f.write(f"{exp_id}: {info['error']}\n")
        
        print(f"\n📄 Report saved to: {report_file}")
        
        # Also generate JSON report
        json_report_file = self.output_dir / 'reports' / 'diehl_cook_batch_report.json'
        with open(json_report_file, 'w') as f:
            json.dump({
                'completed': list(self.completed),
                'failed': self.failed,
                'experiment_times': self.experiment_times,
                'generated': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"📄 JSON report saved to: {json_report_file}")
    
    def run_all(self, skip_on_error: bool = False):
        """Run all pending experiments."""
        self.start_time = time.time()
        
        pending = self.get_pending_experiments()
        
        if not pending:
            print("✅ All experiments already completed!")
            self.print_summary()
            return
        
        print("\n" + "=" * 70)
        print(f"🚀 Diehl & Cook Batch Training")
        print("=" * 70)
        print(f"\nPending experiments: {len(pending)}")
        for dataset, seed in pending:
            print(f"  - {dataset}_seed{seed} (est. {self.estimate_time(dataset)})")
        print()
        
        # Confirm
        response = input("Start batch training? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
        
        # Run experiments
        for i, (dataset, seed) in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}]")
            success = self.run_experiment(dataset, seed)
            
            if not success and not skip_on_error:
                print("\n⚠️  Experiment failed. Continue? (y/N): ")
                response = input()
                if response.lower() != 'y':
                    print("❌ Batch training stopped")
                    break
        
        # Final summary
        self.print_summary()
        
        # Generate report
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(
        description='Batch training for Diehl & Cook baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all remaining experiments
  python scripts/run_diehl_cook_batch.py
  
  # Run only MNIST
  python scripts/run_diehl_cook_batch.py --datasets mnist
  
  # Run only seeds 1 and 2
  python scripts/run_diehl_cook_batch.py --seeds 1 2
  
  # Skip CIFAR-10 (fast mode)
  python scripts/run_diehl_cook_batch.py --skip-cifar10
  
  # Resume from checkpoint
  python scripts/run_diehl_cook_batch.py --resume
  
  # Continue on error
  python scripts/run_diehl_cook_batch.py --skip-on-error
"""
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['mnist', 'fashion_mnist', 'cifar10'],
        help='Datasets to run (default: all)'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        help='Seeds to run (default: 0 1 2)'
    )
    parser.add_argument(
        '--skip-cifar10',
        action='store_true',
        help='Skip CIFAR-10 (faster)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--skip-on-error',
        action='store_true',
        help='Continue on error without prompting'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Determine datasets
    datasets = args.datasets
    if datasets is None:
        datasets = ['mnist', 'fashion_mnist']
        if not args.skip_cifar10:
            datasets.append('cifar10')
    
    # Create runner
    runner = DiehlCookBatchRunner(
        datasets=datasets,
        seeds=args.seeds,
        output_dir=args.output_dir
    )
    
    # Run
    runner.run_all(skip_on_error=args.skip_on_error)


if __name__ == '__main__':
    main()
