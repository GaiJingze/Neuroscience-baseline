#!/usr/bin/env python
"""
Test single or multiple baselines with flexible options.

Usage:
    # Test single baseline
    python scripts/test_baseline.py flyhash

    # Test multiple baselines
    python scripts/test_baseline.py flyhash diehl_cook

    # Test with options
    python scripts/test_baseline.py flyhash --dataset mnist --seeds 0 1 2
    
    # Test all available baselines
    python scripts/test_baseline.py --all
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Available baselines
AVAILABLE_BASELINES = {
    'flyhash': {
        'config': 'configs/flyhash.yaml',
        'description': 'FlyHash - Fruit fly-inspired hashing',
        'requires_training': False,
        'gpu_required': False,
    },
    'softhebb': {
        'config': 'configs/softhebb.yaml',
        'description': 'SoftHebb - Soft Hebbian learning',
        'requires_training': True,
        'gpu_required': False,
    },
    'krotov': {
        'config': 'configs/krotov.yaml',
        'description': 'Krotov-Hopfield - Competing hidden units',
        'requires_training': True,
        'gpu_required': False,
    },
    'biohash': {
        'config': 'configs/biohash.yaml',
        'description': 'BioHash - Hebbian-learned sparse projections',
        'requires_training': True,
        'gpu_required': False,
    },
    'wta_hash': {
        'config': 'configs/wta_hash.yaml',
        'description': 'WTA Hash - Winner-Take-All hashing',
        'requires_training': False,
        'gpu_required': False,
    },
    'som': {
        'config': 'configs/som.yaml',
        'description': 'SOM - Self-Organizing Map (Kohonen)',
        'requires_training': True,
        'gpu_required': False,
    },
    'lsh': {
        'config': 'configs/lsh.yaml',
        'description': 'LSH/SimHash - Random hyperplane hashing',
        'requires_training': False,
        'gpu_required': False,
    },
    'diehl_cook': {
        'config': 'configs/diehl_cook.yaml',
        'description': 'Diehl & Cook - STDP-based SNN (requires BindsNET)',
        'requires_training': True,
        'gpu_required': False,  # Optional, but recommended
    },
    'flyhash_sift1m': {
        'config': 'configs/flyhash_sift1m.yaml',
        'description': 'FlyHash on SIFT-1M dataset',
        'requires_training': False,
        'gpu_required': False,
    },
}


def print_header(text: str):
    """Print formatted header."""
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)
    print()


def print_section(text: str):
    """Print formatted section."""
    print()
    print(f"--- {text} ---")
    print()


def test_baseline(
    baseline: str,
    dataset: str = 'mnist',
    seed: int = 0,
    force: bool = False,
    verbose: bool = True
) -> bool:
    """
    Test a single baseline.
    
    Args:
        baseline: Baseline name
        dataset: Dataset name
        seed: Random seed
        force: Force re-encoding
        verbose: Print detailed output
    
    Returns:
        True if test passed, False otherwise
    """
    if baseline not in AVAILABLE_BASELINES:
        print(f"ERROR: Unknown baseline '{baseline}'")
        print(f"Available: {', '.join(AVAILABLE_BASELINES.keys())}")
        return False
    
    info = AVAILABLE_BASELINES[baseline]
    
    if verbose:
        print(f"Testing: {baseline}")
        print(f"  Description: {info['description']}")
        print(f"  Dataset: {dataset}")
        print(f"  Seed: {seed}")
    
    # Check if config exists
    config_path = Path(info['config'])
    if not config_path.exists():
        print(f"  ERROR: Config not found: {config_path}")
        return False
    
    # Build command
    cmd = [
        'python', 'scripts/run_baseline.py',
        '--config', info['config'],
        '--seed', str(seed),
        '--dataset', dataset,
    ]
    
    if force:
        cmd.append('--force')
    
    if verbose:
        print(f"  Command: {' '.join(cmd)}")
    
    # Run test
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=False
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            if verbose:
                print(f"  ✓ SUCCESS ({elapsed:.1f}s)")
            return True
        else:
            print(f"  ✗ FAILED ({elapsed:.1f}s)")
            if not verbose and result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
    
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def list_baselines():
    """List all available baselines."""
    print_header("Available Baselines")
    
    for name, info in AVAILABLE_BASELINES.items():
        status = "✓" if Path(info['config']).exists() else "✗"
        training = "Needs training" if info['requires_training'] else "No training"
        gpu = "GPU recommended" if info['gpu_required'] else "CPU ok"
        
        print(f"{status} {name:15s} - {info['description']}")
        print(f"   {'':15s}   ({training}, {gpu})")
        print(f"   {'':15s}   Config: {info['config']}")
        print()


def test_multiple_baselines(
    baselines: List[str],
    dataset: str = 'mnist',
    seeds: List[int] = [0],
    force: bool = False,
    verbose: bool = True
) -> dict:
    """
    Test multiple baselines.
    
    Args:
        baselines: List of baseline names
        dataset: Dataset name
        seeds: List of random seeds
        force: Force re-encoding
        verbose: Print detailed output
    
    Returns:
        Dictionary with test results
    """
    print_header(f"Testing {len(baselines)} Baseline(s) on {dataset}")
    
    results = {}
    total_tests = len(baselines) * len(seeds)
    current_test = 0
    
    for baseline in baselines:
        if baseline not in AVAILABLE_BASELINES:
            print(f"⚠ Skipping unknown baseline: {baseline}")
            continue
        
        results[baseline] = {}
        
        print_section(f"Baseline: {baseline}")
        
        for seed in seeds:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Testing {baseline} with seed={seed}")
            
            success = test_baseline(
                baseline=baseline,
                dataset=dataset,
                seed=seed,
                force=force,
                verbose=verbose
            )
            
            results[baseline][seed] = success
        
        # Summary for this baseline
        successes = sum(results[baseline].values())
        total = len(results[baseline])
        print(f"\n{baseline} Summary: {successes}/{total} passed")
    
    return results


def print_summary(results: dict):
    """Print test summary."""
    print_header("Test Summary")
    
    total_passed = 0
    total_failed = 0
    
    for baseline, seed_results in results.items():
        passed = sum(seed_results.values())
        failed = len(seed_results) - passed
        total_passed += passed
        total_failed += failed
        
        status = "✓" if failed == 0 else "✗"
        print(f"{status} {baseline:15s}: {passed}/{len(seed_results)} passed")
    
    print()
    print(f"Total: {total_passed + total_failed} tests")
    print(f"✓ Passed: {total_passed}")
    print(f"✗ Failed: {total_failed}")
    print()
    
    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='Test single or multiple baselines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single baseline
  python scripts/test_baseline.py flyhash
  
  # Test multiple baselines
  python scripts/test_baseline.py flyhash diehl_cook
  
  # Test with multiple seeds
  python scripts/test_baseline.py flyhash --seeds 0 1 2
  
  # Test all available baselines
  python scripts/test_baseline.py --all
  
  # Test on different dataset
  python scripts/test_baseline.py flyhash --dataset fashion_mnist
  
  # Force re-encoding (ignore cache)
  python scripts/test_baseline.py flyhash --force
  
  # Quiet mode (less output)
  python scripts/test_baseline.py flyhash --quiet
        """
    )
    
    parser.add_argument(
        'baselines',
        nargs='*',
        help='Baseline name(s) to test'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all available baselines'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available baselines and exit'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='Dataset to use (default: mnist)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0],
        help='Random seeds to test (default: 0)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-encoding (ignore cached features)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode (less output)'
    )
    
    args = parser.parse_args()
    
    # List baselines and exit
    if args.list:
        list_baselines()
        return 0
    
    # Determine which baselines to test
    if args.all:
        baselines = list(AVAILABLE_BASELINES.keys())
    elif args.baselines:
        baselines = args.baselines
    else:
        print("ERROR: Please specify baseline(s) or use --all")
        print("Run with --list to see available baselines")
        parser.print_help()
        return 1
    
    # Filter to only existing baselines
    baselines = [b for b in baselines if b in AVAILABLE_BASELINES]
    
    if not baselines:
        print("ERROR: No valid baselines specified")
        print("Run with --list to see available baselines")
        return 1
    
    # Run tests
    verbose = not args.quiet
    
    results = test_multiple_baselines(
        baselines=baselines,
        dataset=args.dataset,
        seeds=args.seeds,
        force=args.force,
        verbose=verbose
    )
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
