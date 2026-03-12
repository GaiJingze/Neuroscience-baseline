#!/usr/bin/env python
"""
Main entry point for the clustering/hashing pipeline.

This is the primary interface for running baselines and evaluating results.

Usage:
    # Run a baseline
    python run.py --baseline flyhash --dataset mnist --seed 0
    
    # Test mode (quick validation)
    python run.py --test
    
    # List available baselines
    python run.py --list
    
    # Run with config file
    python run.py --config configs/flyhash.yaml
"""

import sys
import argparse
from pathlib import Path

# Ensure scripts directory is in path
scripts_dir = Path(__file__).parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

# Import from scripts
from scripts.run_baseline import main as run_baseline_main
from scripts.test_baseline import list_baselines, AVAILABLE_BASELINES
from pipeline.utils import setup_logging


def print_welcome():
    """Print welcome message."""
    print()
    print("="*70)
    print("  Clustering/Hashing Pipeline")
    print("  Bio-inspired SNN Baseline Evaluation")
    print("="*70)
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Clustering/Hashing Pipeline - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a baseline
  python run.py --baseline flyhash --dataset mnist --seed 0
  
  # Run with config file
  python run.py --config configs/flyhash.yaml
  
  # Quick test mode
  python run.py --test
  
  # List available baselines
  python run.py --list
  
  # Get help for specific modes
  python run.py --help-test      # Testing help
  python run.py --help-config    # Config file help
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--list',
        action='store_true',
        help='List available baselines and exit'
    )
    mode_group.add_argument(
        '--test',
        action='store_true',
        help='Run quick test mode'
    )
    mode_group.add_argument(
        '--help-test',
        action='store_true',
        help='Show testing help and exit'
    )
    mode_group.add_argument(
        '--help-config',
        action='store_true',
        help='Show config file help and exit'
    )
    
    # Config file or baseline
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to config YAML file'
    )
    config_group.add_argument(
        '--baseline',
        type=str,
        choices=list(AVAILABLE_BASELINES.keys()),
        help='Baseline name (alternative to --config)'
    )
    
    # Parameters (used with --baseline)
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='Dataset name (default: mnist)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed (default: 0)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-encoding (ignore cache)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./outputs',
        help='Root directory for log files (default: ./outputs)'
    )
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.list:
        print_welcome()
        list_baselines()
        return 0
    
    if args.help_test:
        print_welcome()
        print("Testing Help")
        print("="*70)
        print()
        print("To test baselines, use the testing scripts:")
        print()
        print("  # Test single baseline")
        print("  python scripts/test_baseline.py flyhash")
        print()
        print("  # Test multiple baselines")
        print("  python scripts/test_baseline.py flyhash softhebb biohash")
        print()
        print("  # Batch test")
        print("  bash scripts/batch_test.sh --quick")
        print()
        print("  # Or use Makefile")
        print("  make test-baseline BASELINE=flyhash")
        print("  make batch-test-quick")
        print()
        print("For more details, see: docs/TESTING_SUMMARY.md")
        return 0
    
    if args.help_config:
        print_welcome()
        print("Config File Help")
        print("="*70)
        print()
        print("Config files are YAML files in configs/ directory.")
        print()
        print("Example: configs/flyhash.yaml")
        print("```yaml")
        print("experiment_name: flyhash_mnist")
        print("dataset: mnist")
        print("encoder: flyhash")
        print("encoder_config:")
        print("  input_dim: 784")
        print("  projection_dim: 2000")
        print("```")
        print()
        print("Available config files:")
        config_dir = Path('configs')
        if config_dir.exists():
            for config_file in sorted(config_dir.glob('*.yaml')):
                print(f"  - {config_file}")
        print()
        print("For more details, see: docs/clustering_hashing_baseline_guide.md")
        return 0
    
    if args.test:
        print_welcome()
        print("Running quick test...")
        print()
        import subprocess
        result = subprocess.run(
            ['python', 'scripts/quick_test.py'],
            check=False
        )
        return result.returncode
    
    # Standard execution mode
    if not args.config and not args.baseline:
        print("ERROR: Please specify --config or --baseline")
        print("Run 'python run.py --list' to see available baselines")
        parser.print_help()
        return 1
    
    # Build arguments for run_baseline.py
    run_args = []
    
    if args.config:
        run_args.extend(['--config', args.config])
    elif args.baseline:
        # Use baseline name to construct config path
        config_path = f'configs/{args.baseline}.yaml'
        if not Path(config_path).exists():
            print(f"ERROR: Config file not found: {config_path}")
            return 1
        run_args.extend(['--config', config_path])
    
    # Add optional parameters
    if args.dataset != 'mnist':
        run_args.extend(['--dataset', args.dataset])
    
    if args.seed != 0:
        run_args.extend(['--seed', str(args.seed)])
    
    if args.force:
        run_args.append('--force')
    
    # Setup logging for the main entry point
    baseline_name = args.baseline or Path(args.config).stem
    logger = setup_logging(
        experiment_name=f'run_{baseline_name}',
        output_dir=args.log_dir,
        seed=args.seed,
    )

    # Print execution info
    print_welcome()
    logger.info(f"Running: scripts/run_baseline.py {' '.join(run_args)}")

    # Call the original run_baseline script
    sys.argv = ['run_baseline.py'] + run_args

    try:
        result = run_baseline_main(argparse.Namespace(**{
            'config': args.config if args.config else f'configs/{args.baseline}.yaml',
            'seed': args.seed,
            'dataset': args.dataset if args.baseline else None,
            'force': args.force
        }))
        logger.info("Pipeline finished successfully.")
        if logger.get_log_file():
            logger.info(f"Full log saved to: {logger.get_log_file()}")
        return result
    except Exception as e:
        logger.error(f"{e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
