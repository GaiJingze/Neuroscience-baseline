#!/usr/bin/env python
"""
Clear cached features for baselines.

Usage:
    # Clear all cache
    python scripts/clear_cache.py --all
    
    # Clear cache for specific baseline
    python scripts/clear_cache.py --baseline flyhash
    
    # Clear cache for specific baseline and dataset
    python scripts/clear_cache.py --baseline flyhash --dataset mnist
    
    # Dry run (show what would be deleted)
    python scripts/clear_cache.py --all --dry-run
"""

import argparse
import shutil
from pathlib import Path


def clear_cache(baseline=None, dataset=None, output_dir='./outputs', dry_run=False):
    """Clear cache directories."""
    
    codes_dir = Path(output_dir) / 'codes'
    
    if not codes_dir.exists():
        print(f"No cache directory found at {codes_dir}")
        return
    
    # Determine what to delete
    if baseline is None:
        # Clear all
        targets = [codes_dir]
        desc = "all cache"
    elif dataset is None:
        # Clear specific baseline
        targets = [codes_dir / baseline]
        desc = f"cache for baseline '{baseline}'"
    else:
        # Clear specific baseline + dataset
        targets = [codes_dir / baseline / dataset]
        desc = f"cache for baseline '{baseline}' on dataset '{dataset}'"
    
    # Calculate size and count files
    total_size = 0
    total_files = 0
    
    for target in targets:
        if target.exists():
            for file in target.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
                    total_files += 1
    
    # Show what will be deleted
    print("=" * 70)
    print(f"Clear Cache: {desc}")
    print("=" * 70)
    
    for target in targets:
        if target.exists():
            print(f"\nTarget: {target}")
            print(f"  Files: {total_files}")
            print(f"  Size: {total_size / 1024 / 1024:.2f} MB")
    
    if not any(t.exists() for t in targets):
        print("\n⚠️  No cache found to clear")
        return
    
    if dry_run:
        print("\n🔍 Dry run - no files were deleted")
        return
    
    # Confirm deletion
    response = input("\n⚠️  Delete cache? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    # Delete
    deleted_count = 0
    for target in targets:
        if target.exists():
            print(f"\nDeleting: {target}")
            shutil.rmtree(target)
            deleted_count += 1
    
    print(f"\n✅ Cleared {deleted_count} cache director{'ies' if deleted_count != 1 else 'y'}")
    print(f"   Freed {total_size / 1024 / 1024:.2f} MB")


def list_cache(output_dir='./outputs'):
    """List all cached data."""
    
    codes_dir = Path(output_dir) / 'codes'
    
    if not codes_dir.exists():
        print(f"No cache directory found at {codes_dir}")
        return
    
    print("=" * 70)
    print("Cached Data")
    print("=" * 70)
    
    total_size = 0
    total_files = 0
    
    for baseline_dir in sorted(codes_dir.iterdir()):
        if baseline_dir.is_dir():
            print(f"\n📦 {baseline_dir.name}")
            
            for dataset_dir in sorted(baseline_dir.iterdir()):
                if dataset_dir.is_dir():
                    # Count files and size
                    files = list(dataset_dir.rglob('*.npy'))
                    size = sum(f.stat().st_size for f in files)
                    
                    total_size += size
                    total_files += len(files)
                    
                    print(f"  └─ {dataset_dir.name}: {len(files)} files, {size / 1024 / 1024:.2f} MB")
    
    print(f"\n{'=' * 70}")
    print(f"Total: {total_files} files, {total_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description='Clear cached features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all cache
  python scripts/clear_cache.py --list
  
  # Clear all cache
  python scripts/clear_cache.py --all
  
  # Clear cache for FlyHash
  python scripts/clear_cache.py --baseline flyhash
  
  # Clear cache for FlyHash on MNIST
  python scripts/clear_cache.py --baseline flyhash --dataset mnist
  
  # Dry run (preview deletion)
  python scripts/clear_cache.py --all --dry-run
"""
    )
    
    parser.add_argument('--all', action='store_true', help='Clear all cache')
    parser.add_argument('--baseline', type=str, help='Baseline name (e.g., flyhash)')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., mnist)')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    parser.add_argument('--list', action='store_true', help='List all cached data')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if args.list:
        list_cache(args.output_dir)
        return
    
    if not args.all and not args.baseline:
        parser.print_help()
        print("\n❌ Error: Specify --all, --baseline, or --list")
        return
    
    # Override confirmation if --yes flag is set
    if args.yes:
        import builtins
        original_input = builtins.input
        builtins.input = lambda prompt: 'y'
    
    try:
        clear_cache(
            baseline=args.baseline if not args.all else None,
            dataset=args.dataset,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )
    finally:
        if args.yes:
            builtins.input = original_input


if __name__ == '__main__':
    main()
