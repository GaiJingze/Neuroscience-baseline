#!/usr/bin/env python
"""
Run supervised evaluation (SVM classification) on extracted features.

This script follows the original Diehl & Cook (2015) evaluation method:
1. Extract features using unsupervised method (STDP, FlyHash, etc.)
2. Train supervised classifier (Linear SVM) on features
3. Evaluate classification accuracy

Usage:
    python scripts/run_supervised_eval.py --baseline flyhash --dataset mnist
    python scripts/run_supervised_eval.py --baseline diehl_cook --methods linear_svm logistic
"""

import argparse
import yaml
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.datasets import load_dataset
from pipeline.supervised_eval import run_supervised_evaluation, compare_supervised_unsupervised
from pipeline.utils import set_seed, Logger


def load_encoded_features(codes_path):
    """Load cached encoded features."""
    import pickle
    with open(codes_path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Run supervised evaluation (SVM) on baseline features',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--baseline', type=str, required=True,
                       help='Baseline name (e.g., "flyhash", "diehl_cook")')
    parser.add_argument('--dataset', type=str, default='mnist',
                       help='Dataset name')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--methods', nargs='+', 
                       default=['linear_svm'],
                       choices=['linear_svm', 'kernel_svm', 'logistic'],
                       help='Classifier methods to use')
    parser.add_argument('--codes_path', type=str, default=None,
                       help='Path to cached codes (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    logger = Logger()
    
    # Set seed
    set_seed(args.seed)
    
    # Auto-detect codes path if not provided
    if args.codes_path is None:
        codes_path = Path(args.output_dir) / 'codes' / f'{args.baseline}_{args.dataset}_seed{args.seed}.pkl'
    else:
        codes_path = Path(args.codes_path)
    
    if not codes_path.exists():
        logger.error(f"Codes not found: {codes_path}")
        logger.error(f"Please run the baseline first:")
        logger.error(f"  python run.py --baseline {args.baseline} --dataset {args.dataset} --seed {args.seed}")
        sys.exit(1)
    
    logger.log(f"Loading encoded features from: {codes_path}")
    
    # Load features
    try:
        encoded_data = load_encoded_features(codes_path)
        train_features = encoded_data['train_codes']
        test_features = encoded_data['test_codes']
    except KeyError:
        # Alternative format
        train_features = encoded_data['codes']
        test_features = encoded_data.get('test_codes', None)
        if test_features is None:
            logger.error("Test codes not found in cached file")
            sys.exit(1)
    
    # Load labels
    logger.log(f"Loading dataset labels: {args.dataset}")
    dataset = load_dataset(args.dataset, root='./data')
    train_labels = dataset['train_labels']
    test_labels = dataset['test_labels']
    
    # Handle feature shape
    if len(train_features.shape) > 2:
        train_features = train_features.reshape(len(train_features), -1)
    if len(test_features.shape) > 2:
        test_features = test_features.reshape(len(test_features), -1)
    
    logger.log(f"Train features: {train_features.shape}")
    logger.log(f"Test features: {test_features.shape}")
    logger.log(f"Train labels: {train_labels.shape}")
    logger.log(f"Test labels: {test_labels.shape}")
    
    # Run supervised evaluation
    logger.log(f"\nRunning supervised evaluation with methods: {args.methods}")
    results = run_supervised_evaluation(
        train_features, train_labels,
        test_features, test_labels,
        methods=args.methods
    )
    
    # Load clustering results for comparison (if available)
    clustering_results_path = Path(args.output_dir) / 'results' / f'{args.baseline}_{args.dataset}_seed{args.seed}.json'
    if clustering_results_path.exists():
        with open(clustering_results_path, 'r') as f:
            full_results = json.load(f)
            clustering_results = full_results.get('clustering', {})
        
        compare_supervised_unsupervised(clustering_results, results)
    
    # Save results
    output_path = Path(args.output_dir) / 'results' / f'{args.baseline}_{args.dataset}_seed{args.seed}_supervised.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'baseline': args.baseline,
        'dataset': args.dataset,
        'seed': args.seed,
        'methods': args.methods,
        'supervised_results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.log(f"\nResults saved to: {output_path}")
    
    # Print summary
    logger.log("\n" + "="*80)
    logger.log("SUPERVISED EVALUATION SUMMARY")
    logger.log("="*80)
    logger.log(f"Baseline: {args.baseline}")
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"Seed: {args.seed}")
    logger.log("-"*80)
    for method, metrics in results.items():
        logger.log(f"{method}:")
        logger.log(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.log(f"  Precision: {metrics['precision']:.4f}")
        logger.log(f"  Recall: {metrics['recall']:.4f}")
        logger.log(f"  F1: {metrics['f1']:.4f}")
    logger.log("="*80)
    
    # Comparison with Diehl & Cook paper
    if args.baseline == 'diehl_cook' and 'linear_svm' in results:
        paper_accuracy = 0.95
        our_accuracy = results['linear_svm']['accuracy']
        logger.log(f"\n💡 Comparison with Diehl & Cook (2015) paper:")
        logger.log(f"  Paper (full STDP training): ~{paper_accuracy:.2%}")
        logger.log(f"  Ours: {our_accuracy:.2%}")
        if our_accuracy < paper_accuracy * 0.8:
            logger.log(f"  ⚠️  Note: Using skeleton implementation. Full STDP training should improve this.")
        elif our_accuracy >= paper_accuracy * 0.9:
            logger.log(f"  ✅ Excellent! Very close to paper results.")


if __name__ == '__main__':
    main()
