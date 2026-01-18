#!/usr/bin/env python
"""
Run a single baseline encoder and evaluate on clustering/retrieval tasks.

Usage:
    python scripts/run_baseline.py --config configs/flyhash.yaml
    python scripts/run_baseline.py --config configs/diehl_cook.yaml --seed 1
"""

import argparse
import yaml
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline import (
    load_dataset,
    run_clustering_evaluation,
    run_retrieval_evaluation,
    set_seed,
    save_results,
    Logger
)
from pipeline.binarization import top_k_binarization, top_k_percent_binarization
from baselines.base_encoder import DummyEncoder
from baselines.flyhash.encoder import FlyHashEncoder
from baselines.diehl_cook.encoder import DiehlCookEncoder
from baselines.softhebb.encoder import SoftHebbEncoder


def get_encoder(name: str, config: dict):
    """Get encoder instance by name."""
    if name == 'dummy':
        return DummyEncoder(config)
    elif name == 'flyhash':
        return FlyHashEncoder(config)
    elif name == 'diehl_cook':
        return DiehlCookEncoder(config)
    elif name == 'softhebb':
        return SoftHebbEncoder(config)
    # Add more encoders here
    else:
        raise ValueError(f"Unknown encoder: {name}")


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.seed is not None:
        config['seed'] = args.seed
    if args.dataset is not None:
        config['dataset'] = args.dataset
    
    # Setup
    set_seed(config['seed'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = output_dir / 'logs' / f"{config['experiment_name']}_seed{config['seed']}.log"
    logger = Logger(str(log_file))
    
    logger.log("="*80)
    logger.log(f"Experiment: {config['experiment_name']}")
    logger.log(f"Encoder: {config['encoder']}")
    logger.log(f"Dataset: {config['dataset']}")
    logger.log(f"Seed: {config['seed']}")
    logger.log("="*80)
    
    # Load dataset
    logger.log("\n[1/5] Loading dataset...")
    
    # Extract dataset-specific config
    dataset_config = config.get('dataset_config', {})
    
    dataset = load_dataset(
        name=config['dataset'],
        root=config.get('data_root', './data'),
        **dataset_config
    )
    logger.log(f"  Train samples: {len(dataset['train_data'])}")
    logger.log(f"  Test samples: {len(dataset['test_data'])}")
    
    # Initialize encoder
    logger.log("\n[2/5] Initializing encoder...")
    encoder = get_encoder(
        name=config['encoder'],
        config=config['encoder_config']
    )
    logger.log(f"  Encoder: {encoder}")
    
    # Train encoder
    logger.log("\n[3/5] Training encoder...")
    if not encoder.is_trained:
        encoder.fit(dataset['train_data'], dataset.get('train_labels'))
    else:
        logger.log("  Encoder is non-parametric, skipping training.")
    
    # Encode test data
    logger.log("\n[4/5] Encoding test data...")
    
    # Define output paths
    codes_dir = output_dir / 'codes' / config['encoder'] / config['dataset']
    codes_dir.mkdir(parents=True, exist_ok=True)
    
    pre_code_file = codes_dir / f"pre_code_seed{config['seed']}.npy"
    code_file = codes_dir / f"code_seed{config['seed']}.npy"
    
    if code_file.exists() and not args.force:
        logger.log(f"  Loading cached codes from {codes_dir}")
        pre_code = np.load(pre_code_file)
        code = np.load(code_file)
    else:
        logger.log("  Running encoder...")
        encoded = encoder.encode(dataset['test_data'])
        pre_code = encoded['pre_code']
        code = encoded['code']
        
        # Apply additional binarization if needed
        if config.get('binarization_method') and config['binarization_method'] != 'none':
            logger.log(f"  Applying {config['binarization_method']} binarization...")
            if config['binarization_method'] == 'top_k':
                k = config['binarization_params']['k']
                code = top_k_binarization(pre_code, k)
            elif config['binarization_method'] == 'top_k_percent':
                percent = config['binarization_params']['percent']
                code = top_k_percent_binarization(pre_code, percent)
        
        # Save codes
        if config.get('save_codes', True):
            np.save(pre_code_file, pre_code)
            np.save(code_file, code)
            logger.log(f"  Codes saved to {codes_dir}")
    
    logger.log(f"  Pre-code shape: {pre_code.shape}")
    logger.log(f"  Code shape: {code.shape}")
    logger.log(f"  Code sparsity: {1 - np.mean(code):.3f}")
    
    # Evaluate
    logger.log("\n[5/5] Evaluating...")
    results = {
        'config': config,
        'dataset_info': {
            'name': config['dataset'],
            'n_train': len(dataset['train_data']),
            'n_test': len(dataset['test_data']),
        },
        'code_stats': {
            'code_dim': code.shape[1],
            'sparsity': float(1 - np.mean(code)),
        }
    }
    
    # Clustering evaluation
    if config.get('eval_clustering', False):
        logger.log("\n  Running clustering evaluation...")
        
        if dataset.get('test_labels') is None:
            logger.log("    Warning: No labels available, skipping clustering evaluation")
        else:
            clustering_results = run_clustering_evaluation(
                codes=code,
                labels_true=dataset['test_labels'],
                n_clusters=config['n_clusters'],
                methods=config.get('clustering_methods', ['kmeans']),
                is_binary=True,
                verbose=False
            )
            results['clustering'] = clustering_results
            
            logger.log("    Clustering results:")
            for method, metrics in clustering_results.items():
                logger.log(f"      {method}:")
                logger.log(f"        NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, ACC={metrics['acc']:.4f}")
    
    # Retrieval evaluation
    if config.get('eval_retrieval', False):
        logger.log("\n  Running retrieval evaluation...")
        
        if dataset.get('query_data') is None:
            logger.log("    Warning: No query data available, skipping retrieval evaluation")
        else:
            # Encode query data
            query_encoded = encoder.encode(dataset['query_data'])
            query_code = query_encoded['code']
            
            retrieval_results = run_retrieval_evaluation(
                query_codes=query_code,
                database_codes=code,
                groundtruth=dataset['groundtruth'],
                k_values=config.get('retrieval_k_values', [10, 50, 100]),
                metric='hamming',
                verbose=False
            )
            results['retrieval'] = retrieval_results
            
            logger.log("    Retrieval results:")
            for metric_name, value in retrieval_results.items():
                logger.log(f"      {metric_name}: {value:.4f}")
    
    # Save results
    # Use encoder_dataset format for consistency, instead of experiment_name
    result_filename = f"{config['encoder']}_{config['dataset']}_seed{config['seed']}.json"
    results_file = output_dir / 'results' / result_filename
    save_results(results, str(results_file))
    logger.log(f"\nResults saved to {results_file}")
    
    # Save model
    if config.get('save_model', False):
        model_file = codes_dir / f"model_seed{config['seed']}.pkl"
        encoder.save(str(model_file))
        logger.log(f"Model saved to {model_file}")
    
    logger.log("\n" + "="*80)
    logger.log("Experiment complete!")
    logger.log("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline encoder evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (overrides config)')
    parser.add_argument('--force', action='store_true', help='Force re-encoding (ignore cached codes)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching (always re-encode)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache for this baseline before running')
    
    args = parser.parse_args()
    
    # Handle cache clearing
    if args.clear_cache:
        import shutil
        with open(args.config, 'r') as f:
            config_temp = yaml.safe_load(f)
        cache_dir = Path(config_temp.get('output_dir', './outputs')) / 'codes' / config_temp['encoder']
        if cache_dir.exists():
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            print("✅ Cache cleared")
    
    # Set force flag if no-cache is enabled
    if args.no_cache:
        args.force = True
    
    main(args)
