#!/usr/bin/env python
"""
Run a single baseline encoder and evaluate on clustering/retrieval tasks.

Usage:
    python scripts/run_baseline.py --config configs/flyhash.yaml
    python scripts/run_baseline.py --config configs/biohash.yaml --seed 1
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
from baselines.softhebb.encoder import SoftHebbEncoder
from baselines.krotov.encoder import KrotovEncoder
from baselines.biohash.encoder import BioHashEncoder
from baselines.wta_hash.encoder import WTAHashEncoder
from baselines.som.encoder import SOMEncoder
from baselines.lsh.encoder import SimHashEncoder

# BindsNET-based encoders (optional)
try:
    from baselines.diehl_cook.encoder import DiehlCookEncoder
    DIEHL_COOK_AVAILABLE = True
except ImportError:
    DIEHL_COOK_AVAILABLE = False
    DiehlCookEncoder = None

try:
    from baselines.lm_snn.encoder import LMSNNEncoder
    LM_SNN_AVAILABLE = True
except ImportError:
    LM_SNN_AVAILABLE = False
    LMSNNEncoder = None

try:
    from baselines.lc_snn.encoder import LCSNNEncoder
    LC_SNN_AVAILABLE = True
except ImportError:
    LC_SNN_AVAILABLE = False
    LCSNNEncoder = None

try:
    from baselines.deep_stdp.encoder import DeepSTDPEncoder
    DEEP_STDP_AVAILABLE = True
except ImportError:
    DEEP_STDP_AVAILABLE = False
    DeepSTDPEncoder = None


def get_encoder(name: str, config: dict):
    """Get encoder instance by name."""
    encoders = {
        'dummy': DummyEncoder,
        'flyhash': FlyHashEncoder,
        'softhebb': SoftHebbEncoder,
        'krotov': KrotovEncoder,
        'biohash': BioHashEncoder,
        'wta_hash': WTAHashEncoder,
        'som': SOMEncoder,
        'lsh': SimHashEncoder,
    }
    # Add BindsNET-based encoders if available
    if DIEHL_COOK_AVAILABLE and DiehlCookEncoder is not None:
        encoders['diehl_cook'] = DiehlCookEncoder
    if LM_SNN_AVAILABLE and LMSNNEncoder is not None:
        encoders['lm_snn'] = LMSNNEncoder
    if LC_SNN_AVAILABLE and LCSNNEncoder is not None:
        encoders['lc_snn'] = LCSNNEncoder
    if DEEP_STDP_AVAILABLE and DeepSTDPEncoder is not None:
        encoders['deep_stdp'] = DeepSTDPEncoder

    if name not in encoders:
        available = list(encoders.keys())
        bindsnet_names = []
        if not DIEHL_COOK_AVAILABLE:
            bindsnet_names.append('diehl_cook')
        if not LM_SNN_AVAILABLE:
            bindsnet_names.append('lm_snn')
        if not LC_SNN_AVAILABLE:
            bindsnet_names.append('lc_snn')
        if not DEEP_STDP_AVAILABLE:
            bindsnet_names.append('deep_stdp')
        if bindsnet_names:
            available.append(f"{', '.join(bindsnet_names)} (requires BindsNET)")
        raise ValueError(f"Unknown encoder: {name}. Available: {available}")
    return encoders[name](config)


def _apply_binarization(pre_code: np.ndarray, config: dict) -> np.ndarray:
    """Apply pipeline-level binarization if configured.

    Returns binarized code, or None if no binarization is configured.
    """
    method = config.get('binarization_method')
    if not method or method == 'none':
        return None
    params = config.get('binarization_params', {})
    if method == 'top_k':
        k = params.get('k')
        if k is None:
            raise ValueError("binarization_method='top_k' requires binarization_params.k")
        return top_k_binarization(pre_code, k)
    elif method == 'top_k_percent':
        percent = params.get('percent')
        if percent is None:
            raise ValueError("binarization_method='top_k_percent' requires binarization_params.percent")
        return top_k_percent_binarization(pre_code, percent)
    else:
        raise ValueError(f"Unknown binarization_method: {method}")


def preprocess_data(data: np.ndarray, dataset_name: str, encoder_name: str) -> np.ndarray:
    """
    Preprocess data based on dataset and encoder requirements.

    - GloVe vectors contain negative values.
    - Krotov can handle negatives (via sign * |W|^(p-1)) but benefits from normalization.
    - SoftHebb uses ReLU which clips negatives — needs [0, 1] normalization.
    - For other encoders, GloVe is L2-normalized.

    Args:
        data: Raw data array [n_samples, dim]
        dataset_name: Name of the dataset
        encoder_name: Name of the encoder

    Returns:
        Preprocessed data array
    """
    if dataset_name != 'glove':
        return data

    # GloVe-specific normalization
    if encoder_name == 'softhebb':
        # MinMax normalization to [0, 1] per feature (SoftHebb uses ReLU)
        feat_min = data.min(axis=0, keepdims=True)
        feat_max = data.max(axis=0, keepdims=True)
        denom = feat_max - feat_min
        denom[denom == 0] = 1
        data = (data - feat_min) / denom
        print(f"  [Preprocess] GloVe MinMax normalized to [0, 1] for SoftHebb")

    elif encoder_name == 'krotov':
        # L2 normalize + shift to non-negative for Krotov
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        data = data / norms
        # Shift to [0, 1]: (x + 1) / 2 since L2-normalized cosine range is [-1, 1]
        data = (data + 1.0) / 2.0
        print(f"  [Preprocess] GloVe L2-normalized + shifted to [0,1] for Krotov")

    else:
        # Default: L2 normalization for other encoders
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        data = data / norms
        print(f"  [Preprocess] GloVe L2-normalized")

    return data.astype(np.float32)


def build_label_groundtruth(query_labels: np.ndarray, db_labels: np.ndarray,
                            k: int = 100) -> np.ndarray:
    """
    Build retrieval ground truth based on label matching.

    For each query, the relevant documents are those in the database with the same label.
    We randomly sample up to k relevant items per query.

    Args:
        query_labels: Labels for query samples
        db_labels: Labels for database samples
        k: Maximum number of ground truth neighbors per query

    Returns:
        [n_queries, k] array of ground truth indices
    """
    n_queries = len(query_labels)
    groundtruth = np.zeros((n_queries, k), dtype=np.int32)

    for i in range(n_queries):
        # Find all database items with the same label
        same_label_idx = np.where(db_labels == query_labels[i])[0]
        if len(same_label_idx) > k:
            same_label_idx = np.random.choice(same_label_idx, k, replace=False)
        elif len(same_label_idx) < k:
            # Pad with the available indices (repeat if necessary)
            padding = np.random.choice(same_label_idx, k - len(same_label_idx), replace=True)
            same_label_idx = np.concatenate([same_label_idx, padding])
        groundtruth[i] = same_label_idx

    return groundtruth


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
    
    dataset_config = config.get('dataset_config', {})
    dataset = load_dataset(
        name=config['dataset'],
        root=config.get('data_root', './data'),
        **dataset_config
    )
    logger.log(f"  Train samples: {len(dataset['train_data'])}")
    logger.log(f"  Test samples: {len(dataset['test_data'])}")

    # ---- Preprocessing (GloVe normalization for Krotov / SoftHebb) ----
    dataset_name = config['dataset'].lower()
    encoder_name = config['encoder'].lower()

    dataset['train_data'] = preprocess_data(dataset['train_data'], dataset_name, encoder_name)
    dataset['test_data'] = preprocess_data(dataset['test_data'], dataset_name, encoder_name)
    if dataset.get('query_data') is not None:
        dataset['query_data'] = preprocess_data(dataset['query_data'], dataset_name, encoder_name)

    # ---- Auto-detect evaluation modes ----
    has_labels = dataset.get('test_labels') is not None
    has_query_data = dataset.get('query_data') is not None
    has_groundtruth = dataset.get('groundtruth') is not None

    # Clustering: only if labels exist
    do_clustering = has_labels and config.get('eval_clustering', True)
    # Retrieval: always attempt (build groundtruth from labels if needed)
    do_retrieval = config.get('eval_retrieval', True)

    logger.log(f"\n  Eval modes: clustering={do_clustering}, retrieval={do_retrieval}")
    logger.log(f"  Has labels: {has_labels}, Has query_data: {has_query_data}, Has groundtruth: {has_groundtruth}")
    
    # ---- Auto-set input_dim and device from dataset / top-level config ----
    actual_dim = dataset['train_data'].shape[1]
    if config['encoder_config'].get('input_dim') and config['encoder_config']['input_dim'] != actual_dim:
        logger.log(f"  WARNING: encoder_config.input_dim={config['encoder_config']['input_dim']} "
                    f"!= dataset dim={actual_dim}. Overriding to {actual_dim}.")
    config['encoder_config']['input_dim'] = actual_dim

    if config.get('device') and 'device' not in config['encoder_config']:
        config['encoder_config']['device'] = config['device']

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
        
        # Apply additional binarization if configured
        binarized = _apply_binarization(pre_code, config)
        if binarized is not None:
            logger.log(f"  Applying {config['binarization_method']} binarization...")
            code = binarized
        
        # Save codes
        if config.get('save_codes', True):
            np.save(pre_code_file, pre_code)
            np.save(code_file, code)
            logger.log(f"  Codes saved to {codes_dir}")
    
    logger.log(f"  Pre-code shape: {pre_code.shape}")
    logger.log(f"  Code shape: {code.shape}")
    logger.log(f"  Code sparsity: {1 - np.mean(code):.3f}")
    
    # ========== Evaluation ==========
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
    
    # ---- Clustering evaluation (only if labels available) ----
    if do_clustering:
        logger.log("\n  Running clustering evaluation (ACC / NMI)...")
        
        clustering_results = run_clustering_evaluation(
            codes=code,
            labels_true=dataset['test_labels'],
            n_clusters=config.get('n_clusters', 10),
            methods=config.get('clustering_methods', ['kmeans']),
            is_binary=True,
            verbose=False
        )
        results['clustering'] = clustering_results
        
        logger.log("    Clustering results:")
        for method, metrics in clustering_results.items():
            logger.log(f"      {method}:")
            logger.log(f"        NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, ACC={metrics['acc']:.4f}")
    
    # ---- Retrieval evaluation ----
    if do_retrieval:
        logger.log("\n  Running retrieval evaluation (mAP / Recall@K)...")

        k_values = config.get('retrieval_k_values', [1, 10, 50, 100])

        if has_query_data and has_groundtruth:
            # Explicit query data + groundtruth (SIFT1M, GloVe)
            logger.log("    Mode: explicit query_data + groundtruth")
            query_encoded = encoder.encode(dataset['query_data'])
            query_pre = query_encoded['pre_code']
            query_code = query_encoded['code']
            # Apply the same pipeline binarization to query codes for consistency
            binarized = _apply_binarization(query_pre, config)
            if binarized is not None:
                query_code = binarized
            database_code = code
            groundtruth = dataset['groundtruth']

        elif has_labels:
            # Label-based retrieval (MNIST, Fashion-MNIST)
            # Database = encoded train_data, Queries = encoded test_data
            logger.log("    Mode: label-based retrieval (database=train, queries=test)")
            logger.log("    Encoding train_data as retrieval database...")
            train_encoded = encoder.encode(dataset['train_data'])
            train_pre = train_encoded['pre_code']
            database_code = train_encoded['code']
            # Apply the same pipeline binarization to database codes for consistency
            binarized = _apply_binarization(train_pre, config)
            if binarized is not None:
                database_code = binarized
            query_code = code  # test_data already encoded + binarized above
            # Build groundtruth from labels
            groundtruth = build_label_groundtruth(
                dataset['test_labels'], dataset['train_labels'],
                k=max(k_values)
            )

        else:
            logger.log("    Warning: No query/groundtruth data and no labels. Skipping retrieval.")
            do_retrieval = False

        if do_retrieval:
            retrieval_results = run_retrieval_evaluation(
                query_codes=query_code,
                database_codes=database_code,
                groundtruth=groundtruth,
                k_values=k_values,
                metric='hamming',
                verbose=False
            )
            results['retrieval'] = retrieval_results
            
            logger.log("    Retrieval results:")
            for metric_name, value in retrieval_results.items():
                logger.log(f"      {metric_name}: {value:.4f}")
    
    # Save results
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
