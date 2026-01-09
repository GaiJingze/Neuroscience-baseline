"""
Clustering/Hashing Pipeline for SNN Baseline Evaluation
"""

__version__ = "0.1.0"

from .datasets import load_dataset, MNISTDataset, FashionMNISTDataset, SIFT1MDataset
from .metrics import (
    compute_nmi, 
    compute_ari, 
    compute_clustering_accuracy,
    compute_map,
    compute_recall_at_k
)
from .clustering import run_clustering_evaluation
from .retrieval import run_retrieval_evaluation
from .binarization import top_k_binarization, threshold_binarization, wta_binarization
from .utils import set_seed, save_results, load_results, Logger

__all__ = [
    'load_dataset',
    'MNISTDataset',
    'FashionMNISTDataset', 
    'SIFT1MDataset',
    'compute_nmi',
    'compute_ari',
    'compute_clustering_accuracy',
    'compute_map',
    'compute_recall_at_k',
    'run_clustering_evaluation',
    'run_retrieval_evaluation',
    'top_k_binarization',
    'threshold_binarization',
    'wta_binarization',
    'set_seed',
    'save_results',
    'load_results',
    'Logger',
]
