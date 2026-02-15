#!/usr/bin/env python3
"""
run_benchmark.py — Unified benchmark runner for Neuroscience-baseline.

Runs every baseline encoder on every specified dataset with multiple seeds,
saves results in a structured directory layout, and prints a summary table
at the end.

Directory layout produced
─────────────────────────
  outputs/
  └── benchmark/
      ├── mnist/
      │   ├── flyhash/
      │   │   ├── seed0.json
      │   │   ├── seed1.json
      │   │   └── seed2.json
      │   ├── krotov/
      │   │   └── ...
      │   └── ...
      ├── fashion_mnist/
      │   └── ...
      ├── sift1m/
      │   └── ...
      └── summary.json          # aggregated mean±std for every combo

Usage
─────
  # Full benchmark (3 datasets × 8 baselines × 3 seeds)
  python scripts/run_benchmark.py

  # Quick smoke-test (MNIST only, seed 0)
  python scripts/run_benchmark.py --quick

  # Pick baselines / datasets
  python scripts/run_benchmark.py --baselines flyhash krotov --datasets mnist fashion_mnist

  # Specific seeds
  python scripts/run_benchmark.py --seeds 0 1 2 3 4
"""

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so imports work even when called as
#   python scripts/run_benchmark.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets import load_dataset
from pipeline.clustering import run_clustering_evaluation
from pipeline.retrieval import run_retrieval_evaluation
from pipeline.binarization import top_k_binarization, top_k_percent_binarization
from pipeline.utils import set_seed, save_results, Logger

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Baselines that always work (pure-Python / NumPy / PyTorch)
CORE_BASELINES = [
    "flyhash",
    "biohash",
    "krotov",
    "softhebb",
    "som",
    "wta_hash",
    "lsh",
]

# Baselines needing optional deps
OPTIONAL_BASELINES = {
    "diehl_cook": "bindsnet",
}

# Datasets that support clustering (have labels)
CLUSTERING_DATASETS = {"mnist", "fashion_mnist"}
# All supported datasets
ALL_DATASETS = ["mnist", "fashion_mnist", "sift1m"]

# Default evaluation parameters
DEFAULT_N_CLUSTERS = 10
DEFAULT_K_VALUES = [1, 10, 50, 100]
DEFAULT_CLUSTERING_METHODS = ["kmeans"]


def _available_baselines() -> list:
    """Return list of baselines whose dependencies are installed."""
    available = list(CORE_BASELINES)
    for name, pkg in OPTIONAL_BASELINES.items():
        try:
            __import__(pkg)
            available.append(name)
        except ImportError:
            pass
    return available


def _get_encoder(name: str, config: dict):
    """Lightweight encoder factory — avoids import until needed."""
    if name == "flyhash":
        from baselines.flyhash.encoder import FlyHashEncoder
        return FlyHashEncoder(config)
    elif name == "biohash":
        from baselines.biohash.encoder import BioHashEncoder
        return BioHashEncoder(config)
    elif name == "krotov":
        from baselines.krotov.encoder import KrotovEncoder
        return KrotovEncoder(config)
    elif name == "softhebb":
        from baselines.softhebb.encoder import SoftHebbEncoder
        return SoftHebbEncoder(config)
    elif name == "som":
        from baselines.som.encoder import SOMEncoder
        return SOMEncoder(config)
    elif name == "wta_hash":
        from baselines.wta_hash.encoder import WTAHashEncoder
        return WTAHashEncoder(config)
    elif name == "lsh":
        from baselines.lsh.encoder import SimHashEncoder
        return SimHashEncoder(config)
    elif name == "diehl_cook":
        from baselines.diehl_cook.encoder import DiehlCookEncoder
        return DiehlCookEncoder(config)
    else:
        raise ValueError(f"Unknown baseline: {name}")


def _load_encoder_config(baseline: str) -> dict:
    """Load encoder_config from the canonical YAML."""
    cfg_path = REPO_ROOT / "configs" / f"{baseline}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        full = yaml.safe_load(f)
    return full.get("encoder_config", {}), full


def _preprocess(data: np.ndarray, dataset: str, encoder: str) -> np.ndarray:
    """Minimal per-encoder preprocessing (GloVe only)."""
    if dataset not in ("glove",):
        return data.astype(np.float32)
    # GloVe needs normalization
    if encoder == "softhebb":
        lo = data.min(axis=0, keepdims=True)
        hi = data.max(axis=0, keepdims=True)
        denom = hi - lo
        denom[denom == 0] = 1
        return ((data - lo) / denom).astype(np.float32)
    else:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        data = data / norms
        if encoder == "krotov":
            data = (data + 1.0) / 2.0
        return data.astype(np.float32)


def _apply_binarization(pre_code, full_cfg):
    """Apply pipeline-level binarization (if any)."""
    method = full_cfg.get("binarization_method")
    if not method or method == "none":
        return None
    params = full_cfg.get("binarization_params", {})
    if method == "top_k":
        return top_k_binarization(pre_code, params["k"])
    elif method == "top_k_percent":
        return top_k_percent_binarization(pre_code, params["percent"])
    return None


def _build_label_groundtruth(q_labels, db_labels, k=100):
    """Build retrieval ground-truth from labels."""
    n = len(q_labels)
    gt = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        idx = np.where(db_labels == q_labels[i])[0]
        if len(idx) > k:
            idx = np.random.choice(idx, k, replace=False)
        elif len(idx) < k:
            pad = np.random.choice(idx, k - len(idx), replace=True)
            idx = np.concatenate([idx, pad])
        gt[i] = idx[:k]
    return gt


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_single(baseline: str, dataset_name: str, seed: int,
               out_dir: Path, force: bool = False,
               verbose: bool = True) -> dict:
    """Run one (baseline, dataset, seed) experiment.  Returns result dict."""

    result_file = out_dir / dataset_name / baseline / f"seed{seed}.json"
    if result_file.exists() and not force:
        with open(result_file) as f:
            return json.load(f)

    set_seed(seed)

    # --- Load config -------------------------------------------------------
    enc_cfg, full_cfg = _load_encoder_config(baseline)

    # --- Load dataset ------------------------------------------------------
    ds = load_dataset(dataset_name, root=str(REPO_ROOT / "data"))

    # Override input_dim to match data
    enc_cfg["input_dim"] = ds["train_data"].shape[1]
    # Propagate device
    if full_cfg.get("device") and "device" not in enc_cfg:
        enc_cfg["device"] = full_cfg["device"]

    # Preprocess
    ds["train_data"] = _preprocess(ds["train_data"], dataset_name, baseline)
    ds["test_data"] = _preprocess(ds["test_data"], dataset_name, baseline)
    if ds.get("query_data") is not None:
        ds["query_data"] = _preprocess(ds["query_data"], dataset_name, baseline)

    # --- Encoder -----------------------------------------------------------
    encoder = _get_encoder(baseline, enc_cfg)
    if not encoder.is_trained:
        encoder.fit(ds["train_data"], ds.get("train_labels"))
    encoded = encoder.encode(ds["test_data"])
    pre_code = encoded["pre_code"]
    code = encoded["code"]

    binarized = _apply_binarization(pre_code, full_cfg)
    if binarized is not None:
        code = binarized

    # --- Results dict ------------------------------------------------------
    results = {
        "baseline": baseline,
        "dataset": dataset_name,
        "seed": seed,
        "code_dim": int(code.shape[1]),
        "sparsity": float(1 - np.mean(code)),
    }

    # --- Clustering --------------------------------------------------------
    has_labels = ds.get("test_labels") is not None
    if has_labels:
        clust = run_clustering_evaluation(
            codes=code,
            labels_true=ds["test_labels"],
            n_clusters=DEFAULT_N_CLUSTERS,
            methods=DEFAULT_CLUSTERING_METHODS,
            is_binary=True,
            verbose=False,
        )
        results["clustering"] = clust

    # --- Retrieval ---------------------------------------------------------
    has_query = ds.get("query_data") is not None
    has_gt = ds.get("groundtruth") is not None

    if has_query and has_gt:
        q_enc = encoder.encode(ds["query_data"])
        q_code = q_enc["code"]
        b = _apply_binarization(q_enc["pre_code"], full_cfg)
        if b is not None:
            q_code = b
        db_code = code
        gt = ds["groundtruth"]
    elif has_labels:
        tr_enc = encoder.encode(ds["train_data"])
        db_code = tr_enc["code"]
        b = _apply_binarization(tr_enc["pre_code"], full_cfg)
        if b is not None:
            db_code = b
        q_code = code
        gt = _build_label_groundtruth(
            ds["test_labels"], ds["train_labels"], k=max(DEFAULT_K_VALUES)
        )
    else:
        q_code = db_code = gt = None

    if q_code is not None:
        ret = run_retrieval_evaluation(
            query_codes=q_code,
            database_codes=db_code,
            groundtruth=gt,
            k_values=DEFAULT_K_VALUES,
            metric="hamming",
            verbose=False,
        )
        results["retrieval"] = ret

    # --- Persist -----------------------------------------------------------
    result_file.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(result_file))

    return results


# ---------------------------------------------------------------------------
# Aggregate & display
# ---------------------------------------------------------------------------

def aggregate(all_results: list) -> dict:
    """Compute mean±std grouped by (baseline, dataset)."""
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["baseline"], r["dataset"])].append(r)

    summary = {}
    for (bl, ds), runs in sorted(groups.items()):
        entry = {"baseline": bl, "dataset": ds, "n_seeds": len(runs)}

        # Clustering
        if all("clustering" in r for r in runs):
            for method in runs[0]["clustering"]:
                for metric in ("nmi", "ari", "acc"):
                    vals = [r["clustering"][method][metric] for r in runs]
                    entry[f"{method}_{metric}_mean"] = float(np.mean(vals))
                    entry[f"{method}_{metric}_std"] = float(np.std(vals))

        # Retrieval
        if all("retrieval" in r for r in runs):
            for metric_key in runs[0]["retrieval"]:
                vals = [r["retrieval"][metric_key] for r in runs]
                entry[f"{metric_key}_mean"] = float(np.mean(vals))
                entry[f"{metric_key}_std"] = float(np.std(vals))

        entry["sparsity"] = float(np.mean([r["sparsity"] for r in runs]))
        summary[(bl, ds)] = entry

    return summary


def print_table(summary: dict, datasets: list):
    """Print a per-dataset table to stdout."""
    for ds in datasets:
        rows = [(k, v) for k, v in summary.items() if k[1] == ds]
        if not rows:
            continue

        print(f"\n{'='*90}")
        print(f"  Dataset: {ds}")
        print(f"{'='*90}")

        has_clustering = any("kmeans_nmi_mean" in v for _, v in rows)
        has_retrieval = any("map_mean" in v for _, v in rows)

        # Header
        header = f"{'Baseline':<14}"
        if has_clustering:
            header += f"{'NMI':>14}{'ARI':>14}{'ACC':>14}"
        if has_retrieval:
            header += f"{'mAP':>14}{'R@1':>14}{'R@10':>14}"
        header += f"{'Sparsity':>10}"
        print(header)
        print("-" * len(header))

        for (bl, _), v in sorted(rows):
            line = f"{bl:<14}"
            if has_clustering:
                nmi = f"{v.get('kmeans_nmi_mean',0):.4f}±{v.get('kmeans_nmi_std',0):.3f}"
                ari = f"{v.get('kmeans_ari_mean',0):.4f}±{v.get('kmeans_ari_std',0):.3f}"
                acc = f"{v.get('kmeans_acc_mean',0):.4f}±{v.get('kmeans_acc_std',0):.3f}"
                line += f"{nmi:>14}{ari:>14}{acc:>14}"
            if has_retrieval:
                mp = f"{v.get('map_mean',0):.4f}±{v.get('map_std',0):.3f}"
                r1 = f"{v.get('recall@1_mean',0):.4f}±{v.get('recall@1_std',0):.3f}"
                r10 = f"{v.get('recall@10_mean',0):.4f}±{v.get('recall@10_std',0):.3f}"
                line += f"{mp:>14}{r1:>14}{r10:>14}"
            line += f"{v['sparsity']:>10.3f}"
            print(line)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all baselines on multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baselines", nargs="*", default=None,
        help="Baselines to run (default: all available)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to evaluate (default: mnist fashion_mnist sift1m)",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=[0, 1, 2],
        help="Random seeds (default: 0 1 2)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: mnist only, seed 0, skip diehl_cook",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if result JSON already exists",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmark",
        help="Root output directory (default: outputs/benchmark)",
    )
    args = parser.parse_args()

    # Resolve parameters
    available = _available_baselines()
    baselines = args.baselines if args.baselines else list(available)
    datasets = args.datasets if args.datasets else list(ALL_DATASETS)
    seeds = args.seeds

    if args.quick:
        datasets = ["mnist"]
        seeds = [0]
        baselines = [b for b in baselines if b != "diehl_cook"]

    # Filter to actually available
    baselines = [b for b in baselines if b in available]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(baselines) * len(datasets) * len(seeds)
    print(f"\n{'='*70}")
    print(f"  Neuroscience-baseline Benchmark")
    print(f"{'='*70}")
    print(f"  Baselines ({len(baselines)}): {', '.join(baselines)}")
    print(f"  Datasets  ({len(datasets)}):  {', '.join(datasets)}")
    print(f"  Seeds     ({len(seeds)}):     {seeds}")
    print(f"  Total runs: {total}")
    print(f"  Output:     {out_dir}")
    print(f"{'='*70}\n")

    all_results = []
    done = 0
    failed = []
    t0 = time.time()

    for ds in datasets:
        for bl in baselines:
            for seed in seeds:
                done += 1
                tag = f"[{done}/{total}] {bl} / {ds} / seed{seed}"
                print(f"{tag} ...", end=" ", flush=True)
                t1 = time.time()
                try:
                    r = run_single(bl, ds, seed, out_dir,
                                   force=args.force, verbose=False)
                    elapsed = time.time() - t1

                    # one-line summary
                    parts = []
                    if "clustering" in r:
                        nmi = list(r["clustering"].values())[0].get("nmi", 0)
                        acc = list(r["clustering"].values())[0].get("acc", 0)
                        parts.append(f"NMI={nmi:.4f} ACC={acc:.4f}")
                    if "retrieval" in r:
                        parts.append(f"mAP={r['retrieval'].get('map',0):.4f}")
                    print(f"OK  ({elapsed:5.1f}s)  {', '.join(parts)}")

                    all_results.append(r)
                except Exception as e:
                    elapsed = time.time() - t1
                    print(f"FAIL ({elapsed:5.1f}s)  {e}")
                    traceback.print_exc()
                    failed.append(tag)

    wall = time.time() - t0

    # --- Summary -----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Benchmark complete  ({wall:.0f}s wall-clock)")
    print(f"  Passed: {len(all_results)}   Failed: {len(failed)}")
    if failed:
        print(f"\n  Failed runs:")
        for f_ in failed:
            print(f"    - {f_}")
    print(f"{'='*70}")

    if all_results:
        summary = aggregate(all_results)
        print_table(summary, datasets)

        # Persist summary
        summary_file = out_dir / "summary.json"
        # Convert tuple keys to strings for JSON
        json_summary = {f"{bl}__{ds}": v for (bl, ds), v in summary.items()}
        with open(summary_file, "w") as f:
            json.dump(json_summary, f, indent=2)
        print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
