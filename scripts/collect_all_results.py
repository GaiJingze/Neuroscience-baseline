#!/usr/bin/env python3
"""
collect_all_results.py — Collect, aggregate, and display benchmark results.

Reads results from two possible directory layouts:

  New (run_benchmark.py):
    outputs/benchmark/{dataset}/{baseline}/seed{N}.json

  Legacy (run_baseline.py):
    outputs/results/{baseline}_{dataset}_seed{N}.json

Usage:
  python scripts/collect_all_results.py                      # auto-detect
  python scripts/collect_all_results.py --dir outputs/benchmark
  python scripts/collect_all_results.py --latex              # LaTeX table
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# All known baselines (used for parsing legacy filenames)
KNOWN_BASELINES = [
    "diehl_cook", "flyhash", "biohash", "krotov",
    "softhebb", "som", "wta_hash", "lsh",
]


def _collect_new_layout(root: Path) -> list:
    """Collect from outputs/benchmark/{dataset}/{baseline}/seed*.json"""
    results = []
    for jf in sorted(root.rglob("seed*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        # Derive from path: root / dataset / baseline / seedN.json
        baseline = jf.parent.name
        dataset = jf.parent.parent.name
        data.setdefault("baseline", baseline)
        data.setdefault("dataset", dataset)
        results.append(data)
    return results


def _collect_legacy_layout(root: Path) -> list:
    """Collect from outputs/results/{baseline}_{dataset}_seed{N}.json"""
    results = []
    for jf in sorted(root.glob("*_seed*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue

        stem = jf.stem                       # e.g. "flyhash_mnist_seed0"
        parts = stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        prefix = parts[0]                    # "flyhash_mnist"

        # Match known baseline prefix
        baseline = dataset = None
        for bl in sorted(KNOWN_BASELINES, key=len, reverse=True):
            if prefix.startswith(bl + "_"):
                baseline = bl
                dataset = prefix[len(bl) + 1:]
                break
        if baseline is None:
            continue

        data.setdefault("baseline", baseline)
        data.setdefault("dataset", dataset)
        data.setdefault("seed", int(parts[1]))
        results.append(data)
    return results


def collect_results(dirs: list[Path]) -> list:
    """Collect all results from multiple directories."""
    all_results = []
    for d in dirs:
        if not d.exists():
            continue
        # Try new layout first
        new = _collect_new_layout(d)
        if new:
            all_results.extend(new)
        else:
            # Fall back to legacy
            all_results.extend(_collect_legacy_layout(d))
    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list) -> dict:
    """Group by (baseline, dataset), compute mean±std."""
    groups = defaultdict(list)
    for r in results:
        groups[(r["baseline"], r["dataset"])].append(r)

    summary = {}
    for key, runs in sorted(groups.items()):
        bl, ds = key
        entry = {"baseline": bl, "dataset": ds, "n_seeds": len(runs)}

        # Clustering
        if all("clustering" in r for r in runs):
            for method in runs[0]["clustering"]:
                for m in ("nmi", "ari", "acc"):
                    vals = [r["clustering"][method].get(m, 0) for r in runs]
                    entry[f"{method}_{m}_mean"] = float(np.mean(vals))
                    entry[f"{method}_{m}_std"] = float(np.std(vals))

        # Retrieval
        if all("retrieval" in r for r in runs):
            for mk in runs[0]["retrieval"]:
                vals = [r["retrieval"][mk] for r in runs]
                entry[f"{mk}_mean"] = float(np.mean(vals))
                entry[f"{mk}_std"] = float(np.std(vals))

        entry["sparsity"] = float(np.mean([r.get("sparsity", r.get("code_stats", {}).get("sparsity", 0)) for r in runs]))
        summary[key] = entry

    return summary


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _pm(mean, std):
    """±format"""
    return f"{mean:.4f}±{std:.4f}"


def print_summary_table(summary: dict):
    datasets = sorted({ds for _, ds in summary})

    for ds in datasets:
        rows = [(k, v) for k, v in summary.items() if k[1] == ds]
        if not rows:
            continue

        has_clust = any("kmeans_nmi_mean" in v for _, v in rows)
        has_ret = any("map_mean" in v for _, v in rows)

        print(f"\n{'='*100}")
        print(f"  {ds.upper()}  (Mean ± Std)")
        print(f"{'='*100}")

        hdr = f"{'Baseline':<14}{'Seeds':>6}"
        if has_clust:
            hdr += f"{'NMI':>18}{'ARI':>18}{'ACC':>18}"
        if has_ret:
            hdr += f"{'mAP':>18}{'R@1':>18}{'R@10':>18}"
        print(hdr)
        print("-" * len(hdr))

        for (bl, _), v in sorted(rows):
            line = f"{bl:<14}{v['n_seeds']:>6}"
            if has_clust:
                line += f"{_pm(v.get('kmeans_nmi_mean',0), v.get('kmeans_nmi_std',0)):>18}"
                line += f"{_pm(v.get('kmeans_ari_mean',0), v.get('kmeans_ari_std',0)):>18}"
                line += f"{_pm(v.get('kmeans_acc_mean',0), v.get('kmeans_acc_std',0)):>18}"
            if has_ret:
                line += f"{_pm(v.get('map_mean',0), v.get('map_std',0)):>18}"
                line += f"{_pm(v.get('recall@1_mean',0), v.get('recall@1_std',0)):>18}"
                line += f"{_pm(v.get('recall@10_mean',0), v.get('recall@10_std',0)):>18}"
            print(line)
    print()


def print_detailed(summary: dict, all_results: list):
    """Print per-seed breakdown."""
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["baseline"], r["dataset"])].append(r)

    for (bl, ds), runs in sorted(groups.items()):
        runs.sort(key=lambda r: r.get("seed", 0))
        print(f"\n--- {bl} / {ds} ({len(runs)} seeds) ---")
        for r in runs:
            parts = [f"seed={r.get('seed','?')}"]
            if "clustering" in r:
                for method, mv in r["clustering"].items():
                    parts.append(f"{method}: NMI={mv.get('nmi',0):.4f} ARI={mv.get('ari',0):.4f} ACC={mv.get('acc',0):.4f}")
            if "retrieval" in r:
                parts.append(f"mAP={r['retrieval'].get('map',0):.4f}")
            print("  " + "  |  ".join(parts))


def print_latex(summary: dict):
    """Print LaTeX table."""
    datasets = sorted({ds for _, ds in summary})
    baselines = sorted({bl for bl, _ in summary})

    n_ds = len(datasets)
    cols = "l" + "|ccc" * n_ds

    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Clustering Performance (Mean over seeds)}")
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\hline")

    # header row 1
    hdr1 = "\\multirow{2}{*}{Method}"
    for ds in datasets:
        hdr1 += f" & \\multicolumn{{3}}{{c}}{{{ds.replace('_', ' ').title()}}}"
    hdr1 += " \\\\"
    print(hdr1)

    # header row 2
    hdr2 = ""
    for _ in datasets:
        hdr2 += " & NMI & ARI & ACC"
    hdr2 += " \\\\"
    print(hdr2)
    print("\\hline")

    for bl in baselines:
        row = bl.replace("_", "\\_")
        for ds in datasets:
            v = summary.get((bl, ds))
            if v and "kmeans_nmi_mean" in v:
                row += f" & {v['kmeans_nmi_mean']:.4f} & {v['kmeans_ari_mean']:.4f} & {v['kmeans_acc_mean']:.4f}"
            else:
                row += " & - & - & -"
        row += " \\\\"
        print(row)

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def print_tsv(summary: dict):
    """TSV for Excel/Sheets."""
    print("\nBaseline\tDataset\tNMI\tARI\tACC\tmAP\tR@1\tR@10\tSeeds")
    for (bl, ds), v in sorted(summary.items()):
        cols = [bl, ds]
        cols.append(f"{v.get('kmeans_nmi_mean',0):.4f}")
        cols.append(f"{v.get('kmeans_ari_mean',0):.4f}")
        cols.append(f"{v.get('kmeans_acc_mean',0):.4f}")
        cols.append(f"{v.get('map_mean',0):.4f}")
        cols.append(f"{v.get('recall@1_mean',0):.4f}")
        cols.append(f"{v.get('recall@10_mean',0):.4f}")
        cols.append(str(v["n_seeds"]))
        print("\t".join(cols))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect and display benchmark results")
    parser.add_argument("--dir", nargs="*", default=["outputs/benchmark", "outputs/results"],
                        help="Directories to scan (default: outputs/benchmark + outputs/results)")
    parser.add_argument("--detailed", action="store_true", help="Show per-seed breakdown")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    parser.add_argument("--tsv", action="store_true", help="Print TSV for spreadsheets")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dir]
    all_results = collect_results(dirs)

    if not all_results:
        print("No results found.")
        print(f"Searched: {[str(d) for d in dirs]}")
        return 1

    n_baselines = len({r["baseline"] for r in all_results})
    n_datasets = len({r["dataset"] for r in all_results})
    print(f"\nCollected {len(all_results)} result files  "
          f"({n_baselines} baselines, {n_datasets} datasets)")

    summary = aggregate(all_results)

    print_summary_table(summary)

    if args.detailed:
        print_detailed(summary, all_results)
    if args.latex:
        print_latex(summary)
    if args.tsv:
        print_tsv(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
