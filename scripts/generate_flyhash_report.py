#!/usr/bin/env python
"""
Generate a comprehensive benchmark report for FlyHash across multiple datasets.

Usage:
    python scripts/generate_flyhash_report.py
    python scripts/generate_flyhash_report.py --datasets mnist fashion_mnist
    python scripts/generate_flyhash_report.py --seeds 0 1 2 3 4
"""

import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(baseline, dataset, seeds):
    """Load results for a baseline/dataset across multiple seeds."""
    results = []
    missing_seeds = []
    
    for seed in seeds:
        result_file = Path(f"outputs/results/{baseline}_{dataset}_seed{seed}.json")
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        else:
            missing_seeds.append(seed)
    
    if missing_seeds:
        print(f"  Warning: Missing seeds for {dataset}: {missing_seeds}")
    
    return results


def compute_stats(results, metric_path):
    """Compute mean and std for a metric across multiple runs."""
    values = []
    for result in results:
        # Navigate nested dict
        value = result
        for key in metric_path:
            if key in value:
                value = value[key]
            else:
                value = None
                break
        if value is not None and not np.isinf(value):
            values.append(value)
    
    if values:
        return np.mean(values), np.std(values), len(values)
    else:
        return None, None, 0


def generate_markdown_report(baseline, datasets, seeds):
    """Generate markdown report."""
    report = []
    report.append(f"# {baseline.upper()} Benchmark Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Seeds**: {seeds}")
    report.append(f"\n**Datasets**: {', '.join([d.upper() for d in datasets])}")
    
    report.append("\n## Results Summary\n")
    
    # Clustering metrics
    clustering_metrics = [
        (['clustering', 'kmeans', 'nmi'], 'NMI', 'Normalized Mutual Information'),
        (['clustering', 'kmeans', 'ari'], 'ARI', 'Adjusted Rand Index'),
        (['clustering', 'kmeans', 'acc'], 'ACC', 'Clustering Accuracy'),
        (['clustering', 'kmeans', 'purity'], 'Purity', 'Cluster Purity'),
        (['clustering', 'kmeans', 'silhouette'], 'Silhouette', 'Silhouette Score'),
    ]
    
    # Code statistics
    code_metrics = [
        (['code_stats', 'sparsity'], 'Sparsity', 'Code Sparsity'),
        (['code_stats', 'code_dim'], 'Code Dim', 'Code Dimension'),
    ]
    
    all_results = {}
    
    # Table for each dataset
    for dataset in datasets:
        results = load_results(baseline, dataset, seeds)
        all_results[dataset] = results
        
        if not results:
            report.append(f"\n### {dataset.upper()} - ❌ No Results Found\n")
            continue
        
        report.append(f"\n### {dataset.upper()}\n")
        report.append(f"**Runs**: {len(results)} / {len(seeds)}\n")
        
        # Clustering metrics table
        report.append("#### Clustering Performance\n")
        report.append("| Metric | Mean | Std | Range |")
        report.append("|--------|------|-----|-------|")
        
        for metric_path, metric_name, _ in clustering_metrics:
            mean, std, n = compute_stats(results, metric_path)
            if mean is not None and n > 0:
                # Get min/max
                values = []
                for r in results:
                    val = r
                    for key in metric_path:
                        val = val.get(key) if isinstance(val, dict) else None
                        if val is None:
                            break
                    if val is not None and not np.isinf(val):
                        values.append(val)
                
                min_val = min(values) if values else 0
                max_val = max(values) if values else 0
                report.append(f"| {metric_name} | {mean:.4f} | {std:.4f} | [{min_val:.4f}, {max_val:.4f}] |")
            else:
                report.append(f"| {metric_name} | N/A | N/A | N/A |")
        
        # Code statistics
        report.append("\n#### Code Statistics\n")
        report.append("| Statistic | Value |")
        report.append("|-----------|-------|")
        
        for metric_path, metric_name, _ in code_metrics:
            mean, std, n = compute_stats(results, metric_path)
            if mean is not None:
                if metric_name == 'Code Dim':
                    report.append(f"| {metric_name} | {int(mean)} |")
                else:
                    report.append(f"| {metric_name} | {mean:.4f} ± {std:.4f} |")
            else:
                report.append(f"| {metric_name} | N/A |")
    
    # Cross-dataset comparison
    if len(datasets) > 1:
        report.append("\n## Cross-Dataset Comparison\n")
        report.append("| Dataset | NMI | ARI | ACC | Sparsity |")
        report.append("|---------|-----|-----|-----|----------|")
        
        for dataset in datasets:
            results = all_results.get(dataset, [])
            if results:
                nmi_mean, nmi_std, _ = compute_stats(results, ['clustering', 'kmeans', 'nmi'])
                ari_mean, ari_std, _ = compute_stats(results, ['clustering', 'kmeans', 'ari'])
                acc_mean, acc_std, _ = compute_stats(results, ['clustering', 'kmeans', 'acc'])
                sparsity_mean, _, _ = compute_stats(results, ['code_stats', 'sparsity'])
                
                nmi_str = f"{nmi_mean:.3f}±{nmi_std:.3f}" if nmi_mean else "N/A"
                ari_str = f"{ari_mean:.3f}±{ari_std:.3f}" if ari_mean else "N/A"
                acc_str = f"{acc_mean:.3f}±{acc_std:.3f}" if acc_mean else "N/A"
                sparsity_str = f"{sparsity_mean:.3f}" if sparsity_mean else "N/A"
                
                report.append(f"| {dataset.upper()} | {nmi_str} | {ari_str} | {acc_str} | {sparsity_str} |")
    
    # Interpretation
    report.append("\n## Interpretation\n")
    report.append("### Metrics Explanation\n")
    report.append("- **NMI (0-1)**: Higher is better. Measures cluster-label agreement.")
    report.append("- **ARI (-1 to 1)**: Higher is better. Measures clustering similarity.")
    report.append("- **ACC (0-1)**: Higher is better. Clustering accuracy with optimal mapping.")
    report.append("- **Sparsity (0-1)**: Higher means more sparse (fewer active features).")
    
    report.append("\n### Performance Guidelines\n")
    report.append("- **Excellent**: NMI > 0.7, ARI > 0.6, ACC > 0.8")
    report.append("- **Good**: NMI > 0.5, ARI > 0.4, ACC > 0.6")
    report.append("- **Moderate**: NMI > 0.3, ARI > 0.2, ACC > 0.4")
    report.append("- **Poor**: Below moderate thresholds")
    
    return "\n".join(report), all_results


def generate_latex_table(baseline, datasets, seeds):
    """Generate LaTeX table for paper."""
    table = []
    table.append("\\begin{table}[htbp]")
    table.append("\\centering")
    table.append(f"\\caption{{{baseline.upper()} Performance on Multiple Datasets}}")
    table.append("\\label{tab:flyhash_results}")
    table.append("\\begin{tabular}{lcccc}")
    table.append("\\toprule")
    table.append("Dataset & NMI & ARI & ACC & Sparsity \\\\")
    table.append("\\midrule")
    
    for dataset in datasets:
        results = load_results(baseline, dataset, seeds)
        if results:
            nmi_mean, nmi_std, _ = compute_stats(results, ['clustering', 'kmeans', 'nmi'])
            ari_mean, ari_std, _ = compute_stats(results, ['clustering', 'kmeans', 'ari'])
            acc_mean, acc_std, _ = compute_stats(results, ['clustering', 'kmeans', 'acc'])
            sparsity_mean, sparsity_std, _ = compute_stats(results, ['code_stats', 'sparsity'])
            
            if nmi_mean:
                table.append(
                    f"{dataset.replace('_', '-').upper()} & "
                    f"${nmi_mean:.3f} \\pm {nmi_std:.3f}$ & "
                    f"${ari_mean:.3f} \\pm {ari_std:.3f}$ & "
                    f"${acc_mean:.3f} \\pm {acc_std:.3f}$ & "
                    f"${sparsity_mean:.3f}$ \\\\"
                )
    
    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="Generate FlyHash benchmark report")
    parser.add_argument('--baseline', type=str, default='flyhash', help='Baseline name')
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'fashion_mnist'],
                        help='Datasets to include')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Seeds to aggregate')
    parser.add_argument('--output', type=str, default='outputs/flyhash_benchmark',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FlyHash Benchmark Report Generator")
    print("=" * 70)
    print(f"Baseline: {args.baseline}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()
    
    # Generate markdown report
    print("Generating markdown report...")
    markdown_report, all_results = generate_markdown_report(
        args.baseline, args.datasets, args.seeds
    )
    
    # Save markdown
    md_path = output_dir / "benchmark_report.md"
    with open(md_path, 'w') as f:
        f.write(markdown_report)
    print(f"✅ Markdown report saved: {md_path}")
    
    # Print to console
    print("\n" + "=" * 70)
    print(markdown_report)
    print("=" * 70)
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(args.baseline, args.datasets, args.seeds)
    
    latex_path = output_dir / "benchmark_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✅ LaTeX table saved: {latex_path}")
    
    # Save aggregated JSON
    print("\nSaving aggregated JSON...")
    json_output = output_dir / "benchmark_results.json"
    with open(json_output, 'w') as f:
        json.dump({
            'baseline': args.baseline,
            'datasets': args.datasets,
            'seeds': args.seeds,
            'generated': datetime.now().isoformat(),
            'results': {
                dataset: [
                    {
                        'seed': args.seeds[i],
                        'clustering': r.get('clustering', {}),
                        'code_stats': r.get('code_stats', {})
                    }
                    for i, r in enumerate(all_results.get(dataset, []))
                ]
                for dataset in args.datasets
            }
        }, f, indent=2)
    print(f"✅ JSON saved: {json_output}")
    
    print("\n" + "=" * 70)
    print("✅ Report generation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
