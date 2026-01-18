#!/bin/bash

###############################################################################
# FlyHash Comprehensive Testing Script
#
# This script tests FlyHash on multiple datasets:
# - MNIST
# - Fashion-MNIST  
# - SIFT-1M (subset)
#
# For each dataset, it runs multiple seeds for statistical analysis.
###############################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SEEDS=(0 1 2)  # Multiple seeds for statistical analysis
RESULTS_DIR="./outputs/flyhash_benchmark"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULTS_DIR}/test_${TIMESTAMP}.log"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a ${LOG_FILE}
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a ${LOG_FILE}
}

log_section() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}" | tee -a ${LOG_FILE}
    echo -e "${BLUE}║${NC}  $1" | tee -a ${LOG_FILE}
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}\n" | tee -a ${LOG_FILE}
}

# Start testing
log_section "FlyHash Comprehensive Benchmark"
log "Starting FlyHash evaluation on multiple datasets"
log "Seeds: ${SEEDS[@]}"
log "Results will be saved to: ${RESULTS_DIR}"

# Track success/failure
declare -A results
total_tests=0
passed_tests=0
failed_tests=0

###############################################################################
# Test 1: MNIST
###############################################################################
log_section "Test 1/3: MNIST (28x28 grayscale digits)"

for seed in "${SEEDS[@]}"; do
    total_tests=$((total_tests + 1))
    log "Running FlyHash on MNIST (seed=${seed})..."
    
    if python run.py --baseline flyhash --dataset mnist --seed ${seed} 2>&1 | tee -a ${LOG_FILE}; then
        log "✅ MNIST (seed=${seed}) completed successfully"
        results["mnist_${seed}"]="✅ PASS"
        passed_tests=$((passed_tests + 1))
    else
        log_error "MNIST (seed=${seed}) failed"
        results["mnist_${seed}"]="❌ FAIL"
        failed_tests=$((failed_tests + 1))
    fi
    
    echo "" | tee -a ${LOG_FILE}
done

###############################################################################
# Test 2: Fashion-MNIST
###############################################################################
log_section "Test 2/3: Fashion-MNIST (28x28 fashion items)"

for seed in "${SEEDS[@]}"; do
    total_tests=$((total_tests + 1))
    log "Running FlyHash on Fashion-MNIST (seed=${seed})..."
    
    if python run.py --baseline flyhash --dataset fashion_mnist --seed ${seed} 2>&1 | tee -a ${LOG_FILE}; then
        log "✅ Fashion-MNIST (seed=${seed}) completed successfully"
        results["fashion_mnist_${seed}"]="✅ PASS"
        passed_tests=$((passed_tests + 1))
    else
        log_error "Fashion-MNIST (seed=${seed}) failed"
        results["fashion_mnist_${seed}"]="❌ FAIL"
        failed_tests=$((failed_tests + 1))
    fi
    
    echo "" | tee -a ${LOG_FILE}
done

###############################################################################
# Test 3: SIFT-1M (subset for testing)
###############################################################################
log_section "Test 3/3: SIFT-1M (128-dim SIFT features)"

log "${YELLOW}Note: SIFT-1M is a large dataset. Testing on subset...${NC}"

# Check if SIFT-1M data exists
if [ ! -d "./data/sift1m" ]; then
    log "${YELLOW}SIFT-1M data not found. Skipping SIFT-1M tests.${NC}"
    log "${YELLOW}To download SIFT-1M, run: scripts/download_sift1m.sh${NC}"
else
    for seed in "${SEEDS[@]}"; do
        total_tests=$((total_tests + 1))
        log "Running FlyHash on SIFT-1M (seed=${seed})..."
        
        # Note: May need to create a separate config for SIFT-1M
        if python run.py --baseline flyhash --dataset sift1m --seed ${seed} 2>&1 | tee -a ${LOG_FILE}; then
            log "✅ SIFT-1M (seed=${seed}) completed successfully"
            results["sift1m_${seed}"]="✅ PASS"
            passed_tests=$((passed_tests + 1))
        else
            log_error "SIFT-1M (seed=${seed}) failed (may not be implemented yet)"
            results["sift1m_${seed}"]="❌ FAIL"
            failed_tests=$((failed_tests + 1))
        fi
        
        echo "" | tee -a ${LOG_FILE}
    done
fi

###############################################################################
# Generate Summary Report
###############################################################################
log_section "Test Summary"

# Print results table
log "Individual Test Results:"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for key in "${!results[@]}"; do
    log "  ${key}: ${results[$key]}"
done
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log ""
log "Overall Statistics:"
log "  Total Tests:  ${total_tests}"
log "  Passed:       ${GREEN}${passed_tests}${NC}"
log "  Failed:       ${RED}${failed_tests}${NC}"
log "  Success Rate: $(awk "BEGIN {printf \"%.1f%%\", 100*${passed_tests}/${total_tests}}")"

###############################################################################
# Aggregate Results
###############################################################################
log_section "Aggregating Results"

log "Generating aggregated performance report..."

# Create Python script to aggregate results
cat > ${RESULTS_DIR}/aggregate_results.py << 'PYTHON_SCRIPT'
import json
import numpy as np
from pathlib import Path
import sys

def load_results(dataset, seeds):
    """Load results for a dataset across multiple seeds."""
    results = []
    for seed in seeds:
        result_file = Path(f"outputs/results/flyhash_{dataset}_seed{seed}.json")
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        else:
            print(f"Warning: {result_file} not found")
    return results

def compute_stats(results, metric_path):
    """Compute mean and std for a metric."""
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

def generate_report(datasets, seeds):
    """Generate markdown report."""
    report = []
    report.append("# FlyHash Benchmark Report")
    report.append(f"\nGenerated: {Path(__file__).parent.name}")
    report.append(f"\nSeeds: {seeds}")
    report.append("\n## Results Summary\n")
    
    # Table header
    report.append("| Dataset | Metric | Mean ± Std | N |")
    report.append("|---------|--------|------------|---|")
    
    metrics = [
        (['clustering', 'kmeans', 'nmi'], 'NMI'),
        (['clustering', 'kmeans', 'ari'], 'ARI'),
        (['clustering', 'kmeans', 'acc'], 'ACC'),
        (['code_stats', 'sparsity'], 'Sparsity'),
    ]
    
    all_results = {}
    for dataset in datasets:
        results = load_results(dataset, seeds)
        all_results[dataset] = results
        
        if results:
            report.append(f"| **{dataset.upper()}** | | | |")
            for metric_path, metric_name in metrics:
                mean, std, n = compute_stats(results, metric_path)
                if mean is not None:
                    report.append(f"| | {metric_name} | {mean:.4f} ± {std:.4f} | {n} |")
                else:
                    report.append(f"| | {metric_name} | N/A | 0 |")
    
    return "\n".join(report), all_results

if __name__ == "__main__":
    datasets = ['mnist', 'fashion_mnist']  # sift1m may not be implemented
    seeds = [0, 1, 2]
    
    report, all_results = generate_report(datasets, seeds)
    
    # Save report
    output_path = Path(__file__).parent / "benchmark_report.md"
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Report saved to: {output_path}")
    
    # Save aggregated JSON
    json_output = Path(__file__).parent / "benchmark_results.json"
    with open(json_output, 'w') as f:
        json.dump({
            'datasets': datasets,
            'seeds': seeds,
            'results': {
                dataset: [
                    {
                        'seed': seeds[i],
                        'clustering': r.get('clustering', {}),
                        'code_stats': r.get('code_stats', {})
                    }
                    for i, r in enumerate(all_results.get(dataset, []))
                ]
                for dataset in datasets
            }
        }, f, indent=2)
    
    print(f"✅ JSON saved to: {json_output}")
PYTHON_SCRIPT

# Run aggregation
if python ${RESULTS_DIR}/aggregate_results.py 2>&1 | tee -a ${LOG_FILE}; then
    log "✅ Results aggregated successfully"
else
    log_error "Failed to aggregate results"
fi

###############################################################################
# Final Summary
###############################################################################
log_section "Benchmark Complete!"

log "Results saved to:"
log "  - Log file: ${LOG_FILE}"
log "  - Report: ${RESULTS_DIR}/benchmark_report.md"
log "  - JSON: ${RESULTS_DIR}/benchmark_results.json"
log "  - Individual results: outputs/results/flyhash_*.json"

if [ ${failed_tests} -eq 0 ]; then
    log "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    log "${YELLOW}⚠️  ${failed_tests}/${total_tests} tests failed${NC}"
    exit 1
fi
