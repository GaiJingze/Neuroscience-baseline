#!/bin/bash
# Batch test multiple baselines with different configurations

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Batch Baseline Testing"
echo "=========================================="
echo ""

# Configuration
BASELINES="${BASELINES:-flyhash}"  # Can be overridden
DATASETS="${DATASETS:-mnist}"
SEEDS="${SEEDS:-0 1 2}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baselines)
            BASELINES="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --quick)
            # Quick mode: single seed, single dataset
            SEEDS="0"
            DATASETS="mnist"
            shift
            ;;
        --full)
            # Full mode: multiple seeds, multiple datasets
            SEEDS="0 1 2"
            DATASETS="mnist fashion_mnist"
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/batch_test.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --baselines LIST    Space-separated baseline names (default: flyhash)"
            echo "  --datasets LIST     Space-separated dataset names (default: mnist)"
            echo "  --seeds LIST        Space-separated seeds (default: 0 1 2)"
            echo "  --quick             Quick mode: 1 seed, 1 dataset"
            echo "  --full              Full mode: 3 seeds, 2 datasets"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  bash scripts/batch_test.sh --baselines 'flyhash diehl_cook'"
            echo "  bash scripts/batch_test.sh --quick"
            echo "  bash scripts/batch_test.sh --full"
            echo ""
            echo "Environment variables:"
            echo "  BASELINES='flyhash diehl_cook' bash scripts/batch_test.sh"
            echo "  DATASETS='mnist fashion_mnist' bash scripts/batch_test.sh"
            echo "  SEEDS='0 1 2 3 4' bash scripts/batch_test.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Baselines: $BASELINES"
echo "  Datasets: $DATASETS"
echo "  Seeds: $SEEDS"
echo ""

# Count total tests
n_baselines=$(echo $BASELINES | wc -w)
n_datasets=$(echo $DATASETS | wc -w)
n_seeds=$(echo $SEEDS | wc -w)
total_tests=$((n_baselines * n_datasets * n_seeds))

echo "Total tests to run: $total_tests"
echo ""

# Track results
passed=0
failed=0
current=0

# Create results directory
mkdir -p outputs/batch_results
timestamp=$(date +%Y%m%d_%H%M%S)
result_file="outputs/batch_results/batch_test_${timestamp}.txt"

echo "Results will be saved to: $result_file"
echo ""

# Log header
{
    echo "=========================================="
    echo "Batch Test Results"
    echo "Timestamp: $(date)"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Baselines: $BASELINES"
    echo "  Datasets: $DATASETS"
    echo "  Seeds: $SEEDS"
    echo ""
} > "$result_file"

# Run tests
for baseline in $BASELINES; do
    echo -e "${YELLOW}Testing baseline: $baseline${NC}"
    
    for dataset in $DATASETS; do
        echo "  Dataset: $dataset"
        
        for seed in $SEEDS; do
            current=$((current + 1))
            echo -n "    [$current/$total_tests] Seed $seed ... "
            
            # Run test
            if python scripts/run_baseline.py \
                --config "configs/${baseline}.yaml" \
                --dataset "$dataset" \
                --seed "$seed" \
                > "/tmp/test_${baseline}_${dataset}_${seed}.log" 2>&1; then
                
                echo -e "${GREEN}PASS${NC}"
                passed=$((passed + 1))
                echo "PASS: $baseline on $dataset (seed=$seed)" >> "$result_file"
            else
                echo -e "${RED}FAIL${NC}"
                failed=$((failed + 1))
                echo "FAIL: $baseline on $dataset (seed=$seed)" >> "$result_file"
                
                # Log error
                echo "  Error log: /tmp/test_${baseline}_${dataset}_${seed}.log"
            fi
        done
    done
    echo ""
done

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total tests: $total_tests"
echo -e "${GREEN}Passed: $passed${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo ""

{
    echo ""
    echo "=========================================="
    echo "Summary"
    echo "=========================================="
    echo "Total: $total_tests"
    echo "Passed: $passed"
    echo "Failed: $failed"
} >> "$result_file"

echo "Full results saved to: $result_file"

# Exit with appropriate code
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed ✗${NC}"
    exit 1
fi
