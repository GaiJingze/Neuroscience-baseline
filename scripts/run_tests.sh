#!/bin/bash
# One-click test runner for clustering pipeline

set -e  # Exit on error

echo "=========================================="
echo "Clustering Pipeline Test Suite"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: Please run this script from the clustering/ directory${NC}"
    echo "Usage: bash scripts/run_tests.sh"
    exit 1
fi

# Function to run a test and report result
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "${YELLOW}Running: $test_name${NC}"
    if eval "$test_command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Error output:"
        cat /tmp/test_output.log
        return 1
    fi
}

# Track results
total_tests=0
passed_tests=0

echo "[1/6] Testing Python environment..."
total_tests=$((total_tests + 1))
if run_test "Python version" "python --version"; then
    passed_tests=$((passed_tests + 1))
fi
echo ""

echo "[2/6] Testing core imports..."
total_tests=$((total_tests + 1))
if run_test "Core dependencies" "python -c 'import numpy, torch, sklearn; print(\"OK\")'"; then
    passed_tests=$((passed_tests + 1))
fi
echo ""

echo "[3/6] Testing pipeline modules..."
total_tests=$((total_tests + 1))
if run_test "Pipeline imports" "python -c 'from pipeline import load_dataset, compute_nmi, set_seed; print(\"OK\")'"; then
    passed_tests=$((passed_tests + 1))
fi
echo ""

echo "[4/6] Running unit tests..."
total_tests=$((total_tests + 1))
if run_test "Unit tests" "python tests/test_pipeline.py"; then
    passed_tests=$((passed_tests + 1))
fi
echo ""

echo "[5/6] Testing baseline encoders..."
total_tests=$((total_tests + 1))
if run_test "FlyHash encoder" "python baselines/flyhash/encoder.py"; then
    passed_tests=$((passed_tests + 1))
fi
echo ""

echo "[6/6] Testing dataset loading..."
total_tests=$((total_tests + 1))
if run_test "Dataset loading" "python -c 'from pipeline.datasets import load_dataset; d = load_dataset(\"mnist\", download=True); print(\"MNIST:\", d[\"test_data\"].shape)'"; then
    passed_tests=$((passed_tests + 1))
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Total tests: $total_tests"
echo -e "${GREEN}Passed: $passed_tests${NC}"
echo -e "${RED}Failed: $((total_tests - passed_tests))${NC}"
echo ""

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed ✗${NC}"
    echo "Check the output above for details"
    exit 1
fi
