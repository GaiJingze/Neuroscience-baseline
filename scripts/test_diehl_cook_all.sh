#!/bin/bash

###############################################################################
# Diehl & Cook Complete Test Script
#
# Runs all Diehl & Cook experiments with automatic progress tracking
###############################################################################

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║       Diehl & Cook Batch Training                            ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we should skip CIFAR-10
SKIP_CIFAR10=""
if [ "$1" == "--skip-cifar10" ]; then
    SKIP_CIFAR10="--skip-cifar10"
    echo "⚡ Fast mode: Skipping CIFAR-10"
    echo ""
fi

# Run batch training
python scripts/run_diehl_cook_batch.py $SKIP_CIFAR10 --skip-on-error

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║       Batch training complete!                               ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  - outputs/results/diehl_cook_*.json"
echo "  - outputs/reports/diehl_cook_batch_report.txt"
echo ""
echo "View report:"
echo "  cat outputs/reports/diehl_cook_batch_report.txt"
echo ""
