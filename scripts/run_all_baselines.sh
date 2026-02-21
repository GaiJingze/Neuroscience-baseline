#!/usr/bin/env bash
# ===========================================================
# run_all_baselines.sh
#
# Run every baseline on every dataset with a single seed (0).
# This is the "test everything once" convenience script.
#
# Usage:
#   bash scripts/run_all_baselines.sh          # all baselines, all datasets
#   bash scripts/run_all_baselines.sh --quick   # fast baselines on MNIST only
# ===========================================================

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

QUICK=false
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=true ;;
  esac
done

echo "============================================================"
echo "  Neuroscience-baseline  —  Full Benchmark"
echo "============================================================"

if $QUICK; then
  echo "  Mode:  --quick  (MNIST only, skip diehl_cook)"
  echo ""
  python scripts/run_benchmark.py --quick --force
else
  echo "  Mode:  full  (all baselines x all datasets x seed 0)"
  echo ""
  echo "  NOTE: Diehl & Cook on MNIST may take several hours."
  echo "        To skip it, use:  bash scripts/run_all_baselines.sh --quick"
  echo ""
  python scripts/run_benchmark.py --seeds 0 --force
fi

echo ""
echo "Done.  Results saved to outputs/benchmark/"
echo "Summary:  outputs/benchmark/summary.json"
