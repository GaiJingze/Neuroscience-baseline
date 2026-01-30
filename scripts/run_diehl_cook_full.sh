#!/bin/bash
# One-click script to clear old results and run full Diehl & Cook training with real STDP

cd "$(dirname "$0")/.." || exit 1

echo "🔥 Starting Diehl & Cook full training with real STDP..."
echo ""

# Run the Python script
python scripts/run_diehl_cook_full.py \
    --datasets mnist,fashion_mnist \
    --seeds 0,1,2 \
    --output-dir ./outputs

echo ""
echo "✅ Done! Check outputs/diehl_cook/ for results."
