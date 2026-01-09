#!/bin/bash
# Run all baselines with multiple seeds

set -e

SEEDS=(0 1 2)
CONFIGS=(
    "configs/flyhash.yaml"
    # "configs/diehl_cook.yaml"  # Uncomment when ready
)

echo "Running all baselines with seeds: ${SEEDS[@]}"
echo "Configs: ${CONFIGS[@]}"
echo ""

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "Running: $config with seed=$seed"
        echo "=========================================="
        
        python scripts/run_baseline.py \
            --config "$config" \
            --seed "$seed"
        
        echo ""
        echo "Completed: $config seed=$seed"
        echo ""
    done
done

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "View results in outputs/results/"
