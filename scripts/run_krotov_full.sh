#!/bin/bash
# Run complete Krotov experiments on MNIST and Fashion-MNIST
# Total time: ~1.5 hours (5 experiments × 15-20 min each)

cd "$(dirname "$0")/.." || exit 1

echo "======================================================================"
echo "🔥 Running Krotov Method - Complete Experiments"
echo "======================================================================"
echo ""
echo "Datasets: MNIST, Fashion-MNIST"
echo "Seeds: 0, 1, 2"
echo "Total experiments: 6 (5 remaining)"
echo "Estimated time: ~1.5 hours"
echo ""
echo "======================================================================"
echo ""

# Counter
total=5
current=0

# MNIST seed=0 is already done, skip it
echo "⏭️  Skipping MNIST seed=0 (already completed)"
echo ""

# MNIST seeds 1-2
for seed in 1 2; do
    current=$((current + 1))
    echo "======================================================================"
    echo "📊 Experiment $current/$total: MNIST seed=$seed"
    echo "======================================================================"
    python scripts/run_baseline.py --config configs/krotov.yaml --dataset mnist --seed $seed
    
    if [ $? -eq 0 ]; then
        echo "✅ MNIST seed=$seed completed successfully"
    else
        echo "❌ MNIST seed=$seed failed"
    fi
    echo ""
done

# Fashion-MNIST seeds 0-2
for seed in 0 1 2; do
    current=$((current + 1))
    echo "======================================================================"
    echo "📊 Experiment $current/$total: Fashion-MNIST seed=$seed"
    echo "======================================================================"
    python scripts/run_baseline.py --config configs/krotov.yaml --dataset fashion_mnist --seed $seed
    
    if [ $? -eq 0 ]; then
        echo "✅ Fashion-MNIST seed=$seed completed successfully"
    else
        echo "❌ Fashion-MNIST seed=$seed failed"
    fi
    echo ""
done

echo "======================================================================"
echo "🎉 All experiments complete!"
echo "======================================================================"
echo ""
echo "📊 Collect results with:"
echo "   python scripts/collect_all_results.py"
echo ""
