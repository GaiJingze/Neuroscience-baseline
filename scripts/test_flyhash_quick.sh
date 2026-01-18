#!/bin/bash

###############################################################################
# FlyHash Quick Test Script (Single Seed)
#
# Runs FlyHash once on each dataset for quick testing
###############################################################################

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║       FlyHash Quick Test (MNIST + Fashion-MNIST)             ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Clear cache before running to ensure fresh results
echo "Clearing FlyHash cache to ensure fresh results..."
python scripts/clear_cache.py --baseline flyhash --yes
echo ""

SEED=0

# Test 1: MNIST
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1/2: MNIST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run.py --baseline flyhash --dataset mnist --seed ${SEED}
echo ""

# Test 2: Fashion-MNIST
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2/2: Fashion-MNIST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run.py --baseline flyhash --dataset fashion_mnist --seed ${SEED}
echo ""

# Generate quick summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║                    Quick Summary                             ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python << 'EOF'
import json
from pathlib import Path

datasets = ['mnist', 'fashion_mnist']
seed = 0

print("Dataset        | NMI    | ARI    | ACC    | Sparsity")
print("---------------|--------|--------|--------|----------")

for dataset in datasets:
    result_file = Path(f"outputs/results/flyhash_{dataset}_seed{seed}.json")
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            clustering = data.get('clustering', {}).get('kmeans', {})
            nmi = clustering.get('nmi', 0)
            ari = clustering.get('ari', 0)
            acc = clustering.get('acc', 0)
            sparsity = data.get('code_stats', {}).get('sparsity', 0)
            print(f"{dataset:14} | {nmi:.4f} | {ari:.4f} | {acc:.4f} | {sparsity:.4f}")
    else:
        print(f"{dataset:14} | N/A    | N/A    | N/A    | N/A")

print("")
print("✅ Quick test complete!")
print(f"   Results: outputs/results/flyhash_*_seed{seed}.json")
EOF
