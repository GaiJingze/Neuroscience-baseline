#!/bin/bash
###############################################################################
# SIFT-1M Benchmark Script for FlyHash
#
# This script:
# 1. Checks if SIFT-1M data exists
# 2. Downloads if needed
# 3. Runs FlyHash on SIFT-1M
# 4. Generates report
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║           SIFT-1M Benchmark for FlyHash                      ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
SEEDS=(0 1 2)
DATADIR="data/sift1m"

###############################################################################
# Step 1: Check if SIFT-1M exists
###############################################################################
echo -e "${GREEN}[1/4] Checking SIFT-1M dataset...${NC}"

if [ ! -f "${DATADIR}/sift_base.fvecs" ]; then
    echo -e "${YELLOW}SIFT-1M dataset not found!${NC}"
    echo ""
    echo "The SIFT-1M dataset needs to be downloaded first."
    echo "Size: ~500MB compressed, ~1.5GB uncompressed"
    echo ""
    read -p "Do you want to download it now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Downloading SIFT-1M...${NC}"
        bash scripts/download_sift1m.sh
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download SIFT-1M dataset${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Skipping SIFT-1M benchmark${NC}"
        echo ""
        echo "To download manually, run:"
        echo "  bash scripts/download_sift1m.sh"
        exit 0
    fi
else
    echo -e "${GREEN}✅ SIFT-1M dataset found${NC}"
fi

###############################################################################
# Step 2: Test dataset loading
###############################################################################
echo ""
echo -e "${GREEN}[2/4] Testing dataset loading...${NC}"

if python scripts/test_sift1m.py --subset 1000; then
    echo -e "${GREEN}✅ Dataset loading test passed${NC}"
else
    echo -e "${RED}❌ Dataset loading test failed${NC}"
    exit 1
fi

###############################################################################
# Step 3: Run FlyHash on SIFT-1M
###############################################################################
echo ""
echo -e "${GREEN}[3/4] Running FlyHash on SIFT-1M...${NC}"
echo "Seeds: ${SEEDS[@]}"
echo ""

for seed in "${SEEDS[@]}"; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running with seed=${seed}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if python run.py --config configs/flyhash_sift1m.yaml --seed ${seed}; then
        echo -e "${GREEN}✅ Seed ${seed} completed${NC}"
    else
        echo -e "${RED}❌ Seed ${seed} failed${NC}"
    fi
    echo ""
done

###############################################################################
# Step 4: Generate report
###############################################################################
echo ""
echo -e "${GREEN}[4/4] Generating report...${NC}"

python scripts/generate_flyhash_report.py \
    --baseline flyhash \
    --datasets sift1m \
    --seeds "${SEEDS[@]}" \
    --output outputs/sift1m_benchmark

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Report generated successfully${NC}"
    echo ""
    echo "Results:"
    echo "  - Report: outputs/sift1m_benchmark/benchmark_report.md"
    echo "  - LaTeX: outputs/sift1m_benchmark/benchmark_table.tex"
    echo "  - JSON: outputs/sift1m_benchmark/benchmark_results.json"
else
    echo -e "${YELLOW}⚠️  Report generation failed (may be incomplete results)${NC}"
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║              ✅ SIFT-1M Benchmark Complete!                  ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show quick results
echo "Quick results:"
python << 'EOF'
import json
from pathlib import Path

seeds = [0, 1, 2]
results_found = False

print("\nDataset: SIFT-1M")
print("Metric          | Seed 0  | Seed 1  | Seed 2  | Mean")
print("----------------|---------|---------|---------|--------")

metrics = ['nmi', 'ari', 'acc']
for metric in metrics:
    values = []
    for seed in seeds:
        result_file = Path(f"outputs/results/flyhash_sift1m_seed{seed}.json")
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                val = data.get('clustering', {}).get('kmeans', {}).get(metric, 0)
                values.append(val)
                results_found = True
    
    if values:
        mean_val = sum(values) / len(values)
        vals_str = " | ".join([f"{v:.4f}" for v in values])
        print(f"{metric.upper():15} | {vals_str} | {mean_val:.4f}")

if not results_found:
    print("No results found. Check outputs/results/ directory.")

EOF

echo ""
echo "For detailed report, see:"
echo "  cat outputs/sift1m_benchmark/benchmark_report.md"
