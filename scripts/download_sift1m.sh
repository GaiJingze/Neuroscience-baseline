#!/bin/bash
###############################################################################
# Download SIFT1M Dataset
#
# SIFT1M: 1 million 128-dimensional SIFT descriptors
# Source: http://corpus-texmex.irisa.fr/
# Size: ~500MB compressed, ~1.5GB uncompressed
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║              SIFT1M Dataset Downloader                       ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

DATADIR="data/sift1m"

# Check if already downloaded
if [ -f "${DATADIR}/sift_base.fvecs" ]; then
    echo -e "${YELLOW}SIFT1M dataset already exists at ${DATADIR}${NC}"
    echo -e "${YELLOW}Files found:${NC}"
    ls -lh ${DATADIR}/*.fvecs ${DATADIR}/*.ivecs 2>/dev/null || true
    echo ""
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✅ Using existing SIFT1M dataset${NC}"
        exit 0
    fi
    echo -e "${YELLOW}Removing existing data...${NC}"
    rm -rf ${DATADIR}
fi

# Create directory
mkdir -p ${DATADIR}

echo -e "${GREEN}Downloading SIFT1M dataset...${NC}"
echo "Source: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
echo "Size: ~500MB compressed"
echo ""

cd ${DATADIR}

# Download with progress
if command -v wget &> /dev/null; then
    wget --progress=bar:force ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
elif command -v curl &> /dev/null; then
    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
else
    echo -e "${RED}Error: Neither wget nor curl is available${NC}"
    exit 1
fi

# Extract
echo ""
echo -e "${GREEN}Extracting...${NC}"
tar -zxf sift.tar.gz

# Move files to current directory
if [ -d "sift" ]; then
    mv sift/* .
    rmdir sift
fi

# Clean up
rm sift.tar.gz

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║              ✅ SIFT1M Download Complete!                    ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Downloaded files:"
ls -lh *.fvecs *.ivecs 2>/dev/null || ls -lh

echo ""
echo "Dataset information:"
echo "  - sift_base.fvecs: 1,000,000 base vectors (128-dim)"
echo "  - sift_query.fvecs: 10,000 query vectors (128-dim)"
echo "  - sift_groundtruth.ivecs: Ground truth for queries"
echo "  - sift_learn.fvecs: 100,000 learning vectors (optional)"
echo ""
echo "Usage:"
echo "  python run.py --baseline flyhash --dataset sift1m --seed 0"
echo "  python run.py --config configs/flyhash_sift1m.yaml"
echo ""

cd ../..
