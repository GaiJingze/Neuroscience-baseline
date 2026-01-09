#!/bin/bash
# Download SIFT1M dataset

set -e

DATADIR="data/sift1m"
mkdir -p $DATADIR

echo "Downloading SIFT1M dataset..."
cd $DATADIR

# Download
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

# Extract
echo "Extracting..."
tar -zxvf sift.tar.gz

# Move files to current directory
mv sift/* .
rmdir sift

# Clean up
rm sift.tar.gz

echo "SIFT1M download complete!"
echo "Files:"
ls -lh

cd ../..
