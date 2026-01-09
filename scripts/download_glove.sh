#!/bin/bash
# Download GloVe embeddings

set -e

DATADIR="data/glove"
mkdir -p $DATADIR

echo "Downloading GloVe 6B..."
cd $DATADIR

# Download
wget http://nlp.stanford.edu/data/glove.6B.zip

# Extract
echo "Extracting..."
unzip glove.6B.zip

# Clean up
rm glove.6B.zip

echo "GloVe download complete!"
echo "Files:"
ls -lh

cd ../..
