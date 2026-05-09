#!/bin/bash
# DBpedia1M variant of setup.sh. Same Go install + bundle extract as setup.sh,
# but downloads DBpedia-OpenAI-1M (1M × 1536-dim ada-002 embeddings from
# HuggingFace) instead of SIFT1M, and converts the parquet shards to fvecs.
#
# Run after the opaque source bundle has been scp'd to /tmp/opaque-bundle.tar.gz.
set -euo pipefail
export HOME=/home/ubuntu

echo "=== Opaque CPU bench (DBpedia variant) setup — $(date) ==="

# Go 1.24.4
if ! command -v go > /dev/null; then
  echo "Installing Go 1.24.4..."
  cd /tmp
  curl -fSL https://go.dev/dl/go1.24.4.linux-amd64.tar.gz -o go.tar.gz
  sudo rm -rf /usr/local/go
  sudo tar -C /usr/local -xzf go.tar.gz
  rm go.tar.gz
fi

echo 'export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH' | sudo tee -a /home/ubuntu/.bashrc > /dev/null
echo 'export GOPATH=$HOME/go' | sudo tee -a /home/ubuntu/.bashrc > /dev/null
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export GOPATH=$HOME/go

echo "Go version: $(/usr/local/go/bin/go version)"

# Build tools + Python (DBpedia download script needs Python+pyarrow+huggingface-hub).
sudo apt-get update -qq || true
sudo apt-get install -y -qq build-essential curl python3 python3-pip 2>/dev/null || \
  sudo apt-get install -y -qq gcc make curl python3 python3-pip 2>/dev/null || \
  echo "warn: package install partial; continuing"

# Extract opaque source.
echo "Extracting opaque source..."
rm -rf /home/ubuntu/opaque
mkdir -p /home/ubuntu/opaque
tar -xzf /tmp/opaque-bundle.tar.gz -C /home/ubuntu/opaque
sudo chown -R ubuntu:ubuntu /home/ubuntu/opaque

# Download + convert DBpedia-OpenAI-1M (~6GB after fvecs conversion, ~12GB peak).
echo "Downloading + converting DBpedia-OpenAI-1M..."
cd /home/ubuntu/opaque
bash scripts/download_dbpedia1m.sh

echo "=== Setup complete — $(date) ==="
df -h /
touch /home/ubuntu/.setup-complete
