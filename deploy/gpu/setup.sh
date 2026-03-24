#!/bin/bash
# User-data script for GPU benchmark instance.
# Runs on first boot: installs Go, clones repo, builds GPU benchmark tools.
# The Deep Learning AMI already has CUDA + NVIDIA drivers pre-installed.
set -euo pipefail

LOG="/var/log/opaque-setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Opaque GPU Bench Setup — $(date) ==="

# --- Install Go ---
GO_VERSION="1.24.1"
echo "Installing Go $GO_VERSION..."
cd /tmp
curl -sL "https://go.dev/dl/go$GO_VERSION.tar.gz" -o go.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xzf go.tar.gz
rm go.tar.gz

# Set up Go for ubuntu user.
cat >> /home/ubuntu/.bashrc << 'GOEOF'
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export GOPATH=$HOME/go
GOEOF

export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
export GOPATH=/home/ubuntu/go

echo "Go version: $(go version)"

# --- Install build tools for GPU libraries ---
echo "Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential cmake git pkg-config

# --- Clone Opaque ---
echo "Cloning Opaque repo..."
cd /home/ubuntu
if [ -d "opaque" ]; then
  cd opaque && git pull
else
  git clone --branch "${repo_branch}" "${repo_url}" opaque
  cd opaque
fi

# --- Build Go project ---
echo "Building Opaque..."
cd /home/ubuntu/opaque
go build ./...
echo "Go build successful."

# --- Clone and build GPU-NTT (NTT microbenchmark) ---
echo "Cloning GPU-NTT..."
cd /home/ubuntu
if [ ! -d "GPU-NTT" ]; then
  git clone https://github.com/Alisah-Ozcan/GPU-NTT.git
fi

echo "Building GPU-NTT..."
cd GPU-NTT
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 2>/dev/null || echo "GPU-NTT cmake failed (may need CUDA arch adjustment)"
make -j$(nproc) 2>/dev/null || echo "GPU-NTT build failed (non-blocking)"

# --- Clone and build HEonGPU (CKKS GPU benchmark) ---
echo "Cloning HEonGPU..."
cd /home/ubuntu
if [ ! -d "HEonGPU" ]; then
  git clone https://github.com/Alisah-Ozcan/HEonGPU.git
fi

echo "Building HEonGPU..."
cd HEonGPU
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 2>/dev/null || echo "HEonGPU cmake failed (non-blocking)"
make -j$(nproc) 2>/dev/null || echo "HEonGPU build failed (non-blocking)"

# --- Fix permissions ---
chown -R ubuntu:ubuntu /home/ubuntu/opaque /home/ubuntu/GPU-NTT /home/ubuntu/HEonGPU 2>/dev/null || true
chown -R ubuntu:ubuntu /home/ubuntu/go 2>/dev/null || true

# --- Write ready marker ---
touch /home/ubuntu/.setup-complete
echo "=== Setup complete — $(date) ==="
echo "Run benchmarks with: bash opaque/deploy/gpu/run_benchmarks.sh"
