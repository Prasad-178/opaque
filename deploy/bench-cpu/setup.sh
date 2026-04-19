#!/bin/bash
# Installs Go, build tools, and downloads SIFT1M dataset on a fresh Ubuntu
# EC2 instance. Run once per instance after the opaque source bundle has
# been scp'd to /tmp/opaque-bundle.tar.gz.
set -euo pipefail
export HOME=/home/ubuntu

echo "=== Opaque CPU bench setup — $(date) ==="

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

# Build tools (for transitive C dependencies, if any)
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl

# Extract opaque source
echo "Extracting opaque source..."
rm -rf /home/ubuntu/opaque
mkdir -p /home/ubuntu/opaque
tar -xzf /tmp/opaque-bundle.tar.gz -C /home/ubuntu/opaque
sudo chown -R ubuntu:ubuntu /home/ubuntu/opaque

# Download SIFT1M (~600MB)
echo "Downloading SIFT1M..."
cd /home/ubuntu/opaque
bash scripts/download_sift1m.sh

echo "=== Setup complete — $(date) ==="
touch /home/ubuntu/.setup-complete
