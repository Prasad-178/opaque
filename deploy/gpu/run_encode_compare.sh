#!/bin/bash
# run_encode_compare.sh — invoked on the T4 after setup completes.
# Pulls latest opaque code, builds encode_compare, runs it, prints output.
#
# Usage (on instance): bash /home/ubuntu/run_encode_compare.sh

set -euo pipefail

echo "=== Pulling latest opaque ==="
cd /home/ubuntu/opaque
git fetch origin
git checkout main
git reset --hard origin/main

echo "=== Regenerating build ==="
cd deploy/gpu/gpu-he-server
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 2>&1 | tail -10

echo "=== Building encode_compare ==="
make encode_compare -j$(nproc) 2>&1 | tail -20

echo "=== Running encode_compare ==="
./encode_compare /tmp/heongpu_dump.bin

echo "=== Done. Dump at /tmp/heongpu_dump.bin ==="
ls -la /tmp/heongpu_dump.bin
