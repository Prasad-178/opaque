#!/bin/bash
# Run GPU benchmarks on the remote instance.
# Usage: bash deploy/gpu/run_benchmarks.sh
#
# Prerequisites:
#   1. terraform apply -var="enabled=true"  (from deploy/gpu/)
#   2. Wait ~5 min for setup.sh to complete
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEY_FILE="$SCRIPT_DIR/gpu-bench-key.pem"
RESULTS_DIR="$SCRIPT_DIR/results"

# Get instance IP from Terraform output.
cd "$SCRIPT_DIR"
IP=$(terraform output -raw instance_ip 2>/dev/null)

if [ "$IP" = "disabled" ] || [ -z "$IP" ]; then
  echo "ERROR: GPU instance is not running."
  echo "Run: cd deploy/gpu && terraform apply -var='enabled=true'"
  exit 1
fi

echo "=== Opaque GPU Benchmark Runner ==="
echo "Instance: $IP"
echo ""

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $KEY_FILE ubuntu@$IP"

# Wait for setup to complete.
echo "Waiting for instance setup to complete..."
for i in $(seq 1 60); do
  if $SSH "test -f /home/ubuntu/.setup-complete" 2>/dev/null; then
    echo "Setup complete!"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: Setup did not complete within 10 minutes."
    echo "Check logs: $SSH 'cat /var/log/opaque-setup.log'"
    exit 1
  fi
  echo "  Waiting... ($i/60)"
  sleep 10
done

# Print GPU info.
echo ""
echo "=== GPU Info ==="
$SSH "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
echo ""

# Run CUDA availability check.
echo "=== CUDA Check ==="
$SSH "nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not in PATH'"
echo ""

# Run Opaque HE benchmarks (CPU baseline on GPU instance for fair comparison).
echo "=== HE CPU Baseline (on GPU instance) ==="
$SSH "cd /home/ubuntu/opaque && go test -bench=BenchmarkEncryption -benchmem ./pkg/crypto/ 2>&1 | grep -E 'Benchmark|ok'" || true
echo ""

# Run GPU-NTT benchmark if available.
echo "=== GPU-NTT Benchmark ==="
$SSH "cd /home/ubuntu/GPU-NTT/build && ls *.out 2>/dev/null && echo 'Running...' || echo 'GPU-NTT not built (check setup log)'"
$SSH "cd /home/ubuntu/GPU-NTT/build && ./ntt_benchmark.out 2>/dev/null || echo 'GPU-NTT benchmark not available'" || true
echo ""

# Run HEonGPU benchmark if available.
echo "=== HEonGPU CKKS Benchmark ==="
$SSH "cd /home/ubuntu/HEonGPU/build && ls 2>/dev/null && echo 'Build directory exists' || echo 'HEonGPU not built (check setup log)'"
$SSH "cd /home/ubuntu/HEonGPU/build && ./benchmark_ckks 2>/dev/null || echo 'HEonGPU benchmark not available'" || true
echo ""

# Collect results.
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/gpu_bench_$TIMESTAMP.txt"

{
  echo "=== Opaque GPU Benchmark Results ==="
  echo "Date: $(date)"
  echo "Instance: $IP"
  echo ""
  echo "--- GPU Info ---"
  $SSH "nvidia-smi" 2>/dev/null || true
  echo ""
  echo "--- Opaque HE CPU Benchmarks ---"
  $SSH "cd /home/ubuntu/opaque && go test -bench=. -benchmem ./pkg/crypto/ 2>&1" || true
  echo ""
  echo "--- Opaque PQ Benchmarks ---"
  $SSH "cd /home/ubuntu/opaque && go test -bench=. -benchmem ./pkg/pq/ 2>&1" || true
} > "$RESULT_FILE" 2>&1

echo "Results saved to: $RESULT_FILE"
echo ""
echo "=== Done ==="
echo "To SSH manually: $SSH"
echo "To destroy: cd deploy/gpu && terraform destroy -var='enabled=true'"
