#!/bin/bash
# Run GPU end-to-end SIFT 100K benchmark on the remote instance.
#
# This script:
#   1. SSHes to the GPU instance
#   2. Starts the C++ GPU HE server (HEonGPU-based)
#   3. Runs the Go E2E benchmark test with GPU_HE_SERVER pointing to it
#   4. Collects results
#
# Prerequisites:
#   terraform apply -var="enabled=true"  (from deploy/gpu/)
#   Wait ~10 min for setup.sh to complete (builds HEonGPU + GPU server + downloads SIFT)
#
# Usage: bash deploy/gpu/run_gpu_e2e.sh
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

echo "=== Opaque GPU E2E Benchmark Runner ==="
echo "Instance: $IP"
echo ""

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i $KEY_FILE ubuntu@$IP"

# Wait for setup to complete.
echo "Waiting for instance setup to complete..."
for i in $(seq 1 90); do
  if $SSH "test -f /home/ubuntu/.setup-complete" 2>/dev/null; then
    echo "Setup complete!"
    break
  fi
  if [ $i -eq 90 ]; then
    echo "ERROR: Setup did not complete within 15 minutes."
    echo "Check logs: $SSH 'tail -50 /var/log/opaque-setup.log'"
    exit 1
  fi
  echo "  Waiting... ($i/90)"
  sleep 10
done

# Print GPU info.
echo ""
echo "=== GPU Info ==="
$SSH "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader" || true
echo ""

# Check if GPU HE server binary exists.
echo "=== Checking GPU HE server ==="
$SSH "ls -la /home/ubuntu/opaque/deploy/gpu/gpu-he-server/build/gpu-he-server 2>/dev/null && echo 'GPU server binary found' || echo 'GPU server binary NOT found'"
echo ""

# Check if SIFT data exists.
echo "=== Checking SIFT data ==="
$SSH "ls -la /home/ubuntu/opaque/data/sift/sift_base.fvecs 2>/dev/null && echo 'SIFT data found' || echo 'SIFT data NOT found'"
echo ""

# Kill any existing GPU server.
$SSH "pkill -f gpu-he-server 2>/dev/null || true"
sleep 1

# Start GPU HE server in background.
echo "=== Starting GPU HE server ==="
$SSH "nohup /home/ubuntu/opaque/deploy/gpu/gpu-he-server/build/gpu-he-server --port 50052 > /home/ubuntu/gpu-server.log 2>&1 &"
sleep 3

# Verify server is running.
if $SSH "pgrep -f gpu-he-server > /dev/null 2>&1"; then
  echo "GPU HE server running on port 50052"
  $SSH "tail -3 /home/ubuntu/gpu-server.log" || true
else
  echo "ERROR: GPU HE server failed to start"
  $SSH "cat /home/ubuntu/gpu-server.log" || true
  exit 1
fi
echo ""

# Run the E2E benchmark.
echo "=== Running GPU E2E Benchmark (SIFT 100K) ==="
echo "This will take ~10-15 minutes..."
echo ""

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/gpu_e2e_$TIMESTAMP.txt"

# Run with screen/nohup to survive SSH disconnects. Poll for completion.
$SSH "cd /home/ubuntu/opaque && nohup bash -c '
  export PATH=/usr/local/go/bin:\$HOME/go/bin:\$PATH
  export GOPATH=\$HOME/go
  GPU_HE_SERVER=localhost:50052 go test -tags sift1m -v -run TestGPU_E2E_SIFT100K ./test/ -timeout 30m
' > /home/ubuntu/gpu_e2e_result.log 2>&1 &"

echo "Benchmark started in background. Polling for completion..."
echo "(Will check every 30 seconds)"
echo ""

for i in $(seq 1 60); do
  # Check if the test process is still running
  if ! $SSH "pgrep -f 'TestGPU_E2E_SIFT100K' > /dev/null 2>&1" ; then
    echo "Test process finished."
    break
  fi

  # Show latest output
  LAST_LINE=$($SSH "tail -1 /home/ubuntu/gpu_e2e_result.log 2>/dev/null" || echo "...")
  echo "  [$i/60] Running... $LAST_LINE"
  sleep 30
done

echo ""
echo "=== RESULTS ==="
echo ""

# Fetch results.
$SSH "cat /home/ubuntu/gpu_e2e_result.log" | tee "$RESULT_FILE"

echo ""
echo "Results saved to: $RESULT_FILE"

# Also save GPU server log.
$SSH "cat /home/ubuntu/gpu-server.log" > "$RESULTS_DIR/gpu_server_$TIMESTAMP.log" 2>/dev/null || true

# Stop GPU server.
$SSH "pkill -f gpu-he-server 2>/dev/null || true"

echo ""
echo "=== Done ==="
echo "To SSH manually: $SSH"
echo "To destroy: cd deploy/gpu && terraform destroy -var='enabled=true'"
