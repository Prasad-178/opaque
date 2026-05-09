#!/bin/bash
# Orchestrates an Opaque SIFT 1M benchmark run on a fresh EC2 CPU instance.
#
# Flow:
#   1. terraform apply (personal AWS profile, given instance type)
#   2. scp opaque source bundle + setup script
#   3. ssh + run setup (installs Go, downloads SIFT1M)
#   4. ssh + run TestSIFT1MAccuracy and TestPQ_SIFT1M (sift1m build tag)
#   5. pull logs into deploy/bench-cpu/results/<timestamp>-<instance>/
#   6. terraform destroy
#
# Cost: ~$0.68/hr × ~1hr (c6i.4xlarge) or $0.34/hr × ~1.5hr (c6i.2xlarge).
# Destroy runs in an always-on trap so an interrupted run still tears down.
#
# Usage: deploy/bench-cpu/run_bench.sh c6i.2xlarge
set -euo pipefail

INSTANCE_TYPE="${1:-c6i.2xlarge}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KEY_FILE="$SCRIPT_DIR/bench-cpu-key.pem"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/${TIMESTAMP}-${INSTANCE_TYPE}"
BUNDLE="/tmp/opaque-bundle-bench.tar.gz"

mkdir -p "$RESULTS_DIR"

echo "=== Opaque SIFT1M CPU bench: $INSTANCE_TYPE ==="
echo "Results: $RESULTS_DIR"

# Bundle source (exclude heavy/irrelevant paths)
echo "Bundling source..."
tar --exclude=.git --exclude=third_party --exclude=data \
    --exclude='*.tfstate*' --exclude='*.pem' --exclude=deploy/gpu/.terraform \
    --exclude=deploy/bench-cpu/.terraform --exclude=deploy/bench-cpu/results \
    -czf "$BUNDLE" -C "$REPO_ROOT" .
echo "Bundle: $(du -h "$BUNDLE" | cut -f1)"

cd "$SCRIPT_DIR"

# Trap teardown on any exit path.
cleanup() {
  echo ""
  echo "=== Destroying infra ==="
  terraform destroy -var="enabled=true" \
    -var="aws_profile=personal" \
    -var="instance_type=$INSTANCE_TYPE" \
    -auto-approve || true
}
trap cleanup EXIT

# Bring up EC2.
terraform init -upgrade > /dev/null
terraform apply -var="enabled=true" \
  -var="aws_profile=personal" \
  -var="instance_type=$INSTANCE_TYPE" \
  -auto-approve

IP="$(terraform output -raw instance_ip)"
echo "Instance IP: $IP"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 ubuntu@$IP"

# Wait for SSH.
echo "Waiting for SSH..."
for i in $(seq 1 30); do
  if $SSH 'echo ready' > /dev/null 2>&1; then
    echo "SSH ready after ${i}×5s"
    break
  fi
  sleep 5
  if [ "$i" -eq 30 ]; then
    echo "SSH never came up"; exit 1
  fi
done

# Upload source + setup script.
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no "$BUNDLE" "ubuntu@$IP:/tmp/opaque-bundle.tar.gz"
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no "$SCRIPT_DIR/setup.sh" "ubuntu@$IP:/tmp/bench-setup.sh"

# Run setup.
echo ""
echo "=== Installing Go + downloading SIFT1M ==="
$SSH 'bash /tmp/bench-setup.sh' 2>&1 | tee "$RESULTS_DIR/setup.log"

# Run SIFT1M accuracy + PQ benchmarks.
echo ""
echo "=== Running SIFT1M benchmarks ==="
$SSH '
  export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
  export GOPATH=$HOME/go
  cd /home/ubuntu/opaque
  nproc
  echo "--- TestSIFT1MAccuracy (NC=128 headline) ---"
  go test -tags sift1m -count=1 -v -run "^TestSIFT1MAccuracy$" ./test/ -timeout 90m
  echo ""
  echo "--- TestSIFT1MAccuracy_NC256 ---"
  go test -tags sift1m -count=1 -v -run "^TestSIFT1MAccuracy_NC256$" ./test/ -timeout 90m
  echo ""
  echo "--- TestPQ_SIFT1M (NC=128 headline) ---"
  go test -tags sift1m -count=1 -v -run "^TestPQ_SIFT1M$" ./test/ -timeout 90m
  echo ""
  echo "--- TestPQ_SIFT1M_NC256 ---"
  go test -tags sift1m -count=1 -v -run "^TestPQ_SIFT1M_NC256$" ./test/ -timeout 90m
' 2>&1 | tee "$RESULTS_DIR/bench.log"

# Meta.
$SSH 'uname -a; lscpu | head -20; free -h' 2>&1 | tee "$RESULTS_DIR/system.log"

echo ""
echo "=== Done — results in $RESULTS_DIR ==="
