#!/bin/bash
# Quick + cheap SIFT 100K benchmark — for post-refactor regression checks.
#
# Same flow as run_bench.sh but only runs TestPQ_SIFT100K (100K subset of
# SIFT1M). Total wall: ~8-10 min including SIFT1M dataset download (~600 MB)
# + Go install + bench. Used to verify that recent refactors (e.g. float32
# AES ciphertext encoding in commit 7a9a369, kmeans_builder mem opts in
# commit 1e735ec) haven't regressed recall or latency.
#
# Default instance: m6i.xlarge (4 vCPU, 16 GB) at ~$0.19/hr — plenty of
# headroom for 100K × 128-dim and the SIFT1M dataset on disk.
#
# Cost: ~$0.19/hr × ~0.2 hr ≈ $0.04 per run.

set -euo pipefail

INSTANCE_TYPE="${1:-m6i.xlarge}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KEY_FILE="$SCRIPT_DIR/bench-cpu-key.pem"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/${TIMESTAMP}-${INSTANCE_TYPE}-sift100k"
BUNDLE="/tmp/opaque-bundle-bench.tar.gz"

mkdir -p "$RESULTS_DIR"

echo "=== Opaque SIFT 100K CPU bench: $INSTANCE_TYPE ==="
echo "Results: $RESULTS_DIR"

echo "Bundling source..."
tar --exclude=.git --exclude=third_party --exclude=data \
    --exclude='*.tfstate*' --exclude='*.pem' --exclude=deploy/gpu/.terraform \
    --exclude=deploy/bench-cpu/.terraform --exclude=deploy/bench-cpu/results \
    -czf "$BUNDLE" -C "$REPO_ROOT" .
echo "Bundle: $(du -h "$BUNDLE" | cut -f1)"

cd "$SCRIPT_DIR"

cleanup() {
  echo ""
  echo "=== Destroying infra ==="
  terraform destroy -var="enabled=true" \
    -var="aws_profile=personal" \
    -var="instance_type=$INSTANCE_TYPE" \
    -auto-approve || true
}
trap cleanup EXIT

terraform init -upgrade > /dev/null
terraform apply -var="enabled=true" \
  -var="aws_profile=personal" \
  -var="instance_type=$INSTANCE_TYPE" \
  -auto-approve

IP="$(terraform output -raw instance_ip)"
echo "Instance IP: $IP"

SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 ubuntu@$IP"

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

scp -i "$KEY_FILE" -o StrictHostKeyChecking=no "$BUNDLE" "ubuntu@$IP:/tmp/opaque-bundle.tar.gz"
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no "$SCRIPT_DIR/setup.sh" "ubuntu@$IP:/tmp/bench-setup.sh"

echo ""
echo "=== Installing Go + downloading SIFT1M ==="
$SSH 'bash /tmp/bench-setup.sh' 2>&1 | tee "$RESULTS_DIR/setup.log"

echo ""
echo "=== Running SIFT 100K benchmark ==="
$SSH '
  export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
  export GOPATH=$HOME/go
  cd /home/ubuntu/opaque
  nproc
  free -h | head -2
  echo "--- TestPQ_SIFT100K ---"
  go test -tags sift1m -count=1 -v -run "^TestPQ_SIFT100K$" ./test/ -timeout 30m
' 2>&1 | tee "$RESULTS_DIR/bench.log"

$SSH 'uname -a; lscpu | head -20; free -h' 2>&1 | tee "$RESULTS_DIR/system.log"

echo ""
echo "=== Done — results in $RESULTS_DIR ==="
