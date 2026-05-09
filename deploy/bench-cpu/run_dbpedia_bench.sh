#!/bin/bash
# Orchestrates the Opaque DBpedia1M (1M × 1536-dim ada-002 text embeddings)
# bench on a fresh EC2 CPU instance. Sibling of run_bench.sh; same flow but
# different setup script and different test invocations.
#
# Default instance: r6i.4xlarge (16 vCPU, 128 GB). The kmeans_builder +
# Storage:File + GOGC=50 optimizations landed in 1e735ec cut the build-phase
# peak from ~128 GB → ~64 GB at 1M × 1536-dim — a 50 % reduction. But the
# remaining peak still hugs 64 GB exactly (raw vectors held twice via
# dataset.Vectors + AddBatch's defensive copy = 24.6 GB plus ciphertext
# accumulator + Go GC slack), so m6i.4xlarge OOMs by single-digit GB.
# r6i.4xlarge gives 2× headroom at slightly cheaper hourly cost than the
# equivalent compute-tier m6i.8xlarge.
#
# Cost: ~$1.01/hr × ~1.5-2 hr ≈ $2.00.
# Destroy runs in an always-on trap so an interrupted run still tears down.
#
# Usage: deploy/bench-cpu/run_dbpedia_bench.sh [r6i.4xlarge]

set -euo pipefail

# Auto-source repo-root .env if present (gitignored, contains HF_TOKEN etc).
SCRIPT_DIR_EARLY="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_EARLY="$(cd "$SCRIPT_DIR_EARLY/../.." && pwd)"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_ROOT_EARLY/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT_EARLY/.env"
  set +a
fi

# Fail fast if HF_TOKEN missing — AWS IPs hit aggressive rate-limiting at
# HF Hub and snapshot_download hangs at 0% indefinitely without auth.
# Past lesson: a 6-hour hang cost ~$5 before this check existed.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "       Either put it in $REPO_ROOT_EARLY/.env (HF_TOKEN=hf_xxx) or"
  echo "       export HF_TOKEN=hf_xxx before re-running."
  echo "       Get a free read-only token at https://huggingface.co/settings/tokens"
  exit 1
fi

INSTANCE_TYPE="${1:-r6i.4xlarge}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KEY_FILE="$SCRIPT_DIR/bench-cpu-key.pem"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/${TIMESTAMP}-${INSTANCE_TYPE}-dbpedia"
BUNDLE="/tmp/opaque-bundle-bench.tar.gz"

mkdir -p "$RESULTS_DIR"

echo "=== Opaque DBpedia1M CPU bench: $INSTANCE_TYPE ==="
echo "Results: $RESULTS_DIR"

# Bundle source (exclude heavy/irrelevant paths). Note: data/ is excluded —
# the EC2 instance downloads + converts DBpedia fresh during setup.
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

# Upload source + DBpedia variant of the setup script.
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no "$BUNDLE" "ubuntu@$IP:/tmp/opaque-bundle.tar.gz"
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no "$SCRIPT_DIR/setup_dbpedia.sh" "ubuntu@$IP:/tmp/bench-setup.sh"

# Run setup (installs Go + Python + downloads/converts DBpedia, ~5-10 min).
# HF_TOKEN is forwarded inline as an env-var assignment on the remote command.
# The EC2 is ephemeral + destroyed at end-of-run so token never persists.
echo ""
echo "=== Installing Go + Python + downloading DBpedia ==="
$SSH "HF_TOKEN='$HF_TOKEN' bash /tmp/bench-setup.sh" 2>&1 | tee "$RESULTS_DIR/setup.log"

# Run DBpedia accuracy + PQ benchmarks.
echo ""
echo "=== Running DBpedia1M benchmarks ==="
$SSH '
  export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
  export GOPATH=$HOME/go
  # GOGC=50 tightens Go GC from default 100 → triggers GC at 50% heap growth.
  # Halves the build-phase memory headroom Go normally keeps for GC overhead.
  # Slight CPU cost (more frequent GC) but build at 1536-dim is RAM-bound.
  export GOGC=50
  cd /home/ubuntu/opaque
  nproc
  free -h | head -2
  echo "--- TestDBpedia1MAccuracy ---"
  go test -tags dbpedia1m -count=1 -v -run "^TestDBpedia1MAccuracy$" ./test/ -timeout 120m
  echo ""
  echo "--- TestPQ_DBpedia1M ---"
  go test -tags dbpedia1m -count=1 -v -run "^TestPQ_DBpedia1M$" ./test/ -timeout 120m
' 2>&1 | tee "$RESULTS_DIR/bench.log"

# Meta.
$SSH 'uname -a; lscpu | head -20; free -h; df -h /' 2>&1 | tee "$RESULTS_DIR/system.log"

echo ""
echo "=== Done — results in $RESULTS_DIR ==="
