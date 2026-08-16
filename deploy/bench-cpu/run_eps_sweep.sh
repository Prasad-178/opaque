#!/bin/bash
# Runs ONLY the Opaque ε-sweep benchmark (access-pattern privacy vs latency,
# the paper's "tunable knob" money figure) on a fresh EC2 CPU instance.
#
# Mirrors run_bench.sh but runs a single test: TestSIFT1M_EpsilonSweep.
# AWS account is PINNED to the `personal` profile on both apply and destroy —
# never touches the realfy or default profiles.
#
# Cost: ~$0.34/hr × ~0.5hr on m6i.2xlarge (no-PQ strict build is light;
# 8 ε points × one 1M build+50 queries each). Destroy runs in an always-on
# trap so an interrupted run still tears down.
#
# Usage: deploy/bench-cpu/run_eps_sweep.sh [instance_type]   # default m6i.2xlarge
set -euo pipefail

INSTANCE_TYPE="${1:-m6i.2xlarge}"
AWS_PROFILE_NAME="personal"   # hard requirement — do not change to realfy/default
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KEY_FILE="$SCRIPT_DIR/bench-cpu-key.pem"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/${TIMESTAMP}-${INSTANCE_TYPE}-epssweep"
BUNDLE="/tmp/opaque-bundle-epssweep.tar.gz"

mkdir -p "$RESULTS_DIR"

echo "=== Opaque ε-SWEEP bench: $INSTANCE_TYPE (AWS profile: $AWS_PROFILE_NAME) ==="
echo "Results: $RESULTS_DIR"

# Sanity: confirm the personal profile resolves to an account before spending.
echo "Verifying AWS identity (profile=$AWS_PROFILE_NAME)..."
aws sts get-caller-identity --profile "$AWS_PROFILE_NAME" --output text --query 'Account' \
  || { echo "ERROR: profile '$AWS_PROFILE_NAME' not usable. Aborting."; exit 1; }

# Bundle source (exclude heavy/irrelevant paths).
echo "Bundling source..."
tar --exclude=.git --exclude=third_party --exclude=data \
    --exclude='*.tfstate*' --exclude='*.pem' --exclude=deploy/gpu/.terraform \
    --exclude=deploy/bench-cpu/.terraform --exclude=deploy/bench-cpu/results \
    -czf "$BUNDLE" -C "$REPO_ROOT" .
echo "Bundle: $(du -h "$BUNDLE" | cut -f1)"

cd "$SCRIPT_DIR"

# Trap teardown on any exit path — always destroys under the personal profile.
cleanup() {
  echo ""
  echo "=== Destroying infra (profile=$AWS_PROFILE_NAME) ==="
  terraform destroy -var="enabled=true" \
    -var="aws_profile=$AWS_PROFILE_NAME" \
    -var="instance_type=$INSTANCE_TYPE" \
    -auto-approve || true
}
trap cleanup EXIT

# Bring up EC2.
terraform init -upgrade > /dev/null
terraform apply -var="enabled=true" \
  -var="aws_profile=$AWS_PROFILE_NAME" \
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

# Run setup (installs Go, downloads SIFT1M).
echo ""
echo "=== Installing Go + downloading SIFT1M ==="
$SSH 'bash /tmp/bench-setup.sh' 2>&1 | tee "$RESULTS_DIR/setup.log"

# Run ONLY the epsilon sweep.
echo ""
echo "=== Running ε-sweep ==="
$SSH '
  export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
  export GOPATH=$HOME/go
  cd /home/ubuntu/opaque
  nproc
  echo "--- TestSIFT1M_EpsilonSweep ---"
  go test -tags sift1m -count=1 -v -run "^TestSIFT1M_EpsilonSweep$" ./test/ -timeout 90m
' 2>&1 | tee "$RESULTS_DIR/eps_sweep.log"

# Meta.
$SSH 'uname -a; lscpu | head -20; free -h' 2>&1 | tee "$RESULTS_DIR/system.log"

# Extract the CSV block for plotting.
echo ""
echo "=== Extracting CSV ==="
# Strip the `    file_test.go:NNN: ` prefix that `go test -v` adds to each t.Log line.
awk '/CSV_BEGIN/{f=1;next} /CSV_END/{f=0} f' "$RESULTS_DIR/eps_sweep.log" \
  | sed -E 's/^.*\.go:[0-9]+: //' > "$RESULTS_DIR/eps_sweep.csv" || true
echo "CSV -> $RESULTS_DIR/eps_sweep.csv"
cat "$RESULTS_DIR/eps_sweep.csv" 2>/dev/null || true

echo ""
echo "=== Done — results in $RESULTS_DIR ==="
