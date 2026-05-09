#!/usr/bin/env bash
set -euo pipefail

# Download and extract the SIFT1M dataset (~170MB compressed, ~500MB extracted).
# Source: http://corpus-texmex.irisa.fr/
#
# Files extracted to data/sift/:
#   sift_base.fvecs          - 1M base vectors (128-dim float32)
#   sift_query.fvecs         - 10K query vectors
#   sift_groundtruth.ivecs   - Ground truth nearest neighbors
#   sift_learn.fvecs         - Learning set (not used by benchmarks)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/sift"
MARKER_FILE="$DATA_DIR/sift_base.fvecs"

if [ -f "$MARKER_FILE" ]; then
    echo "SIFT1M dataset already exists at $DATA_DIR"
    echo "To re-download, remove $DATA_DIR and run again."
    exit 0
fi

echo "Downloading SIFT1M dataset..."
mkdir -p "$DATA_DIR"

TARBALL="$DATA_DIR/sift.tar.gz"
# IRISA's FTP server occasionally drops mid-stream during the 161 MB
# transfer (curl exits 56 = "response reading failed"). Auto-retry up
# to 5 times with backoff so a single network blip doesn't kill the
# whole bench setup.
curl -fSL --retry 5 --retry-delay 10 --retry-all-errors \
  --connect-timeout 30 --max-time 600 \
  "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" -o "$TARBALL"

echo "Extracting..."
tar xzf "$TARBALL" -C "$DATA_DIR" --strip-components=1

rm -f "$TARBALL"

echo "SIFT1M dataset ready at $DATA_DIR"
ls -lh "$DATA_DIR"
