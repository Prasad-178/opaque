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
curl -fSL "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" -o "$TARBALL"

echo "Extracting..."
tar xzf "$TARBALL" -C "$DATA_DIR" --strip-components=1

rm -f "$TARBALL"

echo "SIFT1M dataset ready at $DATA_DIR"
ls -lh "$DATA_DIR"
