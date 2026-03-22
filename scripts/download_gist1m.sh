#!/usr/bin/env bash
set -euo pipefail

# Download and extract the GIST1M dataset (~260MB compressed, ~4GB extracted).
# Source: http://corpus-texmex.irisa.fr/
#
# Files extracted to data/gist/:
#   gist_base.fvecs          - 1M base vectors (960-dim float32)
#   gist_query.fvecs         - 1K query vectors
#   gist_groundtruth.ivecs   - Ground truth nearest neighbors
#   gist_learn.fvecs         - Learning set (not used by benchmarks)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/gist"
MARKER_FILE="$DATA_DIR/gist_base.fvecs"

if [ -f "$MARKER_FILE" ]; then
    echo "GIST1M dataset already exists at $DATA_DIR"
    echo "To re-download, remove $DATA_DIR and run again."
    exit 0
fi

echo "Downloading GIST1M dataset (~260MB)..."
mkdir -p "$DATA_DIR"

TARBALL="$DATA_DIR/gist.tar.gz"
curl -fSL "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" -o "$TARBALL"

echo "Extracting..."
tar xzf "$TARBALL" -C "$DATA_DIR" --strip-components=1

rm -f "$TARBALL"

echo "GIST1M dataset ready at $DATA_DIR"
ls -lh "$DATA_DIR"
