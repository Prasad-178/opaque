#!/bin/bash
# Download SIFT10M: first 10M vectors from BigANN learning set.
# Format: bvecs (uint8, 128-dim). Size: ~1.3GB for 10M vectors.
#
# The BigANN learning set (100M vectors, 9.1GB compressed) is streamed
# and only the first 10M vectors are kept.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data/bigann" && pwd)"
OUTPUT="$DATA_DIR/sift10m_base.bvecs"
QUERY_SRC="$(cd "$(dirname "$0")/../data/sift" && pwd)/sift_query.fvecs"

if [ -f "$OUTPUT" ]; then
    echo "SIFT10M already exists at $OUTPUT"
    echo "Size: $(du -h "$OUTPUT" | cut -f1)"
    exit 0
fi

mkdir -p "$DATA_DIR"

echo "=== Downloading SIFT10M (first 10M vectors from BigANN) ==="
echo "Source: ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz"
echo "This downloads ~9.1GB compressed, extracts first 10M vectors (~1.3GB)"
echo ""

# Download and decompress, extracting only first 10M vectors.
# Each bvecs record: 4 bytes (dim as int32) + 128 bytes (uint8 values) = 132 bytes
# 10M vectors = 10,000,000 * 132 = 1,320,000,000 bytes
BYTES_NEEDED=1320000000

echo "Downloading and extracting first 10M vectors..."
curl -# "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz" | \
    gunzip -c | \
    head -c "$BYTES_NEEDED" > "$OUTPUT"

ACTUAL_SIZE=$(wc -c < "$OUTPUT")
EXPECTED_VECTORS=$((ACTUAL_SIZE / 132))

echo ""
echo "Done! Extracted $EXPECTED_VECTORS vectors"
echo "File: $OUTPUT"
echo "Size: $(du -h "$OUTPUT" | cut -f1)"

# Copy query vectors (reuse SIFT1M queries — same distribution)
if [ -f "$QUERY_SRC" ]; then
    cp "$QUERY_SRC" "$DATA_DIR/sift10m_query.fvecs"
    echo "Copied query vectors from SIFT1M"
fi

echo ""
echo "To run benchmarks:"
echo "  go test -tags sift10m -v -run TestPQ_SIFT10M ./test/ -timeout 120m"
