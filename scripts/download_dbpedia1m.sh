#!/usr/bin/env bash
set -euo pipefail

# Download DBpedia-OpenAI-1M (1M Wikipedia entity descriptions × 1536-dim ada-002
# embeddings) from HuggingFace, then convert the parquet shards to fvecs format
# consumed by pkg/embeddings.DBpedia1M.
#
# Source: https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M
#
# Output (data/dbpedia/):
#   dbpedia_base.fvecs   — first 999K vectors (1536-dim float32)
#   dbpedia_query.fvecs  — last 1K vectors held out as queries
#
# Disk usage: ~12 GB peak (parquet + fvecs simultaneously), ~6 GB after cleanup.
# Runtime: dominated by download (~2-5 min on a fast link, faster on EC2).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/dbpedia"
MARKER_FILE="$DATA_DIR/dbpedia_base.fvecs"

if [ -f "$MARKER_FILE" ] && [ -f "$DATA_DIR/dbpedia_query.fvecs" ]; then
    echo "DBpedia1M dataset already exists at $DATA_DIR"
    echo "To re-download, remove $DATA_DIR and run again."
    exit 0
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Ensuring Python + pyarrow + huggingface-hub are available..."
if ! command -v python3 > /dev/null; then
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq python3 python3-pip || true
fi

python3 -m pip install --quiet --user --upgrade numpy pyarrow huggingface-hub
export PATH="$HOME/.local/bin:$PATH"

echo "Downloading DBpedia-OpenAI-1M parquet shards from HuggingFace..."
# HF_TOKEN env var should be set by the caller — AWS IPs get aggressive
# rate-limiting from HF when unauthenticated and snapshot_download hangs
# indefinitely at 0%. With a token, downloads complete in minutes.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN not set. AWS IPs are heavily rate-limited at HF Hub."
  echo "         Download may hang indefinitely. Set HF_TOKEN env var to fix."
fi
python3 - <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
print(f"HF_TOKEN: {'present (auth)' if token else 'missing (unauth, may rate-limit)'}", flush=True)

snapshot_download(
    repo_id="KShivendu/dbpedia-entities-openai-1M",
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["*.parquet"],
    token=token,
    max_workers=4,
    etag_timeout=30,
)
PYEOF

echo "Converting parquet → fvecs (streaming, shard by shard)..."
python3 - <<'PYEOF'
# Streaming converter — reads each parquet shard one at a time and appends to
# the output fvecs files, never holding more than one shard in memory at once.
# A naive concat_tables + to_pylist of all 1M × 1536 floats blows past 64 GB
# RAM (Python list overhead is ~10x raw size on FixedSizeList<float>).
import glob
import os
import struct
import sys

import numpy as np
import pyarrow.parquet as pq

files = sorted(glob.glob("**/*.parquet", recursive=True))
if not files:
    raise SystemExit("No parquet files found after download")
print(f"Reading {len(files)} parquet shard(s)...", flush=True)

# Pass 1: count rows + locate embedding column from the first shard.
total_rows = 0
emb_col = None
for f in files:
    md = pq.read_metadata(f)
    total_rows += md.num_rows
    if emb_col is None:
        cols = md.schema.to_arrow_schema().names
        for cand in ("openai", "embedding", "embeddings", "vector", "text-embedding-ada-002"):
            if cand in cols:
                emb_col = cand
                break
        if emb_col is None:
            raise SystemExit(f"No embedding column found. Available: {cols}")

print(f"Total rows: {total_rows} | embedding column: {emb_col!r}", flush=True)

# Hold out last 1000 rows as the query set, rest is base.
n_query = min(1000, total_rows // 100)
n_base = total_rows - n_query
print(f"Split: {n_base} base + {n_query} query", flush=True)

base_path = "dbpedia_base.fvecs"
query_path = "dbpedia_query.fvecs"
base_f = open(base_path, "wb")
query_f = open(query_path, "wb")

written = 0
dim = None
try:
    for fi, f in enumerate(files, 1):
        # to_pylist() materialises this shard's rows but only this one shard.
        # Each shard is ~38K × 1536 floats ≈ 230 MB raw, ~2.3 GB Python list peak.
        col = pq.read_table(f, columns=[emb_col]).column(emb_col)
        chunk = np.asarray(col.to_pylist(), dtype=np.float32)
        if chunk.ndim != 2:
            raise SystemExit(f"Expected 2D shard, got shape {chunk.shape}")
        if dim is None:
            dim = chunk.shape[1]
        elif chunk.shape[1] != dim:
            raise SystemExit(f"Dim mismatch: {chunk.shape[1]} vs {dim}")
        # Append rows to base until we hit n_base; remainder go to query.
        for row in chunk:
            target = base_f if written < n_base else query_f
            target.write(struct.pack("<i", dim))
            target.write(row.tobytes())
            written += 1
        del chunk, col
        print(f"  shard {fi}/{len(files)}: written={written}/{total_rows}", flush=True)
finally:
    base_f.close()
    query_f.close()

base_size = os.path.getsize(base_path)
query_size = os.path.getsize(query_path)
print(f"Wrote {base_path}: {n_base} × {dim} ({base_size/1e9:.2f} GB)", flush=True)
print(f"Wrote {query_path}: {n_query} × {dim} ({query_size/1e6:.2f} MB)", flush=True)
PYEOF

echo "Removing parquet shards (no longer needed)..."
find . -name "*.parquet" -delete
rm -rf .cache 2>/dev/null || true

echo "DBpedia1M ready at $DATA_DIR"
ls -lh "$DATA_DIR"
