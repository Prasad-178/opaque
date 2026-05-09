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

python3 -m pip install --quiet --user --upgrade pyarrow huggingface-hub
export PATH="$HOME/.local/bin:$PATH"

echo "Downloading DBpedia-OpenAI-1M parquet shards from HuggingFace..."
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KShivendu/dbpedia-entities-openai-1M",
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["*.parquet"],
)
PYEOF

echo "Converting parquet → fvecs..."
python3 - <<'PYEOF'
import glob
import os
import struct
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

files = sorted(glob.glob("**/*.parquet", recursive=True))
if not files:
    raise SystemExit("No parquet files found after download")

print(f"Reading {len(files)} parquet shard(s)...")
tables = [pq.read_table(f) for f in files]
table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
print(f"Total rows: {table.num_rows}")
print(f"Columns:    {table.column_names}")

# Locate embedding column. KShivendu's repo uses 'openai' as of writing,
# but stay flexible across forks.
emb_col = None
for cand in ("openai", "embedding", "embeddings", "vector", "text-embedding-ada-002"):
    if cand in table.column_names:
        emb_col = cand
        break
if emb_col is None:
    raise SystemExit(f"No embedding column found. Available: {table.column_names}")
print(f"Using embedding column: {emb_col!r}")

raw = table.column(emb_col).to_pylist()
embs = np.asarray(raw, dtype=np.float32)
print(f"Shape: {embs.shape}, dtype: {embs.dtype}")
if embs.ndim != 2:
    raise SystemExit(f"Expected 2D array, got shape {embs.shape}")

n_total, dim = embs.shape
n_query = min(1000, n_total // 100)
n_base = n_total - n_query

base = embs[:n_base]
query = embs[n_base:]
print(f"Split: {n_base} base + {n_query} query")


def write_fvecs(path, mat):
    dim = mat.shape[1]
    with open(path, "wb") as f:
        for v in mat:
            f.write(struct.pack("<i", dim))
            f.write(v.astype(np.float32, copy=False).tobytes())
    print(f"Wrote {path} ({mat.shape[0]} × {dim})")


write_fvecs("dbpedia_base.fvecs", base)
write_fvecs("dbpedia_query.fvecs", query)
PYEOF

echo "Removing parquet shards (no longer needed)..."
find . -name "*.parquet" -delete
rm -rf .cache 2>/dev/null || true

echo "DBpedia1M ready at $DATA_DIR"
ls -lh "$DATA_DIR"
