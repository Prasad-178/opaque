#!/usr/bin/env bash
set -euo pipefail

# Download SIFT 1M dataset from ann-benchmarks.com mirror (HDF5 format,
# ~501 MB), convert to the IRISA-style fvecs/ivecs format the rest of the
# codebase expects.
#
# Switched from ftp.irisa.fr → ann-benchmarks.com on 2026-05-10 after
# IRISA's FTP server had a multi-hour outage that exhausted curl-retry
# (6 attempts spanning ~3 minutes all failed mid-stream / timeout).
# ann-benchmarks.com is HTTPS, supports range requests, and serves the
# same SIFT1M data in the standard ANN-benchmarks HDF5 layout.
#
# Files produced in data/sift/:
#   sift_base.fvecs          - 1M base vectors (128-dim float32)
#   sift_query.fvecs         - 10K query vectors
#   sift_groundtruth.ivecs   - top-100 ground truth neighbors

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/sift"
MARKER_FILE="$DATA_DIR/sift_base.fvecs"
HDF5_URL="http://ann-benchmarks.com/sift-128-euclidean.hdf5"
HDF5_FILE="$DATA_DIR/sift-128-euclidean.hdf5"

if [ -f "$MARKER_FILE" ]; then
    echo "SIFT1M dataset already exists at $DATA_DIR"
    echo "To re-download, remove $DATA_DIR and run again."
    exit 0
fi

echo "Downloading SIFT1M dataset from ann-benchmarks mirror..."
mkdir -p "$DATA_DIR"

curl -fSL --retry 5 --retry-delay 10 --retry-all-errors \
  --connect-timeout 30 --max-time 600 \
  "$HDF5_URL" -o "$HDF5_FILE"

echo "Ensuring Python + h5py + numpy are available..."
if ! command -v python3 > /dev/null; then
  sudo apt-get update -qq || true
  sudo apt-get install -y -qq python3 python3-pip || true
fi
python3 -m pip install --quiet --user --upgrade h5py numpy
export PATH="$HOME/.local/bin:$PATH"

echo "Converting HDF5 → fvecs/ivecs..."
python3 - "$HDF5_FILE" "$DATA_DIR" <<'PYEOF'
import struct
import sys

import h5py
import numpy as np

hdf5_path = sys.argv[1]
out_dir = sys.argv[2]


def write_fvecs(path, mat):
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    n, d = mat.shape
    with open(path, "wb") as f:
        for v in mat:
            f.write(struct.pack("<i", d))
            f.write(v.tobytes())
    print(f"Wrote {path}: {n} × {d} ({n * (4 + 4 * d) / 1e6:.1f} MB)")


def write_ivecs(path, mat):
    mat = np.ascontiguousarray(mat, dtype=np.int32)
    n, d = mat.shape
    with open(path, "wb") as f:
        for v in mat:
            f.write(struct.pack("<i", d))
            f.write(v.tobytes())
    print(f"Wrote {path}: {n} × {d} ({n * (4 + 4 * d) / 1e6:.1f} MB)")


with h5py.File(hdf5_path, "r") as f:
    train = np.asarray(f["train"], dtype=np.float32)
    test = np.asarray(f["test"], dtype=np.float32)
    neighbors = np.asarray(f["neighbors"], dtype=np.int32)
    print(f"train  {train.shape}  {train.dtype}")
    print(f"test   {test.shape}   {test.dtype}")
    print(f"neighbors {neighbors.shape} {neighbors.dtype}")

write_fvecs(f"{out_dir}/sift_base.fvecs", train)
write_fvecs(f"{out_dir}/sift_query.fvecs", test)
write_ivecs(f"{out_dir}/sift_groundtruth.ivecs", neighbors)
PYEOF

echo "Removing intermediate HDF5..."
rm -f "$HDF5_FILE"

echo "SIFT1M dataset ready at $DATA_DIR"
ls -lh "$DATA_DIR"
