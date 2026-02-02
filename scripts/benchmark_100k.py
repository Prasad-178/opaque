#!/usr/bin/env python3
"""
Benchmark 100K vectors - comparable to Go benchmark.
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opaque.client.crypto import CryptoClient

# Import LSH directly to avoid server dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "index",
    Path(__file__).resolve().parent.parent / "src" / "opaque" / "server" / "index.py"
)
index_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index_module)
RandomProjectionLSH = index_module.RandomProjectionLSH


def generate_vectors(n: int, dim: int) -> np.ndarray:
    """Generate n normalized random vectors."""
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def main():
    np.random.seed(42)

    NUM_VECTORS = 100000
    DIMENSION = 128
    NUM_QUERIES = 3
    TOP_K = 20
    MAX_CANDIDATES = 100

    print("=" * 60)
    print(f"Python Large Scale Benchmark: {NUM_VECTORS} vectors, {DIMENSION}D")
    print("=" * 60)

    # 1. Generate vectors
    print(f"\n[1/6] Generating {NUM_VECTORS} vectors...")
    start = time.time()
    vectors = generate_vectors(NUM_VECTORS, DIMENSION)
    ids = [f"doc_{i}" for i in range(NUM_VECTORS)]
    print(f"   Generated in {time.time() - start:.3f}s")

    # 2. Build LSH index
    print("\n[2/6] Building LSH index...")
    start = time.time()
    lsh_index = RandomProjectionLSH(dimension=DIMENSION, nbits=128, seed=42)
    lsh_index.add(vectors, ids)
    index_time = time.time() - start
    print(f"   Built index in {index_time:.3f}s")
    print(f"   Vectors: {lsh_index.ntotal}")

    # 3. Initialize crypto client
    print("\n[3/6] Initializing crypto client...")
    start = time.time()
    client = CryptoClient(key_size=2048, precision=5)
    print(f"   Client initialized in {time.time() - start:.3f}s")

    # 4. Benchmark LSH search alone
    print("\n[4/6] Benchmarking LSH search...")
    lsh_times = []
    for i in range(NUM_QUERIES):
        query = vectors[np.random.randint(NUM_VECTORS)]
        start = time.time()
        candidate_ids, distances = lsh_index.search(query, MAX_CANDIDATES)
        lsh_time = time.time() - start
        lsh_times.append(lsh_time)
        print(f"   Query {i+1}: Found {len(candidate_ids)} candidates in {lsh_time*1000:.2f}ms")
    avg_lsh = np.mean(lsh_times)
    print(f"   Average LSH search time: {avg_lsh*1000:.2f}ms")

    # 5. Full search pipeline
    print("\n[5/6] Benchmarking full search pipeline...")
    print("   (This will take a while due to encryption/decryption...)")

    results = []

    for i in range(NUM_QUERIES):
        target_idx = np.random.randint(NUM_VECTORS)
        query = vectors[target_idx] + np.random.randn(DIMENSION).astype(np.float32) * 0.1
        query = query / np.linalg.norm(query)

        result = {
            'query_idx': target_idx,
            'lsh_time': 0,
            'encrypt_time': 0,
            'dot_product_time': 0,
            'decrypt_time': 0,
            'total_time': 0,
            'num_candidates': 0,
        }

        total_start = time.time()

        # LSH search
        lsh_start = time.time()
        candidate_ids, _ = lsh_index.search(query, MAX_CANDIDATES)
        result['lsh_time'] = time.time() - lsh_start
        result['num_candidates'] = len(candidate_ids)

        # Encryption (shift to positive values for LightPHE)
        enc_start = time.time()
        query_shifted = (query + 1.0).tolist()  # Shift to [0, 2] range
        encrypted_query = client.encrypt_vector(query_shifted, silent=True)
        result['encrypt_time'] = time.time() - enc_start

        # Homomorphic dot products (only TOP_K)
        # Note: LightPHE requires positive values, so we shift by adding offset
        num_to_score = min(TOP_K, len(candidate_ids))
        dot_start = time.time()
        encrypted_scores = []
        for j in range(num_to_score):
            idx = ids.index(candidate_ids[j])
            # Shift vector values to be positive (add 1 to get [0, 2] range)
            vec = (vectors[idx] + 1.0).tolist()
            encrypted_score = encrypted_query @ vec
            encrypted_scores.append(encrypted_score)
        result['dot_product_time'] = time.time() - dot_start

        # Decryption
        dec_start = time.time()
        scores = [client.decrypt_score(s) for s in encrypted_scores]
        result['decrypt_time'] = time.time() - dec_start

        result['total_time'] = time.time() - total_start
        results.append(result)

        print(f"\n   Query {i+1} (similar to doc_{target_idx}):")
        print(f"      LSH:          {result['lsh_time']*1000:.2f}ms ({result['num_candidates']} candidates)")
        print(f"      Encryption:   {result['encrypt_time']*1000:.2f}ms")
        print(f"      {num_to_score}× Dot Prod:  {result['dot_product_time']*1000:.2f}ms")
        print(f"      {num_to_score}× Decrypt:   {result['decrypt_time']*1000:.2f}ms")
        print(f"      Total:        {result['total_time']*1000:.2f}ms")

    # 6. Summary
    print("\n" + "=" * 60)
    print("[6/6] Summary")
    print("=" * 60)

    avg_lsh = np.mean([r['lsh_time'] for r in results])
    avg_enc = np.mean([r['encrypt_time'] for r in results])
    avg_dot = np.mean([r['dot_product_time'] for r in results])
    avg_dec = np.mean([r['decrypt_time'] for r in results])
    avg_total = np.mean([r['total_time'] for r in results])

    print(f"\nAverage times per query:")
    print(f"   LSH Search:      {avg_lsh*1000:.2f}ms")
    print(f"   Encryption:      {avg_enc*1000:.2f}ms")
    print(f"   {TOP_K}× Dot Products: {avg_dot*1000:.2f}ms")
    print(f"   {TOP_K}× Decryption:   {avg_dec*1000:.2f}ms")
    print(f"   Total:           {avg_total*1000:.2f}ms")
    print(f"\nEstimated QPS:      {1/avg_total:.4f}")
    print(f"Dataset size:       {NUM_VECTORS} vectors")


if __name__ == "__main__":
    main()
