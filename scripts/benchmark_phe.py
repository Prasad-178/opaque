#!/usr/bin/env python3
"""
Phase 1: Benchmark LightPHE performance.

This script measures:
1. Key generation time
2. Vector encryption time (various dimensions)
3. Encrypted dot product time (1 query vs N database vectors)
4. Decryption time

Decision point: If 1000 dot products with 1536-dim vectors takes >10s,
we need PCA dimensionality reduction.
"""
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opaque.client.crypto import CryptoClient
from opaque.shared.utils import (
    generate_random_vectors,
    compute_plaintext_similarity,
    Timer,
)
from opaque.shared.protocol import BenchmarkResult


def benchmark_key_generation(key_size: int = 2048, num_trials: int = 3) -> BenchmarkResult:
    """Benchmark Paillier key generation."""
    times = []

    for i in range(num_trials):
        print(f"  Key generation trial {i+1}/{num_trials}...")
        with Timer() as t:
            _ = CryptoClient(key_size=key_size)
        times.append(t.elapsed)

    avg_time = np.mean(times)
    return BenchmarkResult(
        operation="key_generation",
        dimension=0,
        num_operations=num_trials,
        total_time_seconds=sum(times),
        avg_time_per_op_ms=avg_time * 1000,
        throughput_ops_per_sec=1 / avg_time if avg_time > 0 else 0,
        notes=f"key_size={key_size} bits"
    )


def benchmark_encryption(
    client: CryptoClient,
    dimension: int,
    num_trials: int = 5,
) -> BenchmarkResult:
    """Benchmark vector encryption."""
    vectors = generate_random_vectors(num_trials, dimension, normalize=True)
    times = []

    for i in range(num_trials):
        vec = vectors[i].tolist()
        with Timer() as t:
            _ = client.encrypt_vector(vec, silent=True)
        times.append(t.elapsed)
        print(f"  Encryption trial {i+1}/{num_trials}: {t.elapsed*1000:.2f}ms")

    avg_time = np.mean(times)
    return BenchmarkResult(
        operation="vector_encryption",
        dimension=dimension,
        num_operations=num_trials,
        total_time_seconds=sum(times),
        avg_time_per_op_ms=avg_time * 1000,
        throughput_ops_per_sec=1 / avg_time if avg_time > 0 else 0,
        notes=f"precision={client.precision}"
    )


def benchmark_dot_product(
    client: CryptoClient,
    dimension: int,
    num_db_vectors: int = 1000,
) -> Tuple[BenchmarkResult, float]:
    """
    Benchmark encrypted dot product: 1 encrypted query vs N plaintext vectors.

    Returns:
        Tuple of (benchmark result, max accuracy error)
    """
    # Generate query and database vectors
    query = generate_random_vectors(1, dimension, normalize=True)[0]
    db_vectors = generate_random_vectors(num_db_vectors, dimension, normalize=True)

    # Encrypt query
    print(f"  Encrypting {dimension}-dim query vector...")
    with Timer() as encrypt_timer:
        encrypted_query = client.encrypt_vector(query, silent=True)
    print(f"  Encryption took {encrypt_timer.elapsed_ms:.2f}ms")

    # Compute encrypted dot products
    print(f"  Computing {num_db_vectors} encrypted dot products...")
    encrypted_scores = []
    with Timer() as dot_timer:
        for i, db_vec in enumerate(db_vectors):
            # LightPHE's @ operator: encrypted_vector @ plaintext_vector
            encrypted_score = encrypted_query @ db_vec.tolist()
            encrypted_scores.append(encrypted_score)

            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{num_db_vectors} vectors...")

    # Decrypt and verify
    print(f"  Decrypting scores...")
    with Timer() as decrypt_timer:
        decrypted_scores = [client.decrypt_score(s) for s in encrypted_scores]

    # Verify accuracy
    plaintext_scores = compute_plaintext_similarity(query, db_vectors)
    max_error = np.max(np.abs(np.array(decrypted_scores) - plaintext_scores))

    result = BenchmarkResult(
        operation="encrypted_dot_product",
        dimension=dimension,
        num_operations=num_db_vectors,
        total_time_seconds=dot_timer.elapsed,
        avg_time_per_op_ms=(dot_timer.elapsed / num_db_vectors) * 1000,
        throughput_ops_per_sec=num_db_vectors / dot_timer.elapsed if dot_timer.elapsed > 0 else 0,
        notes=(
            f"encrypt_time={encrypt_timer.elapsed_ms:.2f}ms, "
            f"decrypt_time={decrypt_timer.elapsed_ms:.2f}ms, "
            f"max_error={max_error:.6f}"
        )
    )

    return result, max_error


def benchmark_decryption(
    client: CryptoClient,
    dimension: int,
    num_scores: int = 100,
) -> BenchmarkResult:
    """Benchmark score decryption."""
    # Create a dummy encrypted score by encrypting a small vector
    query = generate_random_vectors(1, dimension, normalize=True)[0]
    db_vec = generate_random_vectors(1, dimension, normalize=True)[0]

    encrypted_query = client.encrypt_vector(query, silent=True)
    encrypted_score = encrypted_query @ db_vec.tolist()

    # Time decryption
    with Timer() as t:
        for _ in range(num_scores):
            _ = client.decrypt_score(encrypted_score)

    return BenchmarkResult(
        operation="score_decryption",
        dimension=dimension,
        num_operations=num_scores,
        total_time_seconds=t.elapsed,
        avg_time_per_op_ms=(t.elapsed / num_scores) * 1000,
        throughput_ops_per_sec=num_scores / t.elapsed if t.elapsed > 0 else 0,
    )


def run_full_benchmark(
    dimensions: List[int] = [128, 256, 512, 1536],
    num_db_vectors: int = 1000,
    key_size: int = 2048,
) -> List[BenchmarkResult]:
    """Run complete benchmark suite."""
    print("=" * 60)
    print("Project Opaque - Phase 1: LightPHE Benchmark")
    print("=" * 60)

    results = []

    # 1. Key generation
    print("\n[1/4] Benchmarking key generation...")
    keygen_result = benchmark_key_generation(key_size=key_size)
    results.append(keygen_result)
    print(keygen_result)

    # Create client for remaining tests
    print("\nCreating crypto client for benchmarks...")
    client = CryptoClient(key_size=key_size)

    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"Testing dimension: {dim}")
        print("=" * 60)

        # 2. Encryption
        print(f"\n[2/4] Benchmarking {dim}-dim encryption...")
        enc_result = benchmark_encryption(client, dim, num_trials=3)
        results.append(enc_result)
        print(enc_result)

        # 3. Dot product (the critical benchmark)
        print(f"\n[3/4] Benchmarking {dim}-dim dot product ({num_db_vectors} vectors)...")
        dot_result, max_error = benchmark_dot_product(client, dim, num_db_vectors)
        results.append(dot_result)
        print(dot_result)

        # 4. Decryption
        print(f"\n[4/4] Benchmarking {dim}-dim decryption...")
        dec_result = benchmark_decryption(client, dim, num_scores=100)
        results.append(dec_result)
        print(dec_result)

    # Summary and decision
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    dot_results = [r for r in results if r.operation == "encrypted_dot_product"]

    print("\nDot Product Performance by Dimension:")
    print("-" * 50)
    for r in dot_results:
        status = "PASS" if r.total_time_seconds < 10 else "NEEDS_PCA"
        print(f"  {r.dimension:4d}D: {r.total_time_seconds:8.2f}s ({r.num_operations} ops) [{status}]")

    # Decision point
    dim_1536_result = next((r for r in dot_results if r.dimension == 1536), None)
    if dim_1536_result:
        if dim_1536_result.total_time_seconds > 10:
            print(f"\n*** DECISION: 1536-dim takes {dim_1536_result.total_time_seconds:.2f}s > 10s")
            print("*** RECOMMENDATION: Use PCA to reduce dimensionality to 256 or 128")

            # Find best dimension under threshold
            for r in sorted(dot_results, key=lambda x: x.dimension, reverse=True):
                if r.total_time_seconds < 10:
                    print(f"*** SUGGESTED DIMENSION: {r.dimension}")
                    break
        else:
            print(f"\n*** DECISION: 1536-dim takes {dim_1536_result.total_time_seconds:.2f}s < 10s")
            print("*** RECOMMENDATION: No PCA needed, proceed with full dimension")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LightPHE for Project Opaque")
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1536],
        help="Dimensions to test",
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=1000,
        help="Number of database vectors for dot product test",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=2048,
        choices=[1024, 2048],
        help="Paillier key size in bits",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test only 128 and 256 dimensions with 100 vectors",
    )

    args = parser.parse_args()

    if args.quick:
        dimensions = [128, 256]
        num_vectors = 100
    else:
        dimensions = args.dimensions
        num_vectors = args.num_vectors

    run_full_benchmark(
        dimensions=dimensions,
        num_db_vectors=num_vectors,
        key_size=args.key_size,
    )


if __name__ == "__main__":
    main()
