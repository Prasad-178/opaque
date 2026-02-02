#!/usr/bin/env python3
"""
Benchmark: Parallel vs Sequential PHE Operations.

Measures the performance impact of parallelization on:
1. Encrypted dot product computation (server-side)
2. Score decryption (client-side)

This helps determine optimal worker counts and chunk sizes.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import multiprocessing as mp

# Add src to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opaque.client.crypto import CryptoClient, parallel_decrypt_multiprocess
from opaque.server.compute import ComputeEngine, VectorStore
from opaque.shared.utils import Timer, generate_random_vectors


def benchmark_parallel_operations(
    num_vectors: int = 50,
    dimension: int = 64,
    key_size: int = 2048,
    max_workers: int = None,
):
    """
    Benchmark parallel vs sequential PHE operations.
    """
    if max_workers is None:
        max_workers = mp.cpu_count()

    print("=" * 70)
    print("Project Opaque - Parallelization Benchmark")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Vectors:       {num_vectors}")
    print(f"  Dimension:     {dimension}")
    print(f"  Key size:      {key_size} bits")
    print(f"  Max workers:   {max_workers}")
    print(f"  CPU count:     {mp.cpu_count()}")

    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n" + "=" * 70)
    print("SETUP")
    print("=" * 70)

    # Generate vectors
    print("\n[1] Generating vectors...")
    vectors = generate_random_vectors(num_vectors, dimension, seed=42)
    query = generate_random_vectors(1, dimension, seed=999)[0]

    # Create stores
    vector_store = VectorStore(
        vectors=vectors,
        ids=[f"vec_{i}" for i in range(num_vectors)],
    )
    compute_engine = ComputeEngine(vector_store)

    # Generate keys
    print("[2] Generating Paillier keys...")
    with Timer() as t:
        crypto = CryptoClient(key_size=key_size, precision=5)
    print(f"    Key generation: {t.elapsed_ms:.0f}ms")

    # Encrypt query
    print("[3] Encrypting query vector...")
    with Timer() as t:
        encrypted_query = crypto.encrypt_vector(query, silent=True)
    print(f"    Encryption: {t.elapsed_ms:.0f}ms")

    # =========================================================================
    # BENCHMARK: Server-side Dot Products
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK: Encrypted Dot Products (Server-side)")
    print("=" * 70)

    # Sequential
    print("\n[Sequential]")
    with Timer() as t:
        encrypted_scores_seq, _, _ = compute_engine.compute_encrypted_scores(
            encrypted_query, verbose=False
        )
    seq_compute_time = t.elapsed_ms
    print(f"  Time: {seq_compute_time:.0f}ms")
    print(f"  Rate: {num_vectors / (seq_compute_time/1000):.1f} vectors/sec")

    # Parallel with different worker counts
    print("\n[Parallel - varying workers]")
    print(f"  {'Workers':<10} {'Time (ms)':<12} {'Speedup':<10} {'Rate (v/s)':<12}")
    print("  " + "-" * 44)

    worker_counts = [2, 4, 8, max_workers] if max_workers > 8 else list(range(2, max_workers + 1))
    worker_counts = sorted(set(worker_counts))

    best_compute_speedup = 1.0
    best_compute_workers = 1

    for n_workers in worker_counts:
        with Timer() as t:
            encrypted_scores_par, _, _ = compute_engine.compute_encrypted_scores_parallel(
                encrypted_query, num_workers=n_workers, verbose=False
            )
        par_time = t.elapsed_ms
        speedup = seq_compute_time / par_time
        rate = num_vectors / (par_time / 1000)
        print(f"  {n_workers:<10} {par_time:<12.0f} {speedup:<10.2f}x {rate:<12.1f}")

        if speedup > best_compute_speedup:
            best_compute_speedup = speedup
            best_compute_workers = n_workers

    # =========================================================================
    # BENCHMARK: Client-side Decryption
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK: Score Decryption (Client-side)")
    print("=" * 70)

    # Sequential decryption
    print("\n[Sequential]")
    with Timer() as t:
        scores_seq = crypto.decrypt_scores(encrypted_scores_seq)
    seq_decrypt_time = t.elapsed_ms
    print(f"  Time: {seq_decrypt_time:.0f}ms")
    print(f"  Rate: {num_vectors / (seq_decrypt_time/1000):.1f} scores/sec")

    # Parallel decryption with different worker counts
    print("\n[Parallel - varying workers]")
    print(f"  {'Workers':<10} {'Time (ms)':<12} {'Speedup':<10} {'Rate (s/s)':<12}")
    print("  " + "-" * 44)

    best_decrypt_speedup = 1.0
    best_decrypt_workers = 1

    for n_workers in worker_counts:
        with Timer() as t:
            scores_par = crypto.decrypt_scores(
                encrypted_scores_seq, parallel=True, num_workers=n_workers
            )
        par_time = t.elapsed_ms
        speedup = seq_decrypt_time / par_time
        rate = num_vectors / (par_time / 1000)
        print(f"  {n_workers:<10} {par_time:<12.0f} {speedup:<10.2f}x {rate:<12.1f}")

        if speedup > best_decrypt_speedup:
            best_decrypt_speedup = speedup
            best_decrypt_workers = n_workers

    # Verify correctness
    print("\n[Verification]")
    scores_seq = np.array(scores_seq)
    scores_par = np.array(scores_par)
    if np.allclose(scores_seq, scores_par, rtol=1e-4):
        print("  Sequential and parallel results match!")
    else:
        print("  WARNING: Results differ!")
        print(f"    Max diff: {np.max(np.abs(scores_seq - scores_par))}")

    # =========================================================================
    # BENCHMARK: Chunked Decryption
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK: Chunked Decryption")
    print("=" * 70)

    chunk_sizes = [5, 10, 20, 50] if num_vectors >= 50 else [5, 10]
    print(f"\n  {'Chunk Size':<12} {'Time (ms)':<12} {'Speedup':<10}")
    print("  " + "-" * 34)

    for chunk_size in chunk_sizes:
        with Timer() as t:
            scores_chunked = crypto.decrypt_scores_chunked(
                encrypted_scores_seq, chunk_size=chunk_size
            )
        chunked_time = t.elapsed_ms
        speedup = seq_decrypt_time / chunked_time
        print(f"  {chunk_size:<12} {chunked_time:<12.0f} {speedup:<10.2f}x")

    # =========================================================================
    # BENCHMARK: Multiprocessing Decryption (True parallelism)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK: Multiprocessing Decryption (True parallelism)")
    print("=" * 70)

    # Get keys for multiprocessing
    keys = crypto.get_keys_dict()

    print(f"\n[Multiprocessing - varying workers]")
    print(f"  {'Workers':<10} {'Time (ms)':<12} {'Speedup':<10} {'Rate (s/s)':<12}")
    print("  " + "-" * 44)

    best_mp_speedup = 1.0
    best_mp_workers = 1

    for n_workers in worker_counts:
        try:
            with Timer() as t:
                scores_mp = parallel_decrypt_multiprocess(
                    encrypted_scores_seq,
                    keys=keys,
                    precision=5,
                    num_workers=n_workers,
                    chunk_size=max(1, num_vectors // n_workers),
                )
            mp_time = t.elapsed_ms
            speedup = seq_decrypt_time / mp_time
            rate = num_vectors / (mp_time / 1000)
            print(f"  {n_workers:<10} {mp_time:<12.0f} {speedup:<10.2f}x {rate:<12.1f}")

            if speedup > best_mp_speedup:
                best_mp_speedup = speedup
                best_mp_workers = n_workers
        except Exception as e:
            print(f"  {n_workers:<10} ERROR: {e}")

    # Verify multiprocessing results
    if scores_mp:
        scores_mp_arr = np.array(scores_mp)
        if np.allclose(scores_seq, scores_mp_arr, rtol=1e-4):
            print("\n  Multiprocessing results match sequential!")
        else:
            print(f"\n  WARNING: Max diff: {np.max(np.abs(scores_seq - scores_mp_arr))}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nOptimal Configuration:")
    print(f"  Dot products:        {best_compute_workers} workers ({best_compute_speedup:.2f}x speedup)")
    print(f"  Decryption (thread): {best_decrypt_workers} workers ({best_decrypt_speedup:.2f}x speedup)")
    print(f"  Decryption (mp):     {best_mp_workers} workers ({best_mp_speedup:.2f}x speedup)")

    print(f"\nExpected Total Speedup:")
    total_seq = seq_compute_time + seq_decrypt_time
    total_par = (seq_compute_time / best_compute_speedup) + (seq_decrypt_time / best_decrypt_speedup)
    overall_speedup = total_seq / total_par
    print(f"  Sequential total: {total_seq:.0f}ms")
    print(f"  Parallel total:   {total_par:.0f}ms")
    print(f"  Overall speedup:  {overall_speedup:.2f}x")

    # Performance notes
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("""
  - ThreadPoolExecutor is used because Python GIL is released during
    bignum operations (which LightPHE uses extensively)
  - ProcessPoolExecutor cannot be used easily because LightPHE
    cryptosystem objects are not pickleable
  - Chunked processing can reduce thread creation overhead
  - Optimal worker count often equals CPU count but may vary
  - Speedup is limited by Python overhead and memory bandwidth
""")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel PHE operations"
    )
    parser.add_argument(
        "--num-vectors", "-n",
        type=int,
        default=50,
        help="Number of vectors to process",
    )
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        default=64,
        help="Vector dimension",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=2048,
        help="Paillier key size in bits",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker count to test",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (20 vectors, 32 dim)",
    )

    args = parser.parse_args()

    if args.quick:
        num_vectors = 20
        dimension = 32
    else:
        num_vectors = args.num_vectors
        dimension = args.dimension

    benchmark_parallel_operations(
        num_vectors=num_vectors,
        dimension=dimension,
        key_size=args.key_size,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
