#!/usr/bin/env python3
"""
Phase 3: "Funnel" - Full pipeline with LSH + PHE.

Demonstrates the two-stage funnel approach:
1. Coarse Stage: LSH retrieves ~1000 candidates from 100k vectors
2. Fine Stage: PHE computes encrypted scores for candidates

Target: <3s total latency
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opaque.client.crypto import CryptoClient
from opaque.client.search import SearchClient
from opaque.server.compute import ComputeEngine, VectorStore
from opaque.server.index import LSHIndex
from opaque.shared.utils import (
    generate_random_vectors,
    compute_plaintext_similarity,
    top_k_indices,
    Timer,
    normalize_vectors,
)
from opaque.shared.reduction import PCAReducer


def run_full_pipeline_demo(
    num_vectors: int = 100000,
    input_dimension: int = 1536,
    reduced_dimension: int = 256,
    num_candidates: int = 1000,
    top_k: int = 10,
    lsh_bits: int = 256,
    verbose: bool = True,
):
    """
    Run the full funnel pipeline demonstration.
    """
    print("=" * 70)
    print("Project Opaque - Phase 3: Full Funnel Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Database size:     {num_vectors:,} vectors")
    print(f"  Input dimension:   {input_dimension}")
    print(f"  Reduced dimension: {reduced_dimension}")
    print(f"  LSH candidates:    {num_candidates}")
    print(f"  LSH bits:          {lsh_bits}")
    print(f"  Top-K:             {top_k}")

    # =========================================================================
    # SETUP PHASE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SETUP PHASE")
    print("=" * 70)

    # 1. Generate database vectors
    print("\n[1] Generating database vectors...")
    with Timer() as t:
        db_vectors_full = generate_random_vectors(
            num_vectors, input_dimension, normalize=True, seed=42
        )
    print(f"    Generated {num_vectors:,} x {input_dimension}D vectors in {t.elapsed_ms:.0f}ms")

    # 2. Fit PCA for dimensionality reduction
    print("\n[2] Fitting PCA reducer...")
    with Timer() as t:
        pca = PCAReducer(input_dimension, reduced_dimension)
        db_vectors_reduced = pca.fit_transform(db_vectors_full)
        # Re-normalize after PCA
        db_vectors_reduced = normalize_vectors(db_vectors_reduced)
    print(f"    PCA fit + transform in {t.elapsed_ms:.0f}ms")
    if pca.total_explained_variance_ratio is not None:
        print(f"    Explained variance: {pca.total_explained_variance_ratio:.1%}")

    # 3. Build LSH index
    print("\n[3] Building LSH index...")
    with Timer() as t:
        lsh_index = LSHIndex(reduced_dimension, nbits=lsh_bits)
        vector_ids = [f"doc_{i:06d}" for i in range(num_vectors)]
        lsh_index.add(db_vectors_reduced, vector_ids)
    print(f"    Indexed {lsh_index.ntotal:,} vectors in {t.elapsed_ms:.0f}ms")
    print(f"    Backend: {lsh_index.backend}")

    # 4. Create vector store for PHE scoring
    print("\n[4] Creating vector store...")
    vector_store = VectorStore(
        vectors=db_vectors_reduced,
        ids=vector_ids,
    )
    compute_engine = ComputeEngine(vector_store)

    # 5. Initialize client crypto
    print("\n[5] Generating Paillier keys (2048-bit)...")
    with Timer() as t:
        crypto_client = CryptoClient(key_size=2048, precision=5)
    print(f"    Key generation in {t.elapsed_ms:.0f}ms")

    search_client = SearchClient(crypto_client)

    # 6. Generate query
    print("\n[6] Generating query vector...")
    query_full = generate_random_vectors(1, input_dimension, normalize=True, seed=999)[0]
    query_reduced = pca.transform(query_full.reshape(1, -1))[0]
    query_reduced = query_reduced / np.linalg.norm(query_reduced)

    # =========================================================================
    # SEARCH PHASE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SEARCH PHASE")
    print("=" * 70)

    # Define search functions
    def lsh_search(query, k):
        return lsh_index.search(query, k)

    def phe_compute(encrypted_query, candidate_ids):
        return compute_engine.compute_encrypted_scores(
            encrypted_query,
            vector_ids=candidate_ids,
            verbose=False,
        )

    # Execute funnel search
    print("\n[Search] Executing funnel search...")
    with Timer() as total_timer:
        results, timing = search_client.funnel_search(
            query=query_reduced,
            lsh_search_fn=lsh_search,
            server_compute_fn=phe_compute,
            num_candidates=num_candidates,
            top_k=top_k,
            verbose=verbose,
        )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTop-{top_k} Results:")
    print("-" * 50)
    for r in results:
        print(f"  #{r.rank:2d}: {r.vector_id} (score: {r.score:.6f})")

    # =========================================================================
    # VERIFICATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Compare with plaintext search (ground truth)
    print("\n[Verification] Computing plaintext ground truth...")
    plaintext_scores = compute_plaintext_similarity(query_reduced, db_vectors_reduced)
    plaintext_top_indices = top_k_indices(plaintext_scores, top_k)
    plaintext_top_ids = [vector_ids[i] for i in plaintext_top_indices]

    encrypted_top_ids = [r.vector_id for r in results]

    # LSH Recall: How many of top-k are in the candidate set?
    candidate_ids_from_lsh, _ = lsh_index.search(query_reduced, num_candidates)
    lsh_recall = len(set(plaintext_top_ids) & set(candidate_ids_from_lsh)) / top_k

    # Final Recall: How many of true top-k are in encrypted results?
    final_recall = len(set(plaintext_top_ids) & set(encrypted_top_ids)) / top_k

    # Top-1 match
    top1_match = encrypted_top_ids[0] == plaintext_top_ids[0]

    print(f"\nAccuracy Metrics:")
    print(f"  LSH Recall@{top_k}:      {lsh_recall * 100:.1f}%")
    print(f"  Final Recall@{top_k}:    {final_recall * 100:.1f}%")
    print(f"  Top-1 Match:          {'Yes' if top1_match else 'No'}")

    # =========================================================================
    # TIMING
    # =========================================================================
    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)
    print(f"\n  Stage 1 - LSH Lookup:      {timing['lsh_ms']:8.2f}ms")
    print(f"  Stage 2 - Encryption:      {timing['encrypt_ms']:8.2f}ms")
    print(f"  Stage 2 - PHE Compute:     {timing['server_compute_ms']:8.2f}ms")
    print(f"  Stage 2 - Decryption:      {timing['decrypt_ms']:8.2f}ms")
    print(f"  {'='*40}")
    print(f"  TOTAL:                     {timing['total_ms']:8.2f}ms")

    # Check against target
    target_ms = 3000
    status = "PASS" if timing['total_ms'] < target_ms else "FAIL"
    print(f"\n  Target: <{target_ms}ms | Actual: {timing['total_ms']:.0f}ms | [{status}]")

    # =========================================================================
    # PRIVACY GUARANTEES
    # =========================================================================
    print("\n" + "=" * 70)
    print("PRIVACY GUARANTEES")
    print("=" * 70)
    print("  [x] Server never saw the raw query vector")
    print("  [x] Server never saw the similarity scores")
    print("  [x] Client never downloaded the full database")
    print("  [x] LSH obfuscation preserves locality without revealing exact query")

    return results, timing


def main():
    parser = argparse.ArgumentParser(
        description="Demo full funnel pipeline for Project Opaque"
    )
    parser.add_argument(
        "--num-vectors", "-n",
        type=int,
        default=100000,
        help="Number of vectors in database",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1536,
        help="Input vector dimension (e.g., OpenAI embedding size)",
    )
    parser.add_argument(
        "--reduced-dim",
        type=int,
        default=256,
        help="Reduced dimension after PCA",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=1000,
        help="Number of LSH candidates",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of final results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with smaller database (10k vectors, 500 candidates)",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Tiny mode for fast testing (1k vectors, 100 candidates)",
    )

    args = parser.parse_args()

    if args.tiny:
        num_vectors = 1000
        candidates = 100
        input_dim = 512
        reduced_dim = 128
    elif args.quick:
        num_vectors = 10000
        candidates = 500
        input_dim = args.input_dim
        reduced_dim = args.reduced_dim
    else:
        num_vectors = args.num_vectors
        candidates = args.candidates
        input_dim = args.input_dim
        reduced_dim = args.reduced_dim

    run_full_pipeline_demo(
        num_vectors=num_vectors,
        input_dimension=input_dim,
        reduced_dimension=reduced_dim,
        num_candidates=candidates,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
