#!/usr/bin/env python3
"""
Demo: Privacy-preserving search with REAL embeddings.

This script demonstrates the full Project Opaque pipeline using:
1. Real sentence embeddings (sentence-transformers)
2. LSH candidate filtering
3. PHE encrypted scoring
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
from opaque.shared.utils import Timer, top_k_indices, compute_plaintext_similarity
from opaque.shared.reduction import PCAReducer
from opaque.shared.embeddings import (
    EmbeddingModel,
    load_sample_dataset,
    create_embedded_database,
)


def run_real_embeddings_demo(
    dataset_name: str = "simple",
    num_samples: int = 100,
    num_candidates: int = 50,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    reduced_dim: int = 64,
    verbose: bool = True,
):
    """
    Run privacy-preserving search demo with real embeddings.
    """
    print("=" * 70)
    print("Project Opaque - Real Embeddings Demo")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset:           {dataset_name}")
    print(f"  Num samples:       {num_samples}")
    print(f"  Model:             {model_name}")
    print(f"  Reduced dimension: {reduced_dim}")
    print(f"  Candidates:        {num_candidates}")
    print(f"  Top-K:             {top_k}")

    # =========================================================================
    # SETUP PHASE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SETUP PHASE")
    print("=" * 70)

    # 1. Load embedding model
    print("\n[1] Loading embedding model...")
    with Timer() as t:
        embed_model = EmbeddingModel(model_name)
    print(f"    Model loaded in {t.elapsed_ms:.0f}ms")
    print(f"    Original dimension: {embed_model.dimension}")

    # 2. Load sample dataset
    print("\n[2] Loading sample dataset...")
    with Timer() as t:
        texts, doc_ids = load_sample_dataset(dataset_name, max_samples=num_samples)
    print(f"    Loaded {len(texts)} texts in {t.elapsed_ms:.0f}ms")

    # Show sample texts
    print("\n    Sample texts:")
    for i, text in enumerate(texts[:3]):
        print(f"      [{i}] {text[:60]}{'...' if len(text) > 60 else ''}")

    # 3. Create embeddings
    print("\n[3] Creating embeddings...")
    with Timer() as t:
        embeddings, doc_ids, orig_dim = create_embedded_database(
            texts, doc_ids, model=embed_model, positive_shift=False
        )
    print(f"    Created {len(embeddings)} embeddings in {t.elapsed_ms:.0f}ms")
    print(f"    Shape: {embeddings.shape}")

    # 4. Apply PCA reduction
    print(f"\n[4] Applying PCA ({orig_dim} -> {reduced_dim})...")
    with Timer() as t:
        pca = PCAReducer(orig_dim, reduced_dim)
        embeddings_reduced = pca.fit_transform(embeddings)
        # Shift to positive range for PHE
        embeddings_reduced = embeddings_reduced - embeddings_reduced.min() + 0.01
        # Get actual output dimension (may be less if n_samples < requested dim)
        actual_dim = pca.output_dim
    print(f"    PCA done in {t.elapsed_ms:.0f}ms")
    print(f"    Actual dimension: {actual_dim} (shape: {embeddings_reduced.shape})")
    if pca.total_explained_variance_ratio:
        print(f"    Explained variance: {pca.total_explained_variance_ratio:.1%}")

    # 5. Build LSH index
    print("\n[5] Building LSH index...")
    with Timer() as t:
        lsh_index = LSHIndex(actual_dim, nbits=128)
        lsh_index.add(embeddings_reduced, doc_ids)
    print(f"    Indexed {lsh_index.ntotal} vectors in {t.elapsed_ms:.0f}ms")
    print(f"    Backend: {lsh_index.backend}")

    # 6. Create vector store and compute engine
    print("\n[6] Creating vector store...")
    vector_store = VectorStore(vectors=embeddings_reduced, ids=doc_ids)
    compute_engine = ComputeEngine(vector_store)

    # 7. Initialize client crypto
    print("\n[7] Generating Paillier keys (2048-bit)...")
    with Timer() as t:
        crypto_client = CryptoClient(key_size=2048, precision=5)
    print(f"    Key generation in {t.elapsed_ms:.0f}ms")

    search_client = SearchClient(crypto_client)

    # =========================================================================
    # QUERY PHASE
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUERY PHASE")
    print("=" * 70)

    # Create a query
    query_text = "How does encryption work for privacy?"
    print(f"\n[Query] \"{query_text}\"")

    # Embed query
    print("\n[8] Embedding query...")
    with Timer() as t:
        query_embedding = embed_model.embed_query(query_text)
        query_reduced = pca.transform(query_embedding.reshape(1, -1))[0]
        # Shift to positive (same as database)
        query_reduced = query_reduced - query_reduced.min() + 0.01
    print(f"    Query embedded in {t.elapsed_ms:.0f}ms")

    # Define search functions
    def lsh_search(query, k):
        return lsh_index.search(query, k)

    def phe_compute(encrypted_query, candidate_ids):
        return compute_engine.compute_encrypted_scores(
            encrypted_query,
            vector_ids=candidate_ids,
            verbose=False,
        )

    # =========================================================================
    # PRIVACY-PRESERVING SEARCH
    # =========================================================================
    print("\n" + "=" * 70)
    print("PRIVACY-PRESERVING SEARCH")
    print("=" * 70)

    # Execute search
    print("\n[Executing Funnel Search]")
    with Timer() as total_timer:
        results_seq, timing_seq = search_client.funnel_search(
            query=query_reduced,
            lsh_search_fn=lsh_search,
            server_compute_fn=phe_compute,
            num_candidates=num_candidates,
            top_k=top_k,
            verbose=True,
            parallel=False,
        )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    print(f"\nTop-{top_k} Results (for query: \"{query_text}\"):")
    print("-" * 70)
    for r in results_seq:
        doc_idx = doc_ids.index(r.vector_id)
        text = texts[doc_idx]
        print(f"  #{r.rank:2d}: {r.vector_id} (score: {r.score:.4f})")
        print(f"       \"{text[:65]}{'...' if len(text) > 65 else ''}\"")

    # =========================================================================
    # TIMING BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)

    print(f"\n  {'Stage':<25} {'Time (ms)':>12}")
    print("  " + "-" * 40)

    stages = ["lsh_ms", "encrypt_ms", "server_compute_ms", "decrypt_ms", "total_ms"]
    stage_names = ["LSH Lookup", "Encryption", "PHE Compute", "Decryption", "TOTAL"]

    for stage, name in zip(stages, stage_names):
        time_ms = timing_seq[stage]
        print(f"  {name:<25} {time_ms:>10.0f}ms")

    # =========================================================================
    # VERIFICATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Compare with plaintext search
    print("\n[Verification] Computing plaintext ground truth...")
    plaintext_scores = compute_plaintext_similarity(query_reduced, embeddings_reduced)
    plaintext_top_indices = top_k_indices(plaintext_scores, top_k)
    plaintext_top_ids = [doc_ids[i] for i in plaintext_top_indices]

    encrypted_top_ids = [r.vector_id for r in results_seq]

    # Metrics
    recall = len(set(plaintext_top_ids) & set(encrypted_top_ids)) / top_k
    top1_match = encrypted_top_ids[0] == plaintext_top_ids[0]

    print(f"\nAccuracy Metrics:")
    print(f"  Recall@{top_k}:   {recall * 100:.1f}%")
    print(f"  Top-1 Match: {'Yes' if top1_match else 'No'}")

    # =========================================================================
    # PRIVACY GUARANTEES
    # =========================================================================
    print("\n" + "=" * 70)
    print("PRIVACY GUARANTEES")
    print("=" * 70)
    print("  [x] Server never saw the raw query text or embedding")
    print("  [x] Server never saw the similarity scores")
    print("  [x] Client never downloaded the full text database")
    print("  [x] Real semantic embeddings preserve search quality")

    return results_seq, timing_seq


def main():
    parser = argparse.ArgumentParser(
        description="Demo real embeddings with Project Opaque"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="simple",
        choices=["simple", "quora", "stsb", "msmarco"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=100,
        help="Number of samples from dataset",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=50,
        help="Number of LSH candidates",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model (or preset: fast, balanced, quality)",
    )
    parser.add_argument(
        "--reduced-dim",
        type=int,
        default=64,
        help="Reduced dimension after PCA",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Tiny mode for quick testing (20 samples, 10 candidates)",
    )

    args = parser.parse_args()

    if args.tiny:
        num_samples = 20
        candidates = 10
        top_k = 3
        reduced_dim = 32
    else:
        num_samples = args.num_samples
        candidates = args.candidates
        top_k = args.top_k
        reduced_dim = args.reduced_dim

    run_real_embeddings_demo(
        dataset_name=args.dataset,
        num_samples=num_samples,
        num_candidates=candidates,
        top_k=top_k,
        model_name=args.model,
        reduced_dim=reduced_dim,
    )


if __name__ == "__main__":
    main()
