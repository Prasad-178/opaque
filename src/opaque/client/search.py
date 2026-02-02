"""
Client-side search orchestration.

Coordinates the full search flow:
1. Encrypt query
2. Send to server (via function call or API)
3. Receive encrypted scores
4. Decrypt and rank results
"""
import numpy as np
from typing import List, Tuple, Callable, Any

from opaque.client.crypto import CryptoClient
from opaque.shared.utils import Timer, top_k_indices
from opaque.shared.protocol import SearchResult


class SearchClient:
    """
    Client-side search coordinator.

    Handles the full search lifecycle while preserving privacy.
    """

    def __init__(self, crypto_client: CryptoClient):
        """
        Initialize search client.

        Args:
            crypto_client: Cryptographic client for encryption/decryption
        """
        self.crypto = crypto_client

    def blind_search(
        self,
        query: np.ndarray,
        server_compute_fn: Callable,
        top_k: int = 10,
        verbose: bool = False,
    ) -> Tuple[List[SearchResult], dict]:
        """
        Perform blind search (Phase 2 - no LSH filtering).

        The server never sees the query vector or the final scores.

        Args:
            query: Query vector (numpy array)
            server_compute_fn: Function that computes encrypted scores
                               Signature: (encrypted_query) -> (encrypted_scores, ids, time_ms)
            top_k: Number of top results to return
            verbose: Print timing information

        Returns:
            Tuple of (search results, timing info)
        """
        timing = {}

        # Step 1: Encrypt query
        if verbose:
            print("Step 1: Encrypting query...")
        with Timer() as t:
            encrypted_query = self.crypto.encrypt_vector(query, silent=True)
        timing["encrypt_ms"] = t.elapsed_ms
        if verbose:
            print(f"  Encryption took {t.elapsed_ms:.2f}ms")

        # Step 2: Server computes encrypted scores
        if verbose:
            print("Step 2: Server computing encrypted scores...")
        encrypted_scores, vector_ids, server_time_ms = server_compute_fn(encrypted_query)
        timing["server_compute_ms"] = server_time_ms
        if verbose:
            print(f"  Server computation took {server_time_ms:.2f}ms")

        # Step 3: Decrypt scores
        if verbose:
            print("Step 3: Decrypting scores...")
        with Timer() as t:
            scores = self.crypto.decrypt_scores(encrypted_scores)
        timing["decrypt_ms"] = t.elapsed_ms
        if verbose:
            print(f"  Decryption took {t.elapsed_ms:.2f}ms")

        # Step 4: Rank and return top-k
        if verbose:
            print("Step 4: Ranking results...")
        scores_array = np.array(scores)
        top_indices = top_k_indices(scores_array, top_k)

        results = [
            SearchResult(
                vector_id=vector_ids[i],
                score=scores[i],
                rank=rank + 1,
            )
            for rank, i in enumerate(top_indices)
        ]

        timing["total_ms"] = sum([
            timing["encrypt_ms"],
            timing["server_compute_ms"],
            timing["decrypt_ms"],
        ])

        if verbose:
            print(f"\nTotal time: {timing['total_ms']:.2f}ms")

        return results, timing

    def funnel_search(
        self,
        query: np.ndarray,
        lsh_search_fn: Callable,
        server_compute_fn: Callable,
        num_candidates: int = 1000,
        top_k: int = 10,
        verbose: bool = False,
    ) -> Tuple[List[SearchResult], dict]:
        """
        Perform full funnel search (Phase 3 - with LSH filtering).

        Stage 1 (Coarse): LSH finds candidates
        Stage 2 (Fine): PHE scores candidates

        Args:
            query: Query vector
            lsh_search_fn: Function for LSH candidate retrieval
                           Signature: (query, k) -> (candidate_ids, distances)
            server_compute_fn: Function for encrypted scoring
                               Signature: (encrypted_query, candidate_ids) -> (scores, ids, time)
            num_candidates: Number of LSH candidates
            top_k: Final number of results
            verbose: Print progress

        Returns:
            Tuple of (results, timing)
        """
        timing = {}

        # Stage 1: LSH Candidate Retrieval (Coarse)
        if verbose:
            print("Stage 1: LSH Candidate Retrieval...")
        with Timer() as t:
            candidate_ids, lsh_distances = lsh_search_fn(query, num_candidates)
        timing["lsh_ms"] = t.elapsed_ms
        if verbose:
            print(f"  Retrieved {len(candidate_ids)} candidates in {t.elapsed_ms:.2f}ms")

        # Stage 2: PHE Scoring (Fine)
        if verbose:
            print("Stage 2: Encrypting query...")
        with Timer() as t:
            encrypted_query = self.crypto.encrypt_vector(query, silent=True)
        timing["encrypt_ms"] = t.elapsed_ms

        if verbose:
            print(f"Stage 2: Computing encrypted scores for {len(candidate_ids)} candidates...")
        encrypted_scores, scored_ids, server_time = server_compute_fn(
            encrypted_query, candidate_ids
        )
        timing["server_compute_ms"] = server_time

        if verbose:
            print("Stage 2: Decrypting scores...")
        with Timer() as t:
            scores = self.crypto.decrypt_scores(encrypted_scores)
        timing["decrypt_ms"] = t.elapsed_ms

        # Rank and return
        scores_array = np.array(scores)
        top_indices = top_k_indices(scores_array, top_k)

        results = [
            SearchResult(
                vector_id=scored_ids[i],
                score=scores[i],
                rank=rank + 1,
            )
            for rank, i in enumerate(top_indices)
        ]

        timing["total_ms"] = (
            timing["lsh_ms"] +
            timing["encrypt_ms"] +
            timing["server_compute_ms"] +
            timing["decrypt_ms"]
        )

        if verbose:
            print(f"\nTotal time: {timing['total_ms']:.2f}ms")

        return results, timing

    def verify_accuracy(
        self,
        query: np.ndarray,
        database_vectors: np.ndarray,
        database_ids: List[str],
        encrypted_results: List[SearchResult],
        top_k: int = 10,
    ) -> dict:
        """
        Verify encrypted search matches plaintext search.

        For testing/validation only - in production, client never has database.

        Args:
            query: Original query vector
            database_vectors: Full database (for verification)
            database_ids: Vector IDs
            encrypted_results: Results from blind search
            top_k: Number of results to compare

        Returns:
            Accuracy metrics
        """
        from opaque.shared.utils import compute_plaintext_similarity

        # Compute plaintext scores
        plaintext_scores = compute_plaintext_similarity(query, database_vectors)
        plaintext_top_indices = top_k_indices(plaintext_scores, top_k)
        plaintext_top_ids = [database_ids[i] for i in plaintext_top_indices]

        # Get encrypted top IDs
        encrypted_top_ids = [r.vector_id for r in encrypted_results[:top_k]]

        # Compute metrics
        # Recall@k: how many of the true top-k are in encrypted top-k
        recall = len(set(plaintext_top_ids) & set(encrypted_top_ids)) / top_k

        # Check if top-1 matches
        top1_match = encrypted_top_ids[0] == plaintext_top_ids[0] if encrypted_top_ids else False

        # Score difference for top result
        encrypted_top_score = encrypted_results[0].score if encrypted_results else 0
        plaintext_top_score = plaintext_scores[plaintext_top_indices[0]]
        score_diff = abs(encrypted_top_score - plaintext_top_score)

        return {
            "recall_at_k": recall,
            "top1_match": top1_match,
            "top1_score_diff": score_diff,
            "plaintext_top_ids": plaintext_top_ids,
            "encrypted_top_ids": encrypted_top_ids,
        }
