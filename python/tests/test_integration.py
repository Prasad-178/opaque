"""Integration tests for the full pipeline."""
import pytest
import numpy as np
from opaque.client.crypto import CryptoClient
from opaque.client.search import SearchClient
from opaque.server.compute import ComputeEngine, VectorStore, create_mock_database
from opaque.server.index import LSHIndex
from opaque.shared.utils import generate_random_vectors
from opaque.shared.reduction import PCAReducer


class TestBlindSearch:
    """Test Phase 2: Blind search without LSH."""

    def test_blind_search_accuracy(self):
        """Test that blind search returns correct results."""
        dim = 64
        n_vectors = 50

        # Setup
        vector_store = create_mock_database(n_vectors, dim, seed=42)
        compute = ComputeEngine(vector_store)
        crypto = CryptoClient(key_size=1024, precision=5)
        search = SearchClient(crypto)

        query = generate_random_vectors(1, dim, seed=123)[0]

        def server_compute(enc_q):
            return compute.compute_encrypted_scores(enc_q)

        results, timing = search.blind_search(query, server_compute, top_k=5)

        # Verify top result matches plaintext
        accuracy = search.verify_accuracy(
            query, vector_store.vectors, vector_store.ids, results, top_k=5
        )

        assert accuracy["recall_at_k"] >= 0.6  # Relaxed for faster tests
        assert accuracy["top1_match"]
        assert len(results) == 5

    def test_blind_search_timing(self):
        """Test that timing information is captured."""
        dim = 32
        n_vectors = 20

        vector_store = create_mock_database(n_vectors, dim, seed=42)
        compute = ComputeEngine(vector_store)
        crypto = CryptoClient(key_size=1024, precision=5)
        search = SearchClient(crypto)

        query = generate_random_vectors(1, dim, seed=123)[0]

        def server_compute(enc_q):
            return compute.compute_encrypted_scores(enc_q)

        _, timing = search.blind_search(query, server_compute, top_k=5)

        assert "encrypt_ms" in timing
        assert "server_compute_ms" in timing
        assert "decrypt_ms" in timing
        assert "total_ms" in timing
        assert timing["total_ms"] > 0


class TestFunnelSearch:
    """Test Phase 3: Full funnel with LSH + PHE."""

    def test_funnel_search(self):
        """Test the full funnel pipeline."""
        dim = 64
        n_vectors = 500
        n_candidates = 100

        # Setup
        vectors = generate_random_vectors(n_vectors, dim, seed=42)
        ids = [f"doc_{i}" for i in range(n_vectors)]

        # LSH index
        lsh = LSHIndex(dim, nbits=128, use_faiss=False, seed=42)
        lsh.add(vectors, ids)

        # Vector store
        store = VectorStore(vectors=vectors, ids=ids)
        compute = ComputeEngine(store)

        # Client
        crypto = CryptoClient(key_size=1024, precision=5)
        search = SearchClient(crypto)

        query = generate_random_vectors(1, dim, seed=999)[0]

        def lsh_search(q, k):
            return lsh.search(q, k)

        def phe_compute(enc_q, cand_ids):
            return compute.compute_encrypted_scores(enc_q, vector_ids=cand_ids)

        results, timing = search.funnel_search(
            query, lsh_search, phe_compute,
            num_candidates=n_candidates, top_k=10
        )

        assert len(results) == 10
        assert timing["total_ms"] > 0
        assert timing["lsh_ms"] >= 0
        assert timing["server_compute_ms"] > 0

    def test_funnel_with_pca(self):
        """Test funnel search with PCA dimensionality reduction."""
        input_dim = 256
        reduced_dim = 64
        n_vectors = 200
        n_candidates = 50

        # Generate high-dim vectors (positive for PHE compatibility)
        vectors_full = generate_random_vectors(n_vectors, input_dim, seed=42, positive=True)

        # Apply PCA
        pca = PCAReducer(input_dim, reduced_dim)
        vectors_reduced = pca.fit_transform(vectors_full)
        # Shift to positive after PCA (PCA can produce negative values)
        vectors_reduced = vectors_reduced - vectors_reduced.min() + 0.01

        ids = [f"doc_{i}" for i in range(n_vectors)]

        # Build index on reduced vectors
        lsh = LSHIndex(reduced_dim, nbits=64, use_faiss=False, seed=42)
        lsh.add(vectors_reduced, ids)

        store = VectorStore(vectors=vectors_reduced, ids=ids)
        compute = ComputeEngine(store)

        crypto = CryptoClient(key_size=1024, precision=5)
        search = SearchClient(crypto)

        # Query
        query_full = generate_random_vectors(1, input_dim, seed=999, positive=True)[0]
        query_reduced = pca.transform(query_full.reshape(1, -1))[0]
        # Shift to positive
        query_reduced = query_reduced - query_reduced.min() + 0.01

        def lsh_search(q, k):
            return lsh.search(q, k)

        def phe_compute(enc_q, cand_ids):
            return compute.compute_encrypted_scores(enc_q, vector_ids=cand_ids)

        results, timing = search.funnel_search(
            query_reduced, lsh_search, phe_compute,
            num_candidates=n_candidates, top_k=5
        )

        assert len(results) == 5


class TestPCAReducer:
    """Test PCA dimensionality reduction."""

    def test_fit_transform(self):
        """Test PCA fit and transform."""
        input_dim = 128
        output_dim = 32
        n_samples = 100

        vectors = generate_random_vectors(n_samples, input_dim, seed=42)

        pca = PCAReducer(input_dim, output_dim)
        reduced = pca.fit_transform(vectors)

        assert reduced.shape == (n_samples, output_dim)
        assert pca._is_fitted

    def test_explained_variance(self):
        """Test explained variance calculation."""
        input_dim = 64
        output_dim = 16
        n_samples = 200

        vectors = generate_random_vectors(n_samples, input_dim, seed=42)

        pca = PCAReducer(input_dim, output_dim)
        pca.fit(vectors)

        assert pca.explained_variance_ratio is not None
        assert len(pca.explained_variance_ratio) == output_dim
        assert pca.total_explained_variance_ratio is not None
        assert 0 < pca.total_explained_variance_ratio <= 1

    def test_transform_consistency(self):
        """Test that transform is consistent."""
        input_dim = 64
        output_dim = 16
        n_samples = 100

        vectors = generate_random_vectors(n_samples, input_dim, seed=42)

        pca = PCAReducer(input_dim, output_dim)
        pca.fit(vectors)

        # Same vector should always produce same result
        v = vectors[0]
        result1 = pca.transform(v.reshape(1, -1))
        result2 = pca.transform(v.reshape(1, -1))

        np.testing.assert_array_almost_equal(result1, result2)

    def test_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        pca = PCAReducer(input_dim=128, output_dim=32)

        wrong_dim_vectors = generate_random_vectors(50, 64, seed=42)

        with pytest.raises(ValueError, match="Expected 128 dims"):
            pca.fit(wrong_dim_vectors)
