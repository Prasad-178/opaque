"""Tests for LSH index."""
import pytest
import numpy as np
from opaque.server.index import LSHIndex, RandomProjectionLSH
from opaque.shared.utils import generate_random_vectors


class TestLSHIndex:
    """Test LSH indexing."""

    def test_add_and_search(self):
        """Test basic add and search."""
        dim = 128
        index = LSHIndex(dim, nbits=128, use_faiss=False)

        vectors = generate_random_vectors(100, dim, seed=42)
        ids = [f"vec_{i}" for i in range(100)]
        index.add(vectors, ids)

        assert index.ntotal == 100

        # Search with one of the vectors (should find itself)
        results, _ = index.search(vectors[0], k=5)
        assert "vec_0" in results

    def test_recall(self):
        """Test that LSH achieves reasonable recall."""
        dim = 128
        n_vectors = 1000
        k = 100

        index = LSHIndex(dim, nbits=256, use_faiss=False, seed=42)
        vectors = generate_random_vectors(n_vectors, dim, seed=42)
        index.add(vectors)

        query = generate_random_vectors(1, dim, seed=123)[0]

        # Ground truth: exact nearest neighbors by L2 distance
        distances = np.linalg.norm(vectors - query, axis=1)
        true_top_k = set(np.argsort(distances)[:k])

        # LSH results
        lsh_results, _ = index.search(query, k)
        lsh_indices = set(int(r.split("_")[1]) for r in lsh_results)

        recall = len(true_top_k & lsh_indices) / k
        # LSH should have at least some recall
        assert recall > 0.1, f"LSH recall too low: {recall}"

    def test_search_k_larger_than_db(self):
        """Test search when k > number of vectors."""
        dim = 64
        index = LSHIndex(dim, nbits=64, use_faiss=False)

        vectors = generate_random_vectors(50, dim, seed=42, positive=False)
        index.add(vectors)

        query = generate_random_vectors(1, dim, seed=123, positive=False)[0]
        results, _ = index.search(query, k=100)

        # Should return all 50 vectors
        assert len(results) == 50

    def test_incremental_add(self):
        """Test adding vectors incrementally."""
        dim = 64
        index = LSHIndex(dim, nbits=64, use_faiss=False)

        vectors1 = generate_random_vectors(50, dim, seed=42)
        ids1 = [f"batch1_{i}" for i in range(50)]
        index.add(vectors1, ids1)
        assert index.ntotal == 50

        vectors2 = generate_random_vectors(30, dim, seed=43)
        ids2 = [f"batch2_{i}" for i in range(30)]
        index.add(vectors2, ids2)
        assert index.ntotal == 80


class TestRandomProjectionLSH:
    """Test pure Python LSH implementation."""

    def test_hash_consistency(self):
        """Test that hashing is consistent."""
        lsh = RandomProjectionLSH(128, nbits=64, seed=42)

        vector = np.random.randn(128).astype(np.float32)
        hash1 = lsh._hash(vector.reshape(1, -1))
        hash2 = lsh._hash(vector.reshape(1, -1))

        assert np.array_equal(hash1, hash2)

    def test_similar_vectors_similar_hash(self):
        """Test that similar vectors have similar hashes."""
        lsh = RandomProjectionLSH(128, nbits=256, seed=42)

        v1 = np.random.randn(128).astype(np.float32)
        v2 = v1 + 0.01 * np.random.randn(128).astype(np.float32)  # Very similar
        v3 = np.random.randn(128).astype(np.float32)  # Random (likely different)

        h1 = lsh._hash(v1.reshape(1, -1))[0]
        h2 = lsh._hash(v2.reshape(1, -1))[0]
        h3 = lsh._hash(v3.reshape(1, -1))[0]

        dist_similar = lsh._hamming_distance(h1.reshape(1, -1), h2)[0]
        dist_random = lsh._hamming_distance(h1.reshape(1, -1), h3)[0]

        # Similar vectors should have lower Hamming distance
        assert dist_similar < dist_random

    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        lsh = RandomProjectionLSH(64, nbits=8, seed=42)

        h1 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        h2 = np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)

        dist = lsh._hamming_distance(h1.reshape(1, -1), h2)[0]
        # Positions 1, 4, 7 differ -> distance = 3
        assert dist == 3


class TestFaissLSHIndex:
    """Test Faiss LSH implementation if available."""

    def test_faiss_backend_available(self):
        """Test that Faiss backend works."""
        try:
            index = LSHIndex(64, nbits=64, use_faiss=True)
            assert index.backend == "faiss"
        except ImportError:
            pytest.skip("Faiss not installed")

    def test_faiss_search(self):
        """Test Faiss LSH search."""
        try:
            dim = 64
            index = LSHIndex(dim, nbits=64, use_faiss=True)

            vectors = generate_random_vectors(100, dim, seed=42)
            index.add(vectors)

            query = vectors[0]
            results, _ = index.search(query, k=5)

            # First result should be the query itself
            assert "vec_0" in results
        except ImportError:
            pytest.skip("Faiss not installed")
