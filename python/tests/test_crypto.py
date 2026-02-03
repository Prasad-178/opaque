"""Tests for client crypto module."""
import pytest
import numpy as np
from opaque.client.crypto import CryptoClient


class TestCryptoClient:
    """Test Paillier cryptographic operations."""

    def test_key_generation(self):
        """Test key pair generation."""
        client = CryptoClient(key_size=1024)  # Use smaller key for faster tests
        assert client.has_private_key
        assert "public_key" in client.public_key
        assert "n" in client.public_key["public_key"]

    def test_encrypt_decrypt_vector(self):
        """Test vector encryption and decryption."""
        client = CryptoClient(key_size=1024, precision=5)

        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        encrypted = client.encrypt_vector(vector)

        # Decrypt using internal cryptosystem
        decrypted = client._cs.decrypt(encrypted)

        assert len(decrypted) == len(vector)
        for orig, dec in zip(vector, decrypted):
            assert abs(orig - dec) < 0.01

    def test_encrypted_dot_product(self):
        """Test encrypted dot product accuracy."""
        client = CryptoClient(key_size=1024, precision=5)

        query = [0.5, 0.3, 0.2]
        db_vec = [0.1, 0.4, 0.5]

        expected = sum(q * d for q, d in zip(query, db_vec))

        encrypted_query = client.encrypt_vector(query)
        encrypted_result = encrypted_query @ db_vec

        decrypted = client.decrypt_score(encrypted_result)

        assert abs(expected - decrypted) < 0.01

    def test_public_key_only(self):
        """Test that public-key-only client cannot decrypt."""
        full_client = CryptoClient(key_size=1024)
        public_only = CryptoClient.from_public_key(full_client.public_key)

        assert not public_only.has_private_key

        vector = [0.1, 0.2, 0.3]
        encrypted = public_only.encrypt_vector(vector)

        with pytest.raises(ValueError, match="Cannot decrypt"):
            public_only.decrypt_score(encrypted)

        # But full client can decrypt
        decrypted = full_client._cs.decrypt(encrypted)
        assert len(decrypted) == 3

    def test_decrypt_scores_batch(self):
        """Test batch decryption of multiple scores."""
        client = CryptoClient(key_size=1024, precision=5)

        query = [0.5, 0.3, 0.2]
        db_vecs = [[0.1, 0.4, 0.5], [0.2, 0.3, 0.4], [0.6, 0.1, 0.3]]

        encrypted_query = client.encrypt_vector(query)

        encrypted_scores = [encrypted_query @ vec for vec in db_vecs]
        decrypted_scores = client.decrypt_scores(encrypted_scores)

        for vec, decrypted in zip(db_vecs, decrypted_scores):
            expected = sum(q * d for q, d in zip(query, vec))
            assert abs(expected - decrypted) < 0.01


class TestCryptoClientNormalized:
    """Test with normalized vectors (realistic scenario)."""

    def test_normalized_dot_product(self):
        """Test dot product with normalized vectors."""
        client = CryptoClient(key_size=1024, precision=5)

        # Generate normalized vectors
        query = np.array([0.5, 0.3, 0.2, 0.1])
        query = query / np.linalg.norm(query)

        db_vec = np.array([0.1, 0.4, 0.5, 0.2])
        db_vec = db_vec / np.linalg.norm(db_vec)

        expected = float(np.dot(query, db_vec))

        encrypted_query = client.encrypt_vector(query.tolist())
        encrypted_result = encrypted_query @ db_vec.tolist()
        decrypted = client.decrypt_score(encrypted_result)

        assert abs(expected - decrypted) < 0.01
