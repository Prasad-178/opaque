"""
Client-side cryptographic operations using LightPHE Paillier.

Supports parallel encryption and decryption for improved performance.
"""
from pathlib import Path
from typing import List, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import numpy as np
from lightphe import LightPHE
from lightphe.models.Tensor import EncryptedTensor


class CryptoClient:
    """
    Client-side cryptographic operations.

    Responsible for:
    - Generating Paillier key pairs
    - Encrypting query vectors
    - Decrypting scores from server
    - Exporting/importing keys
    """

    DEFAULT_KEY_SIZE = 2048  # bits (more secure)
    DEFAULT_PRECISION = 5   # decimal places

    def __init__(
        self,
        key_size: int = DEFAULT_KEY_SIZE,
        precision: int = DEFAULT_PRECISION,
        keys: Optional[dict] = None,
        key_file: Optional[str] = None,
    ):
        """
        Initialize crypto client.

        Args:
            key_size: Paillier key size in bits (2048 recommended for security)
            precision: Decimal precision for floating point operations
            keys: Pre-existing key pair dict
            key_file: Path to load keys from
        """
        self.key_size = key_size
        self.precision = precision

        # Initialize LightPHE cryptosystem
        self._cs = LightPHE(
            algorithm_name="Paillier",
            keys=keys,
            key_file=key_file,
            key_size=key_size,
            precision=precision,
        )

    @property
    def public_key(self) -> dict:
        """Get the public key for sharing with server."""
        keys = self._cs.cs.keys.copy()
        # Only return public key portion
        return {"public_key": keys.get("public_key", {})}

    @property
    def has_private_key(self) -> bool:
        """Check if private key is available."""
        return self._cs.cs.keys.get("private_key") is not None

    def encrypt_vector(
        self,
        vector: Union[List[float], np.ndarray],
        silent: bool = True,
    ) -> EncryptedTensor:
        """
        Encrypt a query vector.

        Args:
            vector: Query vector to encrypt
            silent: Suppress progress bar

        Returns:
            EncryptedTensor from LightPHE
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        return self._cs.encrypt(vector, silent=silent)

    def decrypt_score(self, encrypted_score: EncryptedTensor) -> float:
        """
        Decrypt a single encrypted score.

        Args:
            encrypted_score: Encrypted dot product result

        Returns:
            Decrypted similarity score
        """
        if not self.has_private_key:
            raise ValueError("Cannot decrypt without private key")

        result = self._cs.decrypt(encrypted_score)

        # LightPHE returns a list for tensor operations
        if isinstance(result, list):
            return result[0]
        return result

    def decrypt_scores(
        self,
        encrypted_scores: List[EncryptedTensor],
        parallel: bool = False,
        num_workers: Optional[int] = None,
    ) -> List[float]:
        """
        Decrypt multiple encrypted scores.

        Args:
            encrypted_scores: List of encrypted dot products
            parallel: Use parallel decryption (ThreadPoolExecutor)
            num_workers: Number of workers (defaults to CPU count)

        Returns:
            List of decrypted similarity scores
        """
        if not parallel or len(encrypted_scores) <= 4:
            return [self.decrypt_score(score) for score in encrypted_scores]

        # Use ThreadPoolExecutor for parallel decryption
        # ThreadPool works because Python GIL is released during bignum operations
        if num_workers is None:
            num_workers = min(mp.cpu_count(), len(encrypted_scores))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.decrypt_score, encrypted_scores))

        return results

    def decrypt_scores_chunked(
        self,
        encrypted_scores: List[EncryptedTensor],
        chunk_size: int = 10,
        num_workers: Optional[int] = None,
    ) -> List[float]:
        """
        Decrypt scores in parallel chunks for better performance.

        Processes scores in chunks, with each chunk decrypted sequentially
        but chunks processed in parallel. This can be more efficient than
        per-score parallelization due to reduced overhead.

        Args:
            encrypted_scores: List of encrypted dot products
            chunk_size: Number of scores per chunk
            num_workers: Number of parallel workers

        Returns:
            List of decrypted similarity scores
        """
        if len(encrypted_scores) <= chunk_size:
            return [self.decrypt_score(score) for score in encrypted_scores]

        if num_workers is None:
            num_workers = mp.cpu_count()

        # Create chunks
        chunks = [
            encrypted_scores[i:i + chunk_size]
            for i in range(0, len(encrypted_scores), chunk_size)
        ]

        def decrypt_chunk(chunk: List[EncryptedTensor]) -> List[float]:
            return [self.decrypt_score(score) for score in chunk]

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_results = list(executor.map(decrypt_chunk, chunks))

        # Flatten results
        return [score for chunk_result in chunk_results for score in chunk_result]

    def get_keys_dict(self) -> dict:
        """Export keys as a dictionary for multiprocessing."""
        return self._cs.cs.keys.copy()

    def export_public_key(self, path: Union[str, Path]) -> None:
        """Export public key to file."""
        self._cs.export_keys(str(path), public=True)

    def export_private_key(self, path: Union[str, Path]) -> None:
        """Export full key pair (including private key) to file."""
        self._cs.export_keys(str(path), public=False)

    @classmethod
    def from_public_key(
        cls,
        public_key: dict,
        precision: int = DEFAULT_PRECISION,
    ) -> "CryptoClient":
        """
        Create client with only public key (for server-side operations).

        Args:
            public_key: Public key dict
            precision: Decimal precision

        Returns:
            CryptoClient instance without decryption capability
        """
        # Ensure the key format is correct for LightPHE
        if "public_key" in public_key:
            keys = public_key
        else:
            keys = {"public_key": public_key}
        return cls(keys=keys, precision=precision)

    @classmethod
    def from_key_file(
        cls,
        key_file: Union[str, Path],
        precision: int = DEFAULT_PRECISION,
    ) -> "CryptoClient":
        """
        Create client from exported key file.

        Args:
            key_file: Path to key file
            precision: Decimal precision

        Returns:
            CryptoClient instance
        """
        return cls(key_file=str(key_file), precision=precision)

    def get_cryptosystem(self) -> LightPHE:
        """
        Get the underlying LightPHE cryptosystem.

        Useful for advanced operations or debugging.
        """
        return self._cs


# Global worker state for multiprocessing
_worker_crypto: Optional[CryptoClient] = None


def _init_worker(keys: dict, precision: int):
    """Initialize crypto client in worker process."""
    global _worker_crypto
    _worker_crypto = CryptoClient(keys=keys, precision=precision)


def _decrypt_single(encrypted_score: EncryptedTensor) -> float:
    """Decrypt a single score (for use in worker process)."""
    global _worker_crypto
    return _worker_crypto.decrypt_score(encrypted_score)


def _decrypt_chunk(args: Tuple[List[EncryptedTensor], dict, int]) -> List[float]:
    """Decrypt a chunk of scores with own crypto instance."""
    chunk, keys, precision = args
    crypto = CryptoClient(keys=keys, precision=precision)
    return [crypto.decrypt_score(score) for score in chunk]


def parallel_decrypt_multiprocess(
    encrypted_scores: List[EncryptedTensor],
    keys: dict,
    precision: int = 5,
    num_workers: Optional[int] = None,
    chunk_size: int = 5,
) -> List[float]:
    """
    Decrypt scores using true multiprocessing for CPU parallelism.

    Creates separate processes, each with its own crypto instance,
    to bypass Python's GIL.

    Args:
        encrypted_scores: List of encrypted dot products
        keys: Key dictionary from CryptoClient.get_keys_dict()
        precision: Decimal precision
        num_workers: Number of worker processes
        chunk_size: Scores per chunk

    Returns:
        List of decrypted similarity scores
    """
    if len(encrypted_scores) <= chunk_size:
        crypto = CryptoClient(keys=keys, precision=precision)
        return [crypto.decrypt_score(score) for score in encrypted_scores]

    if num_workers is None:
        num_workers = mp.cpu_count()

    # Create chunks with keys/precision for each
    chunks = []
    for i in range(0, len(encrypted_scores), chunk_size):
        chunk = encrypted_scores[i:i + chunk_size]
        chunks.append((chunk, keys, precision))

    # Process in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(_decrypt_chunk, chunks))

    # Flatten
    return [score for chunk_result in chunk_results for score in chunk_result]
