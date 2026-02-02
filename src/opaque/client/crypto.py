"""
Client-side cryptographic operations using LightPHE Paillier.
"""
from pathlib import Path
from typing import List, Optional, Union
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
        encrypted_scores: List[EncryptedTensor]
    ) -> List[float]:
        """
        Decrypt multiple encrypted scores.

        Args:
            encrypted_scores: List of encrypted dot products

        Returns:
            List of decrypted similarity scores
        """
        return [self.decrypt_score(score) for score in encrypted_scores]

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
